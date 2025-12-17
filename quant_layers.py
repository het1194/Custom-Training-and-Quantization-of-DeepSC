import os
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transceiver import DeepSC 
from dataset import EurDataset, collate_data
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm

class ZeroLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        print(f"    [Module Init] ZeroLinear ({self.in_features}, {self.out_features}) created.")

    def forward(self, x):
        out_shape = list(x.shape[:-1]) + [self.out_features]
        return x.new_zeros(*out_shape)

    def __repr__(self):
        return f"ZeroLinear(in_features={self.in_features}, out_features={self.out_features}) [PRUNED]"

class MinMaxObserver:
    """Helper class from quant.py"""
    def __init__(self):
        self.min_val, self.max_val = None, None
    def observe(self, x: torch.Tensor):
        x = x.detach(); 
        if x.numel() == 0: return
        mn, mx = float(x.min()), float(x.max())
        self.min_val = mn if self.min_val is None else min(self.min_val, mn)
        self.max_val = mx if self.max_val is None else max(self.max_val, mx)
    def get_qparams(self, num_bits=8, symmetric=False, eps=1e-12):
        if self.min_val is None or self.max_val is None: return 1.0, 0
        qmin, qmax = (-(2**(num_bits-1)), 2**(num_bits-1)-1) if symmetric else (0, 2**num_bits-1)
        mn, mx = self.min_val, self.max_val
        if mx - mn < eps: return 1.0, int((qmin + qmax) // 2)
        scale = (mx - mn) / (qmax - qmin); scale = max(scale, 1e-6)
        if symmetric: zp = 0
        else:
            zp = int(round(qmin - mn / (scale + 1e-12)))
            zp = max(qmin, min(qmax, zp))
        return float(scale), int(zp)

class QuantLinear(nn.Module):
    def __init__(self, fp_linear: nn.Linear = None, name: str = ""):
        super().__init__()
        self.name = name

        if fp_linear is not None:
            self.in_features = fp_linear.in_features
            self.out_features = fp_linear.out_features
        
            self.has_bias = fp_linear.bias is not None
        else:
       
            self.in_features = 0
            self.out_features = 0
            self.has_bias = False
      
        self.register_buffer("int_weight", torch.zeros(self.out_features, self.in_features, dtype=torch.int8))
        self.register_buffer("w_scale", torch.tensor(0.0))
        self.register_buffer("w_zp", torch.tensor(0))
        
        if self.has_bias:
            self.register_buffer("fp_bias", torch.zeros(self.out_features))
        else:
            self.register_buffer("fp_bias", None) 
        
        # float copy 
        self.fp_weight = None 
        
        # observer
        self.act_obs = None 
        self.calibrating = False 
       

    def forward(self, x: torch.Tensor):
        if self.calibrating:
       
            print(f"WARNING: QuantLinear {self.name} is still calibrating!")
            self.act_obs.observe(x.detach())
        
        if self.int_weight.numel() == 0:
   
            raise RuntimeError(f"QuantLinear {self.name} has zero elements in 'int_weight'. Was it loaded?")

        w = self.int_weight.float() * float(self.w_scale)
        out = F.linear(x, w, self.fp_bias) 
        return out

    def is_packed(self):
        return (self.int_weight is not None) and (float(self.w_scale) != 0.0)

    def __repr__(self):
        return f"QuantLinear(name={self.name}, in={self.in_features}, out={self.out_features}, packed={self.is_packed()})"

SKIP_PARAM_NAMES = {
    "encoder.embedding.weight",
    "encoder.enc_layers.0.layernorm1.weight", "encoder.enc_layers.0.layernorm2.weight",
    "encoder.enc_layers.1.layernorm1.weight", "encoder.enc_layers.1.layernorm2.weight",
    "encoder.enc_layers.2.layernorm1.weight", "encoder.enc_layers.2.layernorm2.weight",
    "encoder.enc_layers.3.layernorm1.weight", "encoder.enc_layers.3.layernorm2.weight",
    "channel_encoder.0.weight", "channel_encoder.2.weight",
    "channel_decoder.linear1.weight", "channel_decoder.layernorm.weight",
    "decoder.embedding.weight",
    "decoder.dec_layers.0.layernorm1.weight", "decoder.dec_layers.0.layernorm2.weight", "decoder.dec_layers.0.layernorm3.weight",
    "decoder.dec_layers.1.layernorm1.weight", "decoder.dec_layers.1.layernorm2.weight", "decoder.dec_layers.1.layernorm3.weight",
    "decoder.dec_layers.2.layernorm1.weight", "decoder.dec_layers.2.layernorm2.weight", "decoder.dec_layers.2.layernorm3.weight",
    "decoder.dec_layers.3.layernorm1.weight", "decoder.dec_layers.3.layernorm2.weight", "decoder.dec_layers.3.layernorm3.weight",
}
SKIP_SUBSTR = ("embedding", "layernorm", "pos_encoding", "softmax")

def should_skip_module_path(module_path: str) -> bool:
    low = module_path.lower()
    return any(s in low for s in SKIP_SUBSTR)

def should_skip_by_param_name(module_path: str, skip_param_names: set) -> bool:
    return (module_path + ".weight") in skip_param_names

def _get_parent_and_attr(root, fullname):

    parts = fullname.split('.')
    if len(parts) == 1:
        return root, parts[0]
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def replace_linear_with_quant(module: nn.Module, parent_path: str = ""):
    replaced = []
    for child_name, child in list(module.named_children()):
        child_path = f"{parent_path}.{child_name}" if parent_path else child_name
        
        if not isinstance(child, nn.Linear):
            replaced.extend(replace_linear_with_quant(child, child_path))
            continue
            
        if should_skip_module_path(child_path) or should_skip_by_param_name(child_path, SKIP_PARAM_NAMES):
            print(f"  [Quant Replace] SKIP: {child_path}")
            continue
            
        if isinstance(child, (ZeroLinear, QuantLinear)):
            print(f"  [Quant Replace] Already custom, skipping: {child_path}")
            continue

        qlin = QuantLinear(child, name=child_path) 
        setattr(module, child_name, qlin)
        replaced.append(child_path)
    return replaced

