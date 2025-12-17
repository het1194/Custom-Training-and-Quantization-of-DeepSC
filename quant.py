# Setup 
import os, json, torch
import torch.nn as nn
# file checks 
ROOT = "./"
CKPT_PATH = os.path.join(ROOT, "checkpoint_237.pth")
VOCAB_PATH = os.path.join(ROOT, "snli_vocab.json")
TRAIN_PKL = os.path.join(ROOT, "train.pkl")
TEST_PKL  = os.path.join(ROOT, "test.pkl")

for p in [CKPT_PATH, VOCAB_PATH, TRAIN_PKL, TEST_PKL, os.path.join(ROOT, "transceiver.py")]:
    print(f"{p}: {'OK' if os.path.exists(p) else 'MISSING'}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# load state_dict (weights-only)
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
assert isinstance(ckpt, dict), "Expected a raw state_dict dict in checkpoint_237.pth."

def shape_of(key): return tuple(ckpt[key].shape)
d_model = shape_of("encoder.enc_layers.0.mha.wq.weight")[0]
max_len = shape_of("encoder.pos_encoding.pe")[1]
vocab_from_emb, _ = shape_of("encoder.embedding.weight")
print(f"Inferred d_model={d_model}, max_len={max_len}, vocab_from_emb={vocab_from_emb}")

with open(VOCAB_PATH, "rb") as f:
    vocab = json.load(f)
token_to_idx = vocab["token_to_idx"]
vocab_size = len(token_to_idx)
print(f"Vocab size from file = {vocab_size}")
assert vocab_size == vocab_from_emb, "Vocab size mismatch vs checkpoint embedding!"
#load FP32 model
from transceiver import DeepSC
num_layers, num_heads, dff, dropout = 4, 8, 512, 0.1
net_fp32 = DeepSC(
    num_layers, vocab_size, vocab_size,
    max_len, max_len,
    d_model, num_heads, dff, dropout
).to(device)

missing, unexpected = net_fp32.load_state_dict(ckpt, strict=False)
print("Loaded state_dict. Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
net_fp32.eval()
print("FP32 model ready.")

# quantizing these is a bad idea, so, skipping 
SKIP_PARAM_NAMES = {
    "encoder.embedding.weight",
    "encoder.enc_layers.0.layernorm1.weight",
    "encoder.enc_layers.0.layernorm2.weight",
    "encoder.enc_layers.1.layernorm1.weight",
    "encoder.enc_layers.1.layernorm2.weight",
    "encoder.enc_layers.2.layernorm1.weight",
    "encoder.enc_layers.2.layernorm2.weight",
    "encoder.enc_layers.3.layernorm1.weight",
    "encoder.enc_layers.3.layernorm2.weight",
    "channel_encoder.0.weight",
    "channel_encoder.2.weight",
    "channel_decoder.linear1.weight",
    "channel_decoder.layernorm.weight",
    "decoder.embedding.weight",
    "decoder.dec_layers.0.layernorm1.weight",
    "decoder.dec_layers.0.layernorm2.weight",
    "decoder.dec_layers.0.layernorm3.weight",
    "decoder.dec_layers.1.layernorm1.weight",
    "decoder.dec_layers.1.layernorm2.weight",
    "decoder.dec_layers.1.layernorm3.weight",
    "decoder.dec_layers.2.layernorm1.weight",
    "decoder.dec_layers.2.layernorm2.weight",
    "decoder.dec_layers.2.layernorm3.weight",
    "decoder.dec_layers.3.layernorm1.weight",
    "decoder.dec_layers.3.layernorm2.weight",
    "decoder.dec_layers.3.layernorm3.weight",
}

SKIP_SUBSTR = ("embedding", "layernorm", "pos_encoding", "softmax")

def should_skip_module_path(module_path: str) -> bool:
    low = module_path.lower()
    return any(s in low for s in SKIP_SUBSTR)

def should_skip_by_param_name(module_path: str, skip_param_names: set) -> bool:
    return (module_path + ".weight") in skip_param_names

print(f"Skip set loaded with {len(SKIP_PARAM_NAMES)} entries.")
# prune dead linear layers
import torch.nn as nn

def _get_parent_and_attr(root, fullname):
    parts = fullname.split('.')
    if len(parts) == 1:
        return root, parts[0]
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

class ZeroLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
    def forward(self, x):
        # preserve leading dims, change last dim -> out_features
        out_shape = list(x.shape[:-1]) + [self.out_features]
        return x.new_zeros(*out_shape)

def prune_dead_linears(model: nn.Module, threshold=1e-8):
    pruned = []
    for fullname, module in list(model.named_modules()):
        if fullname == "":  # skip root
            continue
        if isinstance(module, nn.Linear):
            # skip per rules
            if should_skip_module_path(fullname) or should_skip_by_param_name(fullname, SKIP_PARAM_NAMES):
                # print(f"SKIP prune: {fullname}")
                continue
            w = module.weight.detach()
            if w.numel() == 0:
                continue
            if float(w.abs().max()) < threshold:
                parent, attr = _get_parent_and_attr(model, fullname)
                setattr(parent, attr, ZeroLinear(module.in_features, module.out_features))
                pruned.append(fullname)
                print(f"Pruned dead Linear -> ZeroLinear: {fullname}")
    return pruned

print("Running pruning (skips respected)...")
pruned_list = prune_dead_linears(net_fp32, threshold=1e-8)
print("Pruned layers:", pruned_list)
import torch, torch.nn as nn
import torch.nn.functional as F

class MinMaxObserver:
    def __init__(self):
        self.min_val, self.max_val = None, None
    def observe(self, x: torch.Tensor):
        x = x.detach()
        if x.numel() == 0: return
        mn, mx = float(x.min()), float(x.max())
        self.min_val = mn if self.min_val is None else min(self.min_val, mn)
        self.max_val = mx if self.max_val is None else max(self.max_val, mx)
    def get_qparams(self, num_bits=8, symmetric=False, eps=1e-12):
        if self.min_val is None or self.max_val is None:
            return 1.0, 0
        qmin, qmax = (-(2**(num_bits-1)), 2**(num_bits-1)-1) if symmetric else (0, 2**num_bits-1)
        mn, mx = self.min_val, self.max_val
        if mx - mn < eps:
            return 1.0, int((qmin + qmax) // 2)
        scale = (mx - mn) / (qmax - qmin)
        scale = max(scale, 1e-6)
        if symmetric:
            zp = 0
        else:
            zp = int(round(qmin - mn / (scale + 1e-12)))
            zp = max(qmin, min(qmax, zp))
        return float(scale), int(zp)

class QuantLinear(nn.Module):
    def __init__(self, fp_linear: nn.Linear = None, name: str = ""):
        super().__init__()
        self.name = name
        # buffers
        self.register_buffer("int_weight", None)
        self.register_buffer("w_scale", torch.tensor(0.0))
        self.register_buffer("w_zp", torch.tensor(0))
        self.register_buffer("fp_bias", None)
        # float copy 
        self.fp_weight = None
        # observer
        self.act_obs = MinMaxObserver()
        self.act_scale = None
        self.act_zp = None
        self.calibrating = True
        if fp_linear is not None:
            self.in_features = fp_linear.in_features
            self.out_features = fp_linear.out_features
            self.pack_from_fp(fp_linear)

    @torch.no_grad()
    def pack_from_fp(self, fp_linear: nn.Linear, weight_bits: int = 8):
        w = fp_linear.weight.detach().float().clone()
        self.fp_weight = w.clone()
        qmax = 2**(weight_bits-1) - 1
        absmax = float(w.abs().max()) if w.numel() else 0.0
        absmax = max(absmax, 1e-6)
        w_scale = max(absmax / qmax, 1e-6)
        w_int = torch.clamp((w / w_scale).round(), -128, 127).to(torch.int8)
        self.int_weight = w_int
        self.w_scale.copy_(torch.tensor(float(w_scale)))
        self.w_zp.copy_(torch.tensor(0))
        self.fp_bias = None if fp_linear.bias is None else fp_linear.bias.detach().float().clone()

    def forward(self, x: torch.Tensor):
        if self.calibrating:
            self.act_obs.observe(x.detach())
        if self.int_weight is None:
            if self.fp_weight is not None:
                return F.linear(x, self.fp_weight, self.fp_bias)
            else:
                raise RuntimeError(f"QuantLinear {self.name} has no weights packed.")
        w = self.int_weight.float() * float(self.w_scale)
        out = F.linear(x, w, self.fp_bias)
        return out

    @torch.no_grad()
    def finalize(self, act_bits=8):
        scale, zp = self.act_obs.get_qparams(num_bits=act_bits, symmetric=False)
        scale = max(scale, 1e-6)
        self.act_scale = scale
        self.act_zp = zp
        self.calibrating = False

    def is_packed(self):
        return (self.int_weight is not None)
import torch.nn as nn

def replace_linear_with_quant(module: nn.Module, parent_path: str = ""):
    replaced = []
    for child_name, child in list(module.named_children()):
        child_path = f"{parent_path}.{child_name}" if parent_path else child_name
        if not isinstance(child, nn.Linear):
            replaced.extend(replace_linear_with_quant(child, child_path))
            continue
        # Skip rules
        if should_skip_module_path(child_path) or should_skip_by_param_name(child_path, SKIP_PARAM_NAMES):
            continue
        qlin = QuantLinear(child, name=child_path)
        setattr(module, child_name, qlin)
        replaced.append(child_path)
    return replaced

# Use pruned fp32 model 
net_q = net_fp32  
replaced = replace_linear_with_quant(net_q, "")
print(f"Replaced {len(replaced)} Linear layers with QuantLinear (skip respected).")
for i, n in enumerate(replaced[:30]): print(f"[{i:02d}] {n}")
# Calibration
from torch.utils.data import DataLoader
from dataset import EurDataset, collate_data

train_dataset = EurDataset(split="train", data_dir="./")
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_data)

def calibrate(model, loader, num_batches=200):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            src = batch.to(device); tgt = batch.to(device)
            _ = model(src, tgt)
            if i >= num_batches: break
    for m in model.modules():
        if isinstance(m, QuantLinear):
            m.finalize()

print("Starting calibration...")
calibrate(net_q, train_loader, num_batches=200)
print("Calibration done. Quant params frozen.")
# verify and saving model
import numpy as np
bad_layers = []
cnt = 0
print("=== Quantization summary ===")
for name, m in net_q.named_modules():
    if isinstance(m, QuantLinear):
        cnt += 1
        ws = float(m.w_scale) if m.w_scale is not None else None
        ac = m.act_scale if m.act_scale is not None else None
        print(f"{name:40s} | w_scale={ws:.6g} | act_scale={ac}")
        if ac is None:
            bad_layers.append(name)

print("\nDead/Uncalibrated QuantLinear layers (no act_scale):", bad_layers if bad_layers else "None")
print("Total QuantLinear layers found:", cnt)

# save state_dict
torch.save(net_q.state_dict(), "./deepsc_int8_state_pruned.pth")
print("Saved quantized+pruned state_dict")

# layer comparing
def get_module_by_name(root, fullname):
    mod = root
    if fullname == "": return mod
    for p in fullname.split('.'):
        mod = getattr(mod, p)
    return mod

def compare_weights(fp_model, q_model, layer_paths, sample_n=12):
    for path in layer_paths:
        try:
            m_fp = get_module_by_name(fp_model, path)
            m_q  = get_module_by_name(q_model, path)
        except Exception as e:
            print(f"Cannot find {path}: {e}")
            continue

        #  fp weights
        if hasattr(m_fp, "weight"):
            w_fp = m_fp.weight.detach().cpu().float().flatten()
        else:
            print(f"FP32 no weight at {path}")
            continue

        # q-dq weights
        if isinstance(m_q, QuantLinear):
            if m_q.int_weight is None:
                print(f"QuantLinear at {path} has no int_weight.")
                continue
            w_q = (m_q.int_weight.float() * float(m_q.w_scale)).detach().cpu().float().flatten()
        elif hasattr(m_q, "weight"):
            w_q = m_q.weight.detach().cpu().float().flatten()
        else:
            print(f"Quant module type unknown at {path}")
            continue

        n = min(sample_n, w_fp.numel())
        sample_fp = w_fp[:n].numpy()
        sample_q  = w_q[:n].numpy()
        diffs = np.abs(sample_fp - sample_q)
        print(f"\nComparing {path}: mean abs diff (sample) = {diffs.mean():.6g}, max abs fp = {w_fp.abs().max():.6g}")
        print("FP32 sample:", sample_fp)
        print("QUANT sample:", sample_q)

to_check = [
    "encoder.enc_layers.0.mha.wq",
    "encoder.enc_layers.1.mha.wq",
    "dense"
]
compare_weights(net_fp32, net_q, to_check)
