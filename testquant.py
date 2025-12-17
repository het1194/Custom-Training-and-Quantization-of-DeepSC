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
from quant_layers import ZeroLinear, QuantLinear, replace_linear_with_quant, SKIP_PARAM_NAMES, SKIP_SUBSTR, _get_parent_and_attr

print("--- Imports successful ---")

def performance(args, SNR, net):
    print("\n--- Starting Performance Evaluation ---")
    bleu_score_1gram = BleuScore(1, 0, 0, 0)

    print(f"Loading test dataset from: {args.data_dir}/test.pkl")
    test_eur = EurDataset('test', data_dir=args.data_dir)
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)
    print(f"Test dataset loaded: {len(test_eur)} samples.")

    StoT = SeqtoText(token_to_idx, end_idx)
    score = []
    net.eval()
    print("Model is in eval() mode.")
    
    with torch.no_grad():
        print("torch.no_grad() context enabled.")
        for epoch in range(args.epochs):
            print(f"\nRunning evaluation epoch {epoch+1}/{args.epochs}...")
            Tx_word = []
            Rx_word = []

            for snr in tqdm(SNR, desc=f"SNR Loop (Epoch {epoch+1})"):
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)

                for sents in test_iterator:
                    sents = sents.to(device)
                    target = sents
                    out = greedy_decode(net, sents, noise_std, args.MAX_LENGTH, pad_idx,
                                        start_idx, args.channel)

                    sentences = out.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, sentences))
                    word = word + result_string

                    target_sent = target.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + result_string

                Tx_word.append(word)
                Rx_word.append(target_word)

            print("Calculating BLEU score for this epoch...")
            bleu_score = []
            for sent1, sent2 in zip(Tx_word, Rx_word):
                bleu_score.append(bleu_score_1gram.compute_blue_score(sent1, sent2))
    
            bleu_score = np.array(bleu_score)
            bleu_score = np.mean(bleu_score, axis=1)
            score.append(bleu_score)
            print(f"Epoch {epoch+1} BLEU scores (per SNR): {bleu_score}")

    score1 = np.mean(np.array(score), axis=0)
    print("--- Performance Evaluation Finished ---")
    return score1
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='.', type=str, help="Directory containing test.pkl")
    parser.add_argument('--vocab-file', default='snli_vocab.json', type=str)
    parser.add_argument('--checkpoint-path', default='deepsc_int8_state_pruned.pth', type=str)
    parser.add_argument('--channel', default='Rayleigh', type=str)
    parser.add_argument('--MAX-LENGTH', default=30, type=int)
    parser.add_argument('--MIN-LENGTH', default=4, type=int)
    parser.add_argument('--d-model', default=128, type=int, help="[Ignored] Hardcoded to 128")
    parser.add_argument('--dff', default=512, type=int, help="[Ignored] Hardcoded to 512")
    parser.add_argument('--num-layers', default=4, type=int, help="[Ignored] Hardcoded to 4")
    parser.add_argument('--num-heads', default=8, type=int, help="[Ignored] Hardcoded to 8")
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=1, type=int, help="Num eval loops (original was 2)")

    args = parser.parse_args()
    print("\n--- Parsed Arguments ---")
    print(json.dumps(vars(args), indent=2))
    print("------------------------\n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    SNR = [0, 3, 6, 9, 12, 15, 18]
    print(f"Loading vocab from: {args.vocab_file}")
    vocab_path = os.path.join(args.data_dir, args.vocab_file)
    vocab = json.load(open(vocab_path, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]
    print(f"Vocab loaded. Size: {num_vocab}")
    
    MODEL_num_layers = 4
    MODEL_num_heads = 8
    MODEL_dff = 512
    MODEL_dropout = 0.1
    MODEL_d_model = 128
    MODEL_max_len = 5626 
    MODEL_vocab_size = num_vocab 
    deepsc_quant = DeepSC(
        MODEL_num_layers, MODEL_vocab_size, MODEL_vocab_size,
        MODEL_max_len, MODEL_max_len,
        MODEL_d_model, MODEL_num_heads, MODEL_dff, MODEL_dropout
    ).to(device)
    print("Base DeepSC model instantiated.")
    PRUNED_LAYERS = [
        'encoder.enc_layers.1.mha.wq',
        'encoder.enc_layers.1.mha.wk',
        'encoder.enc_layers.2.mha.wq',
        'encoder.enc_layers.2.mha.wk',
        'decoder.dec_layers.3.self_mha.wq',
        'decoder.dec_layers.3.self_mha.wk'
    ]
    for layer_name in PRUNED_LAYERS:
        try:
            parent, attr = _get_parent_and_attr(deepsc_quant, layer_name)
            old_module = getattr(parent, attr)
            
            if isinstance(old_module, nn.Linear):
                new_module = ZeroLinear(old_module.in_features, old_module.out_features)
                setattr(parent, attr, new_module)
                print(f"  [Pruned] Replaced '{layer_name}' with ZeroLinear")
            else:
                print(f"  [Prune Warning] Module '{layer_name}' is not nn.Linear, was {type(old_module)}")
        except Exception as e:
            print(f"  [Prune ERROR] Could not prune '{layer_name}': {e}")

    replaced = replace_linear_with_quant(deepsc_quant, "")
    model_path = os.path.join(args.data_dir, args.checkpoint_path)
    print(f"Loading weights from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"FATAL ERROR: Checkpoint file not found at {model_path}")
    else:
        ckpt = torch.load(model_path, map_location=device)
        missing, unexpected = deepsc_quant.load_state_dict(ckpt, strict=False)
        
        print(f"  Missing Keys ({len(missing)}):")
        if missing:
            for k in missing: print(f"    {k}")
        else:
            print("    None")
            
        print(f"\n  Unexpected Keys ({len(unexpected)}):")
        if unexpected:
            for k in unexpected: print(f"    {k}")
        else:
            print("    None")
        
        print("Quantized model weights loaded.")

    # Model Verification 
    deepsc_quant.eval()
    print("Model set to eval()")
    
    quant_layers_found = 0
    zero_layers_found = 0
    total_params = 0
    
    for name, m in deepsc_quant.named_modules():
        if isinstance(m, QuantLinear):
            quant_layers_found += 1
            if not m.is_packed():
                print(f"  [NOTE] QuantLinear layer '{name}' is packed: {m.is_packed()}")
        if isinstance(m, ZeroLinear):
            zero_layers_found += 1
        
        if hasattr(m, 'weight') and m.weight is not None:
            total_params += m.weight.numel()
        if hasattr(m, 'int_weight') and m.int_weight is not None:
            total_params += m.int_weight.numel()

    print(f"\nVerification Summary:")
    print(f"  Found {quant_layers_found} QuantLinear modules.")
    print(f"  Found {zero_layers_found} ZeroLinear modules (Pruned).")
    print(f"  Total model parameters (approx): {total_params}")
    
    bleu_score = performance(args, SNR, deepsc_quant)
    
    print("\n======================================")
    print(f"   Final Quantized BLEU Score   ")
    print(f"SNR Values: {SNR}")
    print(f"BLEU Scores: {bleu_score}")
    print("\nEvaluation complete.")