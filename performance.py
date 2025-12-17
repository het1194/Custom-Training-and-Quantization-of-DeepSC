import os
import json
import torch
import argparse
import numpy as np
from dataset import EurDataset, collate_data
from transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm
from sklearn.preprocessing import normalize
from w3lib.html import remove_tags

parser = argparse.ArgumentParser()
# <--- FIX 1: The data-dir is the folder '.', not a file
parser.add_argument('--data-dir', default='.', type=str) 
parser.add_argument('--vocab-file', default='snli_vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='./checkpoint_237.pth', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type = int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=1, type = int) # Changed to 1, no need to run eval twice

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def performance(args, SNR, net):
    bleu_score_1gram = BleuScore(1, 0, 0, 0)

    # <--- FIX 2: Tell EurDataset where to find the test.pkl file
    print(f"Loading test data from {args.data_dir}/test.pkl")
    test_eur = EurDataset('test', data_dir=args.data_dir) 
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)

    StoT = SeqtoText(token_to_idx, end_idx)
    score = []
    score2 = []
    net.eval()
    print("Model in eval mode. Starting inference...")
    with torch.no_grad():
        for epoch in range(args.epochs):
            print(f"Running evaluation pass {epoch+1}/{args.epochs}")
            Tx_word = []
            Rx_word = []

            for snr in tqdm(SNR, desc=f"SNR Loop"):
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

            print("Calculating BLEU score...")
            bleu_score = []
            sim_score = []
            for sent1, sent2 in zip(Tx_word, Rx_word):
                # 1-gram
                bleu_score.append(bleu_score_1gram.compute_blue_score(sent1, sent2)) # 7*num_sent
    
            bleu_score = np.array(bleu_score)
            bleu_score = np.mean(bleu_score, axis=1)
            score.append(bleu_score)

    score1 = np.mean(np.array(score), axis=0)

    return score1

if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [0,3,6,9,12,15,18]

    # Use os.path.join for safety
    vocab_path = os.path.join(args.data_dir, args.vocab_file)
    print(f"Loading vocab from: {vocab_path}")
    vocab = json.load(open(vocab_path, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    print("Initializing FP32 DeepSC model...")
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                        num_vocab, num_vocab, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)

    # <--- FIX 3: Remove the directory scanning loop and load the file directly
    model_path = args.checkpoint_path
    print(f"Loading FP32 checkpoint from: {model_path}")
    
    # Load checkpoint, making sure to map it to the correct device
    checkpoint = torch.load(model_path, map_location=device) 
    
    # Check if the checkpoint is a state_dict or a full model
    # Your quant.py script suggests it's a state_dict
    if isinstance(checkpoint, dict) and not 'model_state_dict' in checkpoint:
        deepsc.load_state_dict(checkpoint)
    elif 'model_state_dict' in checkpoint:
         deepsc.load_state_dict(checkpoint['model_state_dict'])
    else:
        # This case is unlikely but safe to have
        deepsc = checkpoint
        
    print('FP32 model loaded successfully!')
    # --->

    bleu_score = performance(args, SNR, deepsc)
    
    print("\n======================================")
    print(f"   Final FP32 BLEU Score   ")
    print("======================================")
    print(f"SNR Values:  {SNR}")
    print(f"BLEU Scores: {bleu_score}")