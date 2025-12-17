# Custom Training and Quantization of DeepSC
This project implements custom dataset training and quantization of DeepSC, a Semantic Communication JSCC model originally created by Huiqiang Xie et al. 

## Project Overview

The codebase supports:
- Custom FP32 training of DeepSC using the SNLI dataset
- Hybrid structural pruning and weight-only INT8 post-training quantization
- Evaluation of both FP32 and quantized models using BLEU-1

## Dataset Preparation

The code expects preprocessed dataset files:
- `train.pkl`
- `test.pkl`
- `snli_vocab.json`

These were generated from the SNLI corpus. Available at: https://nlp.stanford.edu/projects/snli/
The original DeepSC preprocessing logic is retained, but incase you wish to use the original model, it can be found at this repo: https://github.com/13274086/DeepSC

## Execution Pipeline

### 1. Train FP32 DeepSC Model
```bash
python main.py
```
### 2. Evaluate FP32 Model
```bash
python performance.py
```
### 3. Quantize the Trained Model
```bash
python quant.py
```
### 4. Evaluate Quantized Model
```bash
python testquant.py
```
