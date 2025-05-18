# Arithmetic Reasoning with Transformers

A PyTorch implementation of a Transformer-based model to solve arithmetic expressions involving multi-digit operands and various edge cases.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Ablation Studies](#ablation-studies)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [License](#license)

## Overview
This project implements a sequence-to-sequence Transformer model to perform exact arithmetic computations on expressions with up to 7-digit operands. It handles:
- Negative numbers
- Zero operands
- Carry operations
- Leading zeros
- Multi-operand expressions

## Features
- **Dataset Pipeline**: Automatically generates training (30,000), validation (1,763), and test (6,000) samples across targeted edge-case scenarios.
- **Transformer Model**: 4-layer encoder and decoder, 8 attention heads, model dimension = 256, feed-forward dimension = 1024, sinusoidal positional encoding.
- **Training Scripts**: PyTorch training loop with support for checkpointing and configurable hyperparameters.
- **Evaluation**: Calculates exact-match (EM) accuracy and generalization on unseen long operands.
- **Ablation Tools**: Evaluate effects of positional encoding and varying number of attention heads.

## Dataset
The dataset covers four categories:
1. **Standard Arithmetic**: Random pairs of integers 1–7 digits.
2. **Edge Cases**: Expressions with negative results, zeros, carries, and leading zeros.
3. **Multi-Operand**: Chains of 3–5 operands.
4. **Generalization Test**: 8-digit operands not seen during training.

All samples are formatted as `"<operand1>+<operand2>" -> "<result>"` sequences.

## Model Architecture
- **Encoder & Decoder Layers**: 4 each
- **Attention Heads**: 8
- **Model Dimension (`d_model`)**: 256
- **Feed-Forward Dimension (`d_ff`)**: 1024
- **Positional Encoding**: Sinusoidal
- **Optimizer**: Adam
- **Learning Rate Scheduler**: Inverse square root decay

## Installation
```bash
# Clone the repository
git clone https://github.com/Sayam241020/inlp-arithmetic-transformer.git
cd inlp-arithmetic-transformer

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
``` 

## Usage
### Data Generation
```bash
python data/generate_dataset.py --train 30000 --val 1763 --test 6000
```

### Training
```bash
python train.py \
  --dataset data/ \
  --epochs 50 \
  --batch_size 64 \
  --d_model 256 \
  --n_heads 8 \
  --d_ff 1024 \
  --num_layers 4 \
  --save_dir checkpoints/
```

### Evaluation
```bash
python evaluate.py \
  --model_path checkpoints/model_final.pt \
  --test_data data/test.json
```

### Ablation Studies
- **Positional Encoding**: `python train.py --no_positional_encoding`
- **Attention Heads**: `python train.py --n_heads 4` or `--n_heads 16`

## Evaluation Metrics
- **Validation EM Accuracy**: 49.91%
- **Test EM on Unseen 8-Digit Operands**: 9.50%

## Results
| Experiment | EM Accuracy |
|------------|-------------|
| Baseline (7-digit) | 49.91% |
| No Positional Encoding | 12.34% |
| 4 Heads | 45.67% |
| 16 Heads | 50.12% |

## Repository Structure
```
├── data/                   # Dataset generation and samples
├── models/                 # Saved checkpoints
├── src/                    # Source code
│   ├── model.py            # Transformer implementation
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── data_utils.py       # Data preprocessing and tokenization
├── requirements.txt        # Python dependencies
├── README.md               # Project overview
└── LICENSE
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
