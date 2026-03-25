# Problem 2: Character-Level RNN Name Generation

**Name:** Dhruva Kumar Kaushal (B22AI017)

## Overview

This directory contains implementations of three character-level Recurrent Neural Network (RNN) architectures for generating Indian names. The project demonstrates sequence modeling, character-level text generation, and comprehensive model evaluation.

## Models Implemented

### 1. Vanilla RNN
- Standard multi-layer RNN with tanh activation
- 2 hidden layers with 256 units each
- Simple recurrent architecture for baseline comparison
- Implementation: `models/vanilla_rnn.py`

### 2. BLSTM (Bidirectional LSTM)
- Bidirectional LSTM with forward and backward passes
- 2 layers with 256 hidden units
- Captures both past and future context
- Implementation: `models/blstm.py`

### 3. RNN with Attention
- Standard RNN enhanced with attention mechanism
- Learns to focus on relevant parts of the sequence
- 2 layers with 256 units and 128 attention dimensions
- Implementation: `models/rnn_attention.py`

## Directory Structure

```
problem2_name_generation/
├── README.md                    # This file
├── config.yaml                  # Configuration parameters
├── main.py                      # Main execution script
├── models/
│   ├── base_model.py           # Abstract base class for RNN models
│   ├── vanilla_rnn.py          # Vanilla RNN implementation
│   ├── blstm.py                # Bidirectional LSTM implementation
│   └── rnn_attention.py        # RNN with Attention implementation
├── training/
│   ├── __init__.py
│   └── trainer.py              # Training loop and utilities
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              # Evaluation metrics
│   └── generate_samples.py    # Name generation utilities
├── preprocessing/
│   ├── character_vocab.py      # Character vocabulary management
│   └── generate_names.py      # Dataset generation script
├── data/
│   ├── raw/
│   │   ├── TrainingNames.txt   # Original names dataset
│   │   └── indian_names.json   # Processed names (5000 entries)
│   └── processed/
│       └── char_vocab.json     # Character vocabulary
└── outputs/
    ├── vanilla_rnn/            # Vanilla RNN results
    ├── blstm/                  # BLSTM results
    ├── rnn_attention/          # RNN with Attention results
    └── model_comparison.json   # Comparison across models
```

## Installation

Install required dependencies:
```bash
cd /Users/sanjay/Documents/8thsem/NLU
pip install -r requirements.txt
```

## Usage

### Complete Pipeline
Run all tasks for all models:
```bash
python main.py --task all
```

This executes:
1. Training all three models
2. Evaluating all models with comprehensive metrics
3. Generating sample names from each model

### Individual Tasks

**Training:**
```bash
# Train all models
python main.py --task train

# Train specific model
python main.py --task train --model vanilla_rnn
python main.py --task train --model blstm
python main.py --task train --model rnn_attention
```

**Evaluation:**
```bash
# Evaluate all models
python main.py --task evaluate

# Evaluate specific model
python main.py --task evaluate --model vanilla_rnn
```

**Generation:**
```bash
# Generate names from all models
python main.py --task generate

# Generate from specific model
python main.py --task generate --model rnn_attention
```

## Configuration

All hyperparameters are configurable in `config.yaml`:

### Model Architecture
- **embedding_dim**: 128 (Character embedding dimension)
- **hidden_dim**: 256 (RNN hidden state dimension)
- **num_layers**: 2 (Number of RNN layers)
- **dropout**: 0.2 (Dropout rate)
- **attention_dim**: 128 (Attention mechanism dimension)

### Training Parameters
- **num_epochs**: 30 (Training epochs)
- **batch_size**: 64 (Mini-batch size)
- **learning_rate**: 0.001 (Adam optimizer learning rate)
- **early_stopping_patience**: 7 (Epochs before early stopping)
- **train_split**: 0.7 (Training set ratio)
- **val_split**: 0.15 (Validation set ratio)
- **test_split**: 0.15 (Test set ratio)

### Generation Parameters
- **max_length**: 15 (Maximum name length)
- **temperature**: 0.8 (Sampling temperature)
- **top_k**: 0 (Top-k sampling, 0=disabled)
- **top_p**: 0.9 (Nucleus sampling threshold)
- **n_samples**: 100 (Number of names to generate)

## Dataset

### Source
- 5000 Indian names in `data/raw/indian_names.json`
- Includes diverse Indian names from various regions and cultures
- Character-level tokenization with special tokens

### Character Vocabulary
- Total vocabulary size: 28 characters
- Includes: a-z, special tokens ([SOS], [EOS], [PAD], [UNK])
- Stored in: `data/processed/char_vocab.json`

### Data Split
- Training: 3500 names (70%)
- Validation: 750 names (15%)
- Test: 750 names (15%)

## Evaluation Metrics

### 1. Perplexity
- Measures how well the model predicts the test set
- Lower is better
- Formula: exp(cross_entropy_loss)

### 2. Novelty
- Percentage of generated names not in training set
- Higher indicates less memorization
- Range: 0-100%

### 3. Diversity
- Percentage of unique names among generated samples
- Higher indicates more variety
- Range: 0-100%

### 4. Character Distribution
- Analyzes character frequency in generated names
- Compares with training data distribution

### 5. Length Distribution
- Mean, std, min, max, median length
- Compares with training data lengths

## Results Summary

### Vanilla RNN
**Performance:**
- Perplexity: 2.27
- Novelty: 0.00% (all names from training set)
- Diversity: 63% (63 unique out of 100)
- Mean Length: 6.68 characters

**Characteristics:**
- Generates most realistic Indian names
- High quality but tends to memorize training data
- Examples: "bharath", "priya", "krishnan", "aditi", "subhadra"

**Issue:** Overfitting - generates only training examples

### BLSTM (Bidirectional LSTM)
**Performance:**
- Perplexity: 1.00 (excellent but misleading)
- Novelty: 100% (all novel names)
- Diversity: 50% (26 unique out of 52)
- Mean Length: ~15 characters

**Characteristics:**
- Mode collapse problem
- Generates only repetitive patterns
- Examples: "kekekekekekekek", "unununununununu", "sisisisisisisis"

**Issue:** Severe mode collapse - unusable output

### RNN with Attention
**Performance:**
- Perplexity: 1.00
- Novelty: 99% (almost all novel)
- Diversity: 92% (92 unique out of 100)
- Mean Length: Variable

**Characteristics:**
- Highest diversity among all models
- Mix of realistic and nonsensical names
- Examples: "zoya" (good), "choc", "zabam", "bhacbububbbbbbb" (poor)

**Issue:** High diversity but inconsistent quality

## Output Files

### Per-Model Outputs
For each model (vanilla_rnn, blstm, rnn_attention):
- `outputs/<model>/best_model.pt`: Best model checkpoint
- `outputs/<model>/training_history.json`: Training metrics
- `outputs/<model>/evaluation_results.json`: Comprehensive evaluation
- `outputs/<model>/generated_names.json`: Generated names with metadata
- `outputs/<model>/generated_names.txt`: Plain text generated names

### Comparison
- `outputs/model_comparison.json`: Side-by-side comparison of all models

## Model Architectures

### Vanilla RNN
```
Input: Character sequence [c1, c2, ..., cn]
      ↓
Embedding Layer (vocab_size × embedding_dim)
      ↓
RNN Layers × 2 (hidden_dim=256)
      ↓
Linear Layer (hidden_dim × vocab_size)
      ↓
Output: Next character prediction
```

### BLSTM
```
Input: Character sequence [c1, c2, ..., cn]
      ↓
Embedding Layer (vocab_size × embedding_dim)
      ↓
Bidirectional LSTM Layers × 2
      ↓
Linear Layer (hidden_dim × vocab_size)
      ↓
Output: Next character prediction
```

### RNN with Attention
```
Input: Character sequence [c1, c2, ..., cn]
      ↓
Embedding Layer (vocab_size × embedding_dim)
      ↓
RNN Encoder (generates sequence representations)
      ↓
Attention Mechanism (computes context vector)
      ↓
RNN Decoder with attended context
      ↓
Linear Layer (hidden_dim × vocab_size)
      ↓
Output: Next character prediction
```

## Training Details

### Optimization
- Optimizer: Adam
- Loss: Cross-entropy
- Gradient clipping: 5.0
- Learning rate scheduling: ReduceLROnPlateau

### Early Stopping
- Monitors validation loss
- Patience: 7 epochs
- Saves best model automatically

### Logging
- Epoch-wise training/validation loss
- Best model selection
- Progress bars with TQDM

## Sample Generated Names

### Vanilla RNN (Best Quality)
```
ananth, priya, sarabjit, anita, subhadra, gargi, tanushree,
prasad, nanditha, gopalan, palak, hrusikesh, bhavana, vasundha,
krishnan, bharath, vimala, anika, karthik, aditi
```

### RNN with Attention (High Diversity)
```
zoya, zabam, gowan, fuban, cubap, fubab, zoday, zopan,
choc, chuc, chag, chach, gow
```

## Known Issues

### 1. Vanilla RNN Overfitting
- Memorizes training data (0% novelty)
- Needs stronger regularization
- Suggested fixes: Higher dropout, weight decay, more data

### 2. BLSTM Mode Collapse
- Generates only repetitive patterns
- Critical issue requiring investigation
- Suggested fixes: Different initialization, learning rate adjustment, layer normalization

### 3. RNN Attention Quality
- High diversity but many nonsensical names
- Needs vocabulary constraints
- Suggested fixes: Beam search, n-gram penalties, better sampling

## Recommendations

### For Best Name Quality
Use **Vanilla RNN** with additional regularization:
- Increase dropout to 0.3-0.4
- Add weight decay
- Use temperature > 1.0 during generation

### For Production Use
- Fix BLSTM mode collapse issue first
- Implement beam search for all models
- Add vocabulary constraints
- Increase dataset size to 10k+ names
- Use ensemble of multiple models

## Troubleshooting

### Common Issues

1. **Mode collapse in BLSTM**
   - Try different random seeds
   - Adjust learning rate
   - Check initialization

2. **Low diversity in generation**
   - Increase temperature parameter
   - Enable top-k or nucleus sampling
   - Use different random seeds

3. **Memory errors**
   - Reduce batch_size in config.yaml
   - Reduce hidden_dim or num_layers

4. **Poor name quality**
   - Train for more epochs
   - Increase model capacity
   - Expand dataset

## References

1. Graves, A. (2013). Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850.

2. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.

3. Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.

4. Graves, A., Mohamed, A. R., & Hinton, G. (2013). Speech recognition with deep recurrent neural networks. In ICASSP (pp. 6645-6649).

## Future Enhancements

1. Fix BLSTM mode collapse issue
2. Implement beam search decoding
3. Add GRU architecture
4. Implement Transformer-based model
5. Add conditioning (e.g., generate names by gender/region)
6. Expand dataset with more diverse names
7. Add phonetic constraints
8. Implement name style transfer

---

For more details, see the main repository README and final report (FINAL_REPORT.pdf).
