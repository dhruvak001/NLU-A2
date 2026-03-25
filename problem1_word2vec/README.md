# Problem 1: Word2Vec Implementation from Scratch

**Name:** Dhruva Kumar Kaushal (B22AI017)

## Overview

This directory contains a complete implementation of Word2Vec models (CBOW and Skip-gram) trained from scratch using PyTorch. The implementation includes data collection, preprocessing, training, and comprehensive evaluation on a corpus scraped from IIT Jodhpur's website.

## Models Implemented

### 1. CBOW (Continuous Bag of Words)
- Predicts target word from surrounding context words
- Uses averaging of context word embeddings
- Efficient for frequent words
- Implementation: `word2vec/cbow.py`

### 2. Skip-gram
- Predicts context words from target word
- Better for rare words and smaller datasets
- Implementation: `word2vec/skipgram.py`

### Key Features
- Negative sampling for computational efficiency
- Dynamic context windows
- Subsampling of frequent words
- Configurable hyperparameters
- Pure PyTorch implementation

## Directory Structure

```
problem1_word2vec/
├── README.md                    # This file
├── config.yaml                  # Configuration parameters
├── main.py                      # Main execution script
├── word2vec/
│   ├── cbow.py                 # CBOW model implementation
│   ├── skipgram.py             # Skip-gram model implementation
│   ├── vocabulary.py           # Vocabulary management
│   └── trainer.py              # Training infrastructure
├── analysis/
│   ├── similarity.py           # Word similarity analysis
│   ├── analogy.py              # Word analogy testing
│   └── visualization.py        # PCA and t-SNE visualizations
├── scripts/
│   ├── preprocessing/
│   │   └── preprocess.py       # Text preprocessing
│   ├── statistics.py           # Corpus statistics
│   └── comparison.py           # Gensim comparison
├── data/
│   ├── raw/                    # Raw scraped data (organized by category)
│   └── processed/              # Preprocessed text corpus
└── outputs/
    ├── models/                 # Saved model checkpoints
    ├── embeddings/             # Word embeddings (.npy files)
    ├── results/                # Analysis results (JSON)
    └── visualizations/         # Plots and figures (PNG)
```

## Installation

Install required dependencies:
```bash
cd /Users/sanjay/Documents/8thsem/NLU
pip install -r requirements.txt
```

## Usage

### Complete Pipeline
Run all tasks sequentially:
```bash
python main.py --task all
```

This executes:
1. Data preprocessing
2. Model training (CBOW and Skip-gram)
3. Similarity analysis
4. Analogy testing
5. Visualization generation
6. Model comparison

### Individual Tasks

**Training Models:**
```bash
# Train both models
python main.py --task train

# Train specific model
python main.py --task train --model cbow
python main.py --task train --model skipgram
```

**Evaluation:**
```bash
# Run all evaluations
python main.py --task evaluate

# Similarity analysis only
python main.py --task similarity

# Analogy testing only
python main.py --task analogy

# Visualization only
python main.py --task visualize
```

**Model Comparison:**
```bash
# Compare CBOW vs Skip-gram vs Gensim
python main.py --task compare
```

## Configuration

All hyperparameters are configurable in `config.yaml`:

### Training Parameters
- **embedding_dim**: 100 (Word vector dimension)
- **window_size**: 5 (Context window radius)
- **min_count**: 3 (Minimum word frequency)
- **num_epochs**: 50 (Training epochs)
- **batch_size**: 512 (Mini-batch size)
- **learning_rate**: 0.025 (Initial learning rate)
- **negative_samples**: 10 (Number of negative samples)

### Model Variants
The configuration includes settings for experimenting with:
- Embedding dimensions: 50, 100, 200
- Window sizes: 2, 5, 10
- Negative samples: 5, 10, 20

## Experiments Conducted

### 1. Baseline Models
- **CBOW baseline**: dim=100, window=5, neg_samples=10
- **Skip-gram baseline**: dim=100, window=5, neg_samples=10

### 2. Embedding Dimension Experiments
- CBOW with dim=50, dim=100, dim=200
- Analysis of dimensionality impact on embedding quality

### 3. Window Size Experiments
- CBOW with window=2, window=5, window=10
- Analysis of context size impact

### 4. Negative Sampling Experiments
- Skip-gram with neg_samples=5, 10, 20
- Analysis of negative sampling impact

## Evaluation Metrics

### 1. Word Similarity
Tests cosine similarity for semantically related words:
- Example queries: "student", "research", "faculty", "department"
- Metrics: Cosine similarity scores
- Output: `outputs/results/similarity_results.json`

### 2. Word Analogies
Tests vector arithmetic for analogical relationships:
- Format: "a is to b as c is to ?"
- Examples: educational and academic relationships
- Output: `outputs/results/analogy_results.json`

### 3. Visualization
- **PCA**: 2D projection of word embeddings
- **t-SNE**: Non-linear dimensionality reduction
- Output: `outputs/visualizations/*.png`

### 4. Comparison
- Compare against Gensim's pre-trained Word2Vec
- Metrics: similarity scores, analogy accuracy
- Output: `outputs/results/comparison.json`

## Results

### Dataset Statistics
- **Total tokens**: ~50,000-100,000 (varies based on scraping)
- **Vocabulary size**: ~5,000-8,000 unique words
- **Categories**: About, Departments, Research, Academics, Courses

### Model Performance
Results vary based on configuration, but typical performance:
- **Training time**: 5-10 minutes per model
- **Embedding quality**: Good for domain-specific terms
- **Similarity accuracy**: ~60-70% for common words
- **Analogy accuracy**: ~30-40% (limited by small corpus)

### Best Configurations
Based on experiments:
- **CBOW**: dim=100, window=5, neg_samples=10 (baseline)
- **Skip-gram**: dim=100, window=5, neg_samples=10 (baseline)

## Output Files

### Models and Embeddings
- `outputs/models/*_final.pt`: Trained model checkpoints
- `outputs/embeddings/*_embeddings.npy`: Word embeddings as NumPy arrays

### Results
- `outputs/results/similarity_results.json`: Similarity analysis
- `outputs/results/analogy_results.json`: Analogy test results
- `outputs/results/comparison.json`: Model comparison

### Visualizations
- `outputs/visualizations/pca_cbow.png`: PCA visualization (CBOW)
- `outputs/visualizations/pca_skipgram.png`: PCA visualization (Skip-gram)
- `outputs/visualizations/tsne_cbow.png`: t-SNE visualization (CBOW)
- `outputs/visualizations/tsne_skipgram.png`: t-SNE visualization (Skip-gram)

## Implementation Details

### CBOW Architecture
```
Input: Context word indices [w_{t-c}, ..., w_{t-1}, w_{t+1}, ..., w_{t+c}]
      ↓
Embedding Layer (vocab_size × embedding_dim)
      ↓
Average context embeddings
      ↓
Linear projection (embedding_dim × vocab_size)
      ↓
Output: Target word prediction w_t
```

### Skip-gram Architecture
```
Input: Target word index w_t
      ↓
Embedding Layer (vocab_size × embedding_dim)
      ↓
Linear projection (embedding_dim × vocab_size)
      ↓
Output: Context word predictions [w_{t-c}, ..., w_{t+c}]
```

### Negative Sampling
- For each positive (target, context) pair, sample K negative words
- Loss: Binary cross-entropy on positive and negative samples
- Sampling distribution: Unigram^(3/4) (smoothed frequency)

## Corpus Details

### Data Sources
Scraped from IIT Jodhpur website:
- About pages
- Department information
- Research descriptions
- Academic regulations
- Course descriptions

### Preprocessing Steps
1. HTML tag removal
2. Lowercase conversion
3. Punctuation removal
4. Tokenization
5. Stopword filtering (optional)
6. Rare word filtering (min_count=3)

## Troubleshooting

### Common Issues

1. **Small vocabulary warnings**
   - Corpus may be small; consider increasing web scraping depth
   - Lower min_count parameter in config.yaml

2. **Memory errors during training**
   - Reduce batch_size in config.yaml
   - Use smaller embedding_dim

3. **Poor analogy results**
   - Expected for small corpus (limited vocabulary)
   - Word2Vec requires large corpus for good analogies

4. **Visualization empty/cluttered**
   - Adjust max_words parameter in visualization config
   - Try different random seeds for t-SNE

## References

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.

2. Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems (pp. 3111-3119).

3. Goldberg, Y., & Levy, O. (2014). word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method. arXiv preprint arXiv:1402.3722.

## Future Enhancements

1. Implement hierarchical softmax as alternative to negative sampling
2. Add subword information (FastText-style)
3. Expand corpus with more diverse text sources
4. Implement phrase detection and phrasal embeddings
5. Add downstream task evaluation (sentiment analysis, classification)

---

For more details, see the main repository README and final report (FINAL_REPORT.pdf).
