"""
Main Orchestration Script for Problem 1: Word Embeddings
========================================================

This script orchestrates the complete Word2Vec pipeline:
1. Data collection
2. Preprocessing
3. Model training (CBOW and Skip-gram)
4. Semantic analysis
5. Visualization
6. Comparison with Gensim

Author: Dhruva Kumar Kaushal (B22AI017)
Date: March 2026

Usage:
    python main.py --task all                # Run complete pipeline
    python main.py --task collect_data       # Only collect data
    python main.py --task preprocess         # Only preprocess
    python main.py --task train_cbow         # Train CBOW
    python main.py --task train_skipgram     # Train Skip-gram
    python main.py --task analyze            # Run analysis
    python main.py --task visualize          # Create visualizations
"""

import argparse
import yaml
import os
import sys
import torch
from torch.utils.data import DataLoader
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config.yaml."""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config


def task_collect_data(config):
    """Task 1: Collect data from IIT Jodhpur sources."""
    logger.info("="*60)
    logger.info("TASK: DATA COLLECTION")
    logger.info("="*60)
    
    from scripts.data_collection.scraper import IITJodhpurScraper
    
    scraper = IITJodhpurScraper(
        base_url=config['data_collection']['base_url'],
        rate_limit=config['data_collection']['rate_limit_seconds'],
        max_retries=config['data_collection']['max_retries']
    )
    
    output_dir = 'data/raw'
    all_metadata = []
    
    # Scrape each source
    for source in config['data_collection']['sources']:
        logger.info(f"Scraping source: {source['name']}")
        metadata = scraper.scrape_source(source, output_dir)
        all_metadata.extend(metadata)
    
    # Save combined metadata
    import json
    with open(os.path.join(output_dir, 'all_metadata.json'), 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    logger.info(f"Data collection complete. Total documents: {len(all_metadata)}")


def task_preprocess(config):
    """Task 2: Preprocess collected data."""
    logger.info("="*60)
    logger.info("TASK: PREPROCESSING")
    logger.info("="*60)
    
    from scripts.preprocessing.preprocess import TextPreprocessor
    
    preprocessor = TextPreprocessor(config['preprocessing'])
    
    # Process corpus
    input_dir = 'data/raw'
    sentences = preprocessor.process_corpus(input_dir)
    
    # Save processed corpus
    output_file = 'data/processed/corpus.txt'
    preprocessor.save_corpus(sentences, output_file)
    
    # Compute and save statistics
    stats = preprocessor.compute_statistics(sentences)
    preprocessor.save_statistics(stats, 'data/statistics/corpus_stats.json')
    
    logger.info(f"Preprocessing complete. Total sentences: {len(sentences)}")
    
    # Generate word cloud
    generate_word_cloud(sentences)


def generate_word_cloud(sentences):
    """Generate word cloud from processed corpus."""
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        from collections import Counter
        
        logger.info("Generating word cloud...")
        
        # Count all words
        all_words = [word for sentence in sentences for word in sentence]
        word_freq = Counter(all_words)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=1920,
            height=1080,
            background_color='white',
            colormap='viridis',
            max_words=200,
            relative_scaling=0.5
        ).generate_from_frequencies(word_freq)
        
        # Plot
        plt.figure(figsize=(16, 9), dpi=300)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - IIT Jodhpur Corpus', fontsize=20, fontweight='bold')
        
        # Save
        output_path = 'outputs/visualizations/wordcloud.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Word cloud saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to generate word cloud: {e}")


def task_train_models(config):
    """Task 3: Train all Word2Vec models."""
    logger.info("="*60)
    logger.info("TASK: MODEL TRAINING")
    logger.info("="*60)
    
    # Load vocabulary and corpus
    from word2vec.vocabulary import Vocabulary, load_corpus
    
    corpus_file = 'data/processed/corpus.txt'
    sentences = load_corpus(corpus_file)
    
    # Build or load vocabulary
    vocab_file = 'data/processed/vocabulary.pkl'
    if os.path.exists(vocab_file):
        logger.info("Loading existing vocabulary...")
        vocab = Vocabulary.load(vocab_file)
    else:
        logger.info("Building vocabulary...")
        vocab = Vocabulary(min_count=config['preprocessing']['min_word_count'])
        vocab.build_vocabulary(sentences)
        vocab.save(vocab_file)
    
    # Encode sentences
    encoded_sentences = [vocab.encode_sentence(sent) for sent in sentences]
    
    # Train models for each experiment
    for exp_config in config['word2vec']['experiments']:
        train_single_model(exp_config, encoded_sentences, vocab)


def train_single_model(exp_config, encoded_sentences, vocab):
    """Train a single Word2Vec model."""
    from word2vec.cbow import CBOW, create_training_pairs
    from word2vec.skipgram import SkipGram, create_skipgram_pairs, SkipGramDataset
    from word2vec.trainer import Word2VecTrainer, CBOWDataset
    
    model_name = exp_config['name']
    model_type = exp_config['model_type']
    
    logger.info(f"\nTraining model: {model_name}")
    logger.info(f"Type: {model_type}")
    logger.info(f"Config: {exp_config}")
    
    # Create model
    if model_type == 'cbow':
        model = CBOW(vocab.vocab_size, exp_config['embedding_dim'])
        # Prepare data
        training_pairs = create_training_pairs(
            encoded_sentences,
            exp_config['window_size']
        )
        dataset = CBOWDataset(training_pairs, vocab, exp_config['negative_samples'])
    else:  # skipgram
        model = SkipGram(vocab.vocab_size, exp_config['embedding_dim'])
        # Prepare data
        training_pairs = create_skipgram_pairs(
            encoded_sentences,
            exp_config['window_size']
        )
        dataset = SkipGramDataset(training_pairs, vocab, exp_config['negative_samples'])
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=exp_config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    # Create trainer
    trainer = Word2VecTrainer(model, vocab, exp_config)
    
    # Train
    history = trainer.train(
        data_loader,
        epochs=exp_config['epochs'],
        model_type=model_type,
        save_dir=f'outputs/models/{model_name}'
    )
    
    # Save final model
    model_path = f'outputs/models/{model_name}_final.pt'
    trainer.save_model(model_path)
    
    # Save embeddings
    embeddings_path = f'outputs/embeddings/{model_name}_embeddings.npy'
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    model.save_embeddings(embeddings_path)
    
    # Save history
    history_path = f'outputs/results/{model_name}_history.json'
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    trainer.save_history(history_path)
    
    logger.info(f"Model {model_name} training complete")


def task_analyze(config):
    """Task 4: Perform semantic analysis."""
    logger.info("="*60)
    logger.info("TASK: SEMANTIC ANALYSIS")
    logger.info("="*60)
    
    from word2vec.vocabulary import Vocabulary
    from word2vec.cbow import CBOW
    from word2vec.skipgram import SkipGram
    from analysis.similarity import analyze_required_words
    
    # Load vocabulary
    vocab = Vocabulary.load('data/processed/vocabulary.pkl')
    
    # Analyze baseline models
    for model_type in ['cbow', 'skipgram']:
        model_name = f"{model_type}_baseline"
        logger.info(f"\nAnalyzing {model_name}...")
        
        # Load model
        if model_type == 'cbow':
            model = CBOW(vocab.vocab_size, 100)
        else:
            model = SkipGram(vocab.vocab_size, 100)
        
        model_path = f'outputs/models/{model_name}_final.pt'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            
            # Analyze
            output_dir = f'outputs/results/{model_name}'
            analyze_required_words(model, vocab, output_dir)
        else:
            logger.warning(f"Model not found: {model_path}")


def task_visualize(config):
    """Task 5: Create visualizations."""
    logger.info("="*60)
    logger.info("TASK: VISUALIZATION")
    logger.info("="*60)
    
    logger.info("Visualization task - implement PCA and t-SNE visualizations")
    logger.info("See IMPLEMENTATION_GUIDE.md for details")


def task_all(config):
    """Run complete pipeline."""
    logger.info("="*60)
    logger.info("RUNNING COMPLETE PIPELINE")
    logger.info("="*60)
    
    # Check if data exists
    if not os.path.exists('data/raw') or len(os.listdir('data/raw')) == 0:
        logger.info("\n>>> Running data collection...")
        task_collect_data(config)
    else:
        logger.info("\n>>> Skipping data collection (data exists)")
    
    # Check if preprocessed corpus exists
    if not os.path.exists('data/processed/corpus.txt'):
        logger.info("\n>>> Running preprocessing...")
        task_preprocess(config)
    else:
        logger.info("\n>>> Skipping preprocessing (corpus exists)")
    
    # Train models
    logger.info("\n>>> Training models...")
    task_train_models(config)
    
    # Analysis
    logger.info("\n>>> Running analysis...")
    task_analyze(config)
    
    # Visualization
    logger.info("\n>>> Creating visualizations...")
    task_visualize(config)
    
    logger.info("\n" + "="*60)
    logger.info("COMPLETE PIPELINE FINISHED")
    logger.info("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Problem 1: Word Embeddings from IIT Jodhpur Data'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='all',
        choices=['all', 'collect_data', 'preprocess', 'train_cbow', 
                'train_skipgram', 'train_all', 'analyze', 'visualize'],
        help='Task to run'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Run requested task
    if args.task == 'all':
        task_all(config)
    elif args.task == 'collect_data':
        task_collect_data(config)
    elif args.task == 'preprocess':
        task_preprocess(config)
    elif args.task == 'train_all':
        task_train_models(config)
    elif args.task == 'analyze':
        task_analyze(config)
    elif args.task == 'visualize':
        task_visualize(config)
    else:
        logger.error(f"Unknown task: {args.task}")
        sys.exit(1)


if __name__ == "__main__":
    main()
