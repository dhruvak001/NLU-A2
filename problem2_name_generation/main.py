"""
Main Orchestration Script for Problem 2: Name Generation
Dhruva Kumar Kaushal (B22AI017)
NLU Assignment - Problem 2

This script orchestrates the complete pipeline for Indian name generation
using character-level RNN models.

Tasks:
- generate_data: Generate Indian names dataset using LLM
- train: Train all three RNN models
- evaluate: Evaluate models on test set
- generate: Generate name samples
- all: Run complete pipeline

Original implementation.
"""

import argparse
import logging
import yaml
from pathlib import Path
import sys
import torch
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from preprocessing.generate_names import IndianNameGenerator
from preprocessing.character_vocab import CharacterVocabulary
from models.vanilla_rnn import VanillaRNN
from models.blstm import BLSTM
from models.rnn_attention import RNNWithAttention
from training.trainer import Trainer
from evaluation.metrics import ModelEvaluator
from evaluation.generate_samples import NameGenerator


def setup_logging(verbose: bool = True):
    """Setup logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/main.log'),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str = 'config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def task_generate_data(config: dict):
    """Generate Indian names dataset."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("TASK: Generate Data")
    logger.info("=" * 80)
    
    # Create output directory
    output_path = Path(config['paths']['raw_data']) / 'indian_names.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    data_config = config['data_generation']
    generator = IndianNameGenerator(data_config)
    
    # Generate dataset
    n_names = data_config.get('n_names', 5000)
    result = generator.generate_dataset(n_names, output_path)
    
    logger.info(f"Generated {result['statistics']['total_generated']} names")
    logger.info("Data generation complete!")


def task_build_vocab(config: dict):
    """Build character vocabulary from generated names."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("TASK: Build Vocabulary")
    logger.info("=" * 80)
    
    # Paths
    data_path = Path(config['paths']['raw_data']) / 'indian_names.json'
    vocab_path = Path(config['paths']['processed_data']) / 'char_vocab.json'
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize vocabulary
    vocab_config = config['vocabulary']
    vocab = CharacterVocabulary(
        lowercase=vocab_config.get('lowercase', True),
        max_vocab_size=vocab_config.get('max_vocab_size', 100)
    )
    
    # Build from dataset
    stats = vocab.build_from_dataset(data_path)
    
    # Save vocabulary
    vocab.save(vocab_path)
    
    logger.info(f"Vocabulary size: {stats['vocab_size']}")
    logger.info("Vocabulary building complete!")


def task_train(config: dict):
    """Train all three RNN models."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("TASK: Train Models")
    logger.info("=" * 80)
    
    # Load vocabulary
    vocab_path = Path(config['paths']['processed_data']) / 'char_vocab.json'
    vocab = CharacterVocabulary.load(vocab_path)
    
    logger.info(f"Loaded vocabulary: {len(vocab)} characters")
    
    # Data path
    data_path = Path(config['paths']['raw_data']) / 'indian_names.json'
    
    # Model configurations
    models_config = config['models']
    training_config = config['training']
    
    device = torch.device('cuda' if torch.cuda.is_available() and training_config.get('use_cuda', True) else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize models
    pad_idx = vocab.char2idx[vocab.PAD_TOKEN]
    sos_idx = vocab.char2idx[vocab.SOS_TOKEN]
    eos_idx = vocab.char2idx[vocab.EOS_TOKEN]
    
    models = {}
    
    # Vanilla RNN
    logger.info("\n" + "=" * 80)
    logger.info("Training Vanilla RNN")
    logger.info("=" * 80)
    rnn_config = models_config['vanilla_rnn']
    models['vanilla_rnn'] = VanillaRNN(
        vocab_size=len(vocab),
        embedding_dim=rnn_config['embedding_dim'],
        hidden_dim=rnn_config['hidden_dim'],
        num_layers=rnn_config['num_layers'],
        dropout=rnn_config['dropout'],
        pad_idx=pad_idx,
        sos_idx=sos_idx,
        eos_idx=eos_idx
    )
    
    trainer_rnn = Trainer(models['vanilla_rnn'], training_config, vocab, str(device))
    train_loader, val_loader, test_loader = trainer_rnn.prepare_data(str(data_path))
    history_rnn = trainer_rnn.train(train_loader, val_loader, 'outputs/vanilla_rnn')
    
    # BLSTM
    logger.info("\n" + "=" * 80)
    logger.info("Training Bidirectional LSTM")
    logger.info("=" * 80)
    blstm_config = models_config['blstm']
    models['blstm'] = BLSTM(
        vocab_size=len(vocab),
        embedding_dim=blstm_config['embedding_dim'],
        hidden_dim=blstm_config['hidden_dim'],
        num_layers=blstm_config['num_layers'],
        dropout=blstm_config['dropout'],
        pad_idx=pad_idx,
        sos_idx=sos_idx,
        eos_idx=eos_idx
    )
    
    trainer_blstm = Trainer(models['blstm'], training_config, vocab, str(device))
    train_loader, val_loader, test_loader = trainer_blstm.prepare_data(str(data_path))
    history_blstm = trainer_blstm.train(train_loader, val_loader, 'outputs/blstm')
    
    # RNN with Attention
    logger.info("\n" + "=" * 80)
    logger.info("Training RNN with Attention")
    logger.info("=" * 80)
    attn_config = models_config['rnn_attention']
    models['rnn_attention'] = RNNWithAttention(
        vocab_size=len(vocab),
        embedding_dim=attn_config['embedding_dim'],
        hidden_dim=attn_config['hidden_dim'],
        num_layers=attn_config['num_layers'],
        attention_dim=attn_config['attention_dim'],
        dropout=attn_config['dropout'],
        pad_idx=pad_idx,
        sos_idx=sos_idx,
        eos_idx=eos_idx
    )
    
    trainer_attn = Trainer(models['rnn_attention'], training_config, vocab, str(device))
    train_loader, val_loader, test_loader = trainer_attn.prepare_data(str(data_path))
    history_attn = trainer_attn.train(train_loader, val_loader, 'outputs/rnn_attention')
    
    logger.info("\n" + "=" * 80)
    logger.info("All models trained successfully!")
    logger.info("=" * 80)


def task_evaluate(config: dict):
    """Evaluate trained models."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("TASK: Evaluate Models")
    logger.info("=" * 80)
    
    # Load vocabulary
    vocab_path = Path(config['paths']['processed_data']) / 'char_vocab.json'
    vocab = CharacterVocabulary.load(vocab_path)
    
    # Load training names for novelty computation
    data_path = Path(config['paths']['raw_data']) / 'indian_names.json'
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, list):
        training_names = set([name_entry['name'].lower() for name_entry in data])
    elif isinstance(data, dict) and 'names' in data:
        training_names = set([name_entry['name'].lower() for name_entry in data['names']])
    else:
        training_names = set()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize models
    pad_idx = vocab.char2idx[vocab.PAD_TOKEN]
    sos_idx = vocab.char2idx[vocab.SOS_TOKEN]
    eos_idx = vocab.char2idx[vocab.EOS_TOKEN]
    
    models_config = config['models']
    generation_config = config['generation']
    
    model_names = ['vanilla_rnn', 'blstm', 'rnn_attention']
    all_results = {}
    
    for model_name in model_names:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Evaluating {model_name}")
        logger.info('=' * 80)
        
        # Load trained model
        model_path = Path('outputs') / model_name / 'best_model.pt'
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}. Skipping...")
            continue
        
        # Initialize model architecture
        model_config = models_config[model_name]
        if model_name == 'vanilla_rnn':
            model = VanillaRNN(
                vocab_size=len(vocab),
                embedding_dim=model_config['embedding_dim'],
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                dropout=model_config['dropout'],
                pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx
            )
        elif model_name == 'blstm':
            model = BLSTM(
                vocab_size=len(vocab),
                embedding_dim=model_config['embedding_dim'],
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                dropout=model_config['dropout'],
                pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx
            )
        elif model_name == 'rnn_attention':
            model = RNNWithAttention(
                vocab_size=len(vocab),
                embedding_dim=model_config['embedding_dim'],
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                attention_dim=model_config['attention_dim'],
                dropout=model_config['dropout'],
                pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx
            )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Create evaluator
        evaluator = ModelEvaluator(model, vocab, str(device))
        
        # Prepare test data
        from training.trainer import Trainer
        temp_trainer = Trainer(model, config['training'], vocab, str(device))
        _, _, test_loader = temp_trainer.prepare_data(str(data_path))
        
        # Comprehensive evaluation
        results = evaluator.evaluate_comprehensive(
            test_loader,
            training_names,
            n_samples=generation_config.get('num_samples', 100),
            temperature=generation_config.get('temperature', 0.8)
        )
        
        # Save results
        output_path = Path('outputs') / model_name / 'evaluation_results.json'
        evaluator.save_results(results, str(output_path))
        
        all_results[model_name] = results
    
    # Save comparison
    comparison_path = Path('outputs') / 'model_comparison.json'
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        import numpy as np
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    all_results_native = convert_to_native(all_results)
    
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(all_results_native, f, indent=2, ensure_ascii=False)
    
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation complete!")
    logger.info("=" * 80)


def task_generate(config: dict):
    """Generate name samples from trained models."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("TASK: Generate Samples")
    logger.info("=" * 80)
    
    # Load vocabulary
    vocab_path = Path(config['paths']['processed_data']) / 'char_vocab.json'
    vocab = CharacterVocabulary.load(vocab_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize models
    pad_idx = vocab.char2idx[vocab.PAD_TOKEN]
    sos_idx = vocab.char2idx[vocab.SOS_TOKEN]
    eos_idx = vocab.char2idx[vocab.EOS_TOKEN]
    
    models_config = config['models']
    generation_config = config['generation']
    
    model_names = ['vanilla_rnn', 'blstm', 'rnn_attention']
    
    for model_name in model_names:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Generating samples from {model_name}")
        logger.info('=' * 80)
        
        # Load trained model
        model_path = Path('outputs') / model_name / 'best_model.pt'
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}. Skipping...")
            continue
        
        # Initialize model architecture
        model_config = models_config[model_name]
        if model_name == 'vanilla_rnn':
            model = VanillaRNN(
                vocab_size=len(vocab),
                embedding_dim=model_config['embedding_dim'],
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                dropout=model_config['dropout'],
                pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx
            )
        elif model_name == 'blstm':
            model = BLSTM(
                vocab_size=len(vocab),
                embedding_dim=model_config['embedding_dim'],
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                dropout=model_config['dropout'],
                pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx
            )
        elif model_name == 'rnn_attention':
            model = RNNWithAttention(
                vocab_size=len(vocab),
                embedding_dim=model_config['embedding_dim'],
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                attention_dim=model_config['attention_dim'],
                dropout=model_config['dropout'],
                pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx
            )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Create generator
        generator = NameGenerator(model, vocab, str(device))
        
        # Generate names
        generated_names = generator.generate(
            n_samples=generation_config.get('num_samples', 100),
            max_length=generation_config.get('max_length', 15),
            temperature=generation_config.get('temperature', 0.8),
            top_k=generation_config.get('top_k', 0),
            top_p=generation_config.get('top_p', 0.9),
            min_length=generation_config.get('min_length', 3),
            filter_duplicates=generation_config.get('filter_duplicates', True)
        )
        
        # Save generated names
        output_path = Path('outputs') / model_name / 'generated_names.json'
        generator.save_generated_names(
            generated_names,
            str(output_path),
            metadata={
                'temperature': generation_config.get('temperature', 0.8),
                'max_length': generation_config.get('max_length', 15)
            }
        )
        
        # Display samples
        logger.info(f"\nSample generated names from {model_name}:")
        for i, name in enumerate(generated_names[:20], 1):
            logger.info(f"  {i:2d}. {name}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Generation complete!")
    logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Character-Level RNN Name Generation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--task',
        type=str,
        default='all',
        choices=['all', 'generate_data', 'build_vocab', 'train', 'evaluate', 'generate'],
        help='Task to run'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Verbose logging'
    )
    
    args = parser.parse_args()
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Print header
    logger.info("")
    logger.info("=" * 80)
    logger.info("CHARACTER-LEVEL RNN NAME GENERATION")
    logger.info("Dhruva Kumar Kaushal (B22AI017)")
    logger.info("NLU Assignment - Problem 2")
    logger.info("=" * 80)
    logger.info("")
    
    # Run tasks
    try:
        if args.task == 'all':
            task_generate_data(config)
            task_build_vocab(config)
            task_train(config)
            task_evaluate(config)
            task_generate(config)
        elif args.task == 'generate_data':
            task_generate_data(config)
        elif args.task == 'build_vocab':
            task_build_vocab(config)
        elif args.task == 'train':
            task_train(config)
        elif args.task == 'evaluate':
            task_evaluate(config)
        elif args.task == 'generate':
            task_generate(config)
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETE!")
        logger.info("=" * 80)
        logger.info("")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
