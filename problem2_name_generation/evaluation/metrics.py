"""
Evaluation Module for Character-Level RNN Name Generation
Dhruva Kumar Kaushal (B22AI017)
NLU Assignment - Problem 2

Comprehensive evaluation metrics for name generation models.
Includes perplexity, novelty, diversity, and linguistic analysis.

Original implementation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import json
from typing import Dict, List, Set
import numpy as np
from collections import Counter
import math

import sys
sys.path.append(str(Path(__file__).parent.parent))


class ModelEvaluator:
    """
    Evaluator for name generation models.
    
    Computes various metrics:
    - Perplexity
    - Novelty (new names not in training)
    - Diversity (unique generated names)
    - Character distribution
    - Length distribution
    """
    
    def __init__(self, model, vocab, device: str = 'cpu'):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model to evaluate
            vocab: CharacterVocabulary instance
            device: Device to run on
        """
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=vocab.char2idx[vocab.PAD_TOKEN],
            reduction='sum'
        )
    
    def compute_perplexity(self, test_loader: DataLoader) -> float:
        """
        Compute perplexity on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Perplexity value
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for inputs, targets, lengths in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                logits, _ = self.model(inputs)
                
                # Compute loss
                logits = logits.view(-1, len(self.vocab))
                targets = targets.view(-1)
                
                loss = self.criterion(logits, targets)
                
                total_loss += loss.item()
                total_tokens += (targets != self.vocab.char2idx[self.vocab.PAD_TOKEN]).sum().item()
        
        # Perplexity = exp(total_loss / total_tokens)
        perplexity = math.exp(total_loss / total_tokens)
        
        return perplexity
    
    def generate_samples(self, n_samples: int = 100, max_length: int = 15,
                        temperature: float = 0.8) -> List[str]:
        """
        Generate name samples.
        
        Args:
            n_samples: Number of names to generate
            max_length: Maximum length per name
            temperature: Sampling temperature
            
        Returns:
            List of generated names
        """
        self.model.eval()
        generated_names = []
        
        for _ in range(n_samples):
            # Generate character indices
            char_indices = self.model.generate(
                max_length=max_length,
                temperature=temperature,
                device=self.device
            )
            
            # Decode to string
            name = self.vocab.decode(char_indices, remove_special_tokens=True)
            generated_names.append(name)
        
        return generated_names
    
    def compute_novelty(self, generated_names: List[str], 
                       training_names: Set[str]) -> float:
        """
        Compute novelty: fraction of generated names not in training set.
        
        Args:
            generated_names: List of generated names
            training_names: Set of training names
            
        Returns:
            Novelty score (0-1)
        """
        novel_count = sum(1 for name in generated_names if name not in training_names)
        novelty = novel_count / len(generated_names) if generated_names else 0.0
        
        return novelty
    
    def compute_diversity(self, generated_names: List[str]) -> float:
        """
        Compute diversity: fraction of unique names.
        
        Args:
            generated_names: List of generated names
            
        Returns:
            Diversity score (0-1)
        """
        unique_count = len(set(generated_names))
        diversity = unique_count / len(generated_names) if generated_names else 0.0
        
        return diversity
    
    def analyze_character_distribution(self, names: List[str]) -> Dict:
        """
        Analyze character distribution in names.
        
        Args:
            names: List of names
            
        Returns:
            Character distribution statistics
        """
        all_chars = ''.join(names)
        char_counts = Counter(all_chars)
        
        total_chars = len(all_chars)
        char_freq = {char: count / total_chars for char, count in char_counts.items()}
        
        return {
            'total_characters': total_chars,
            'unique_characters': len(char_counts),
            'most_common': char_counts.most_common(10),
            'frequencies': char_freq
        }
    
    def analyze_length_distribution(self, names: List[str]) -> Dict:
        """
        Analyze length distribution of names.
        
        Args:
            names: List of names
            
        Returns:
            Length distribution statistics
        """
        lengths = [len(name) for name in names]
        
        return {
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'median_length': np.median(lengths)
        }
    
    def evaluate_comprehensive(self, test_loader: DataLoader,
                             training_names: Set[str],
                             n_samples: int = 100,
                             temperature: float = 0.8) -> Dict:
        """
        Comprehensive evaluation.
        
        Args:
            test_loader: Test data loader
            training_names: Set of training names
            n_samples: Number of samples to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Starting comprehensive evaluation...")
        
        # Compute perplexity
        self.logger.info("Computing perplexity...")
        perplexity = self.compute_perplexity(test_loader)
        
        # Generate samples
        self.logger.info(f"Generating {n_samples} samples...")
        generated_names = self.generate_samples(n_samples, temperature=temperature)
        
        # Filter out empty names
        generated_names = [name for name in generated_names if name]
        
        # Compute novelty
        self.logger.info("Computing novelty...")
        novelty = self.compute_novelty(generated_names, training_names)
        
        # Compute diversity
        self.logger.info("Computing diversity...")
        diversity = self.compute_diversity(generated_names)
        
        # Analyze distributions
        self.logger.info("Analyzing character distribution...")
        char_dist = self.analyze_character_distribution(generated_names)
        
        self.logger.info("Analyzing length distribution...")
        length_dist = self.analyze_length_distribution(generated_names)
        
        # Compile results
        results = {
            'model_name': self.model.__class__.__name__,
            'perplexity': perplexity,
            'novelty': novelty,
            'diversity': diversity,
            'n_generated': len(generated_names),
            'n_unique': len(set(generated_names)),
            'character_distribution': {
                'total_chars': char_dist['total_characters'],
                'unique_chars': char_dist['unique_characters'],
                'most_common_chars': char_dist['most_common']
            },
            'length_distribution': length_dist,
            'sample_names': generated_names[:20]  # First 20 samples
        }
        
        self.logger.info("Evaluation complete!")
        self.logger.info(f"Perplexity: {perplexity:.2f}")
        self.logger.info(f"Novelty: {novelty:.2%}")
        self.logger.info(f"Diversity: {diversity:.2%}")
        
        return results
    
    def save_results(self, results: Dict, output_path: str):
        """
        Save evaluation results to JSON.
        
        Args:
            results: Evaluation results dictionary
            output_path: Path to save results
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
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
        
        results_native = convert_to_native(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_native, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to {output_path}")
