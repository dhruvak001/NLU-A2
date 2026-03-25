"""
Name Generation Module
Dhruva Kumar Kaushal (B22AI017)
NLU Assignment - Problem 2

Generate and save name samples from trained models.
Supports various sampling strategies and filtering.

Original implementation.
"""

import torch
from pathlib import Path
import logging
import json
from typing import Dict, List
import sys

sys.path.append(str(Path(__file__).parent.parent))


class NameGenerator:
    """
    Name generator for trained RNN models.
    
    Provides various sampling strategies and filtering options.
    """
    
    def __init__(self, model, vocab, device: str = 'cpu'):
        """
        Initialize generator.
        
        Args:
            model: Trained model
            vocab: CharacterVocabulary instance
            device: Device to run on
        """
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate(self, n_samples: int = 100, max_length: int = 15,
                temperature: float = 0.8, top_k: int = 0, top_p: float = 0.9,
                min_length: int = 3, filter_duplicates: bool = True) -> List[str]:
        """
        Generate names with filtering.
        
        Args:
            n_samples: Number of names to generate
            max_length: Maximum name length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            min_length: Minimum name length
            filter_duplicates: Remove duplicates
            
        Returns:
            List of generated names
        """
        self.model.eval()
        self.logger.info(f"Generating {n_samples} names...")
        
        generated_names = []
        seen_names = set()
        attempts = 0
        max_attempts = n_samples * 3  # Try up to 3x to get unique names
        
        while len(generated_names) < n_samples and attempts < max_attempts:
            attempts += 1
            
            # Generate character indices
            char_indices = self.model.generate(
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=self.device
            )
            
            # Decode to string
            name = self.vocab.decode(char_indices, remove_special_tokens=True)
            
            # Apply filters
            if len(name) < min_length:
                continue
            
            if filter_duplicates and name in seen_names:
                continue
            
            generated_names.append(name)
            seen_names.add(name)
        
        self.logger.info(f"Generated {len(generated_names)} names in {attempts} attempts")
        
        return generated_names
    
    def generate_batch(self, n_samples: int = 100, **kwargs) -> List[str]:
        """
        Generate names in batch mode.
        
        Args:
            n_samples: Number of names to generate
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated names
        """
        return self.generate(n_samples=n_samples, **kwargs)
    
    def generate_with_variations(self, n_samples_per_temp: int = 50,
                                temperatures: List[float] = [0.5, 0.8, 1.0],
                                **kwargs) -> Dict[str, List[str]]:
        """
        Generate names with different temperature settings.
        
        Args:
            n_samples_per_temp: Samples per temperature
            temperatures: List of temperatures to try
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary mapping temperature to generated names
        """
        results = {}
        
        for temp in temperatures:
            self.logger.info(f"Generating with temperature={temp}")
            names = self.generate(
                n_samples=n_samples_per_temp,
                temperature=temp,
                **kwargs
            )
            results[f"temp_{temp}"] = names
        
        return results
    
    def save_generated_names(self, names: List[str], output_path: str,
                           metadata: Dict = None):
        """
        Save generated names to file.
        
        Args:
            names: List of generated names
            output_path: Path to save file
            metadata: Additional metadata
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'model_name': self.model.__class__.__name__,
            'n_generated': len(names),
            'n_unique': len(set(names)),
            'metadata': metadata or {},
            'names': names
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(names)} names to {output_path}")
        
        # Also save as plain text
        txt_path = Path(output_path).with_suffix('.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            for name in names:
                f.write(name + '\n')
        
        self.logger.info(f"Saved plain text to {txt_path}")
