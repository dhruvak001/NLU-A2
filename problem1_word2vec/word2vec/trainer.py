"""
Word2Vec Trainer
================

This module implements the training loop for CBOW and Skip-gram models.

Author: Dhruva Kumar Kaushal (B22AI017)
Date: March 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
import json
import os
from typing import List, Tuple, Dict
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CBOWDataset(Dataset):
    """Dataset for CBOW training."""
    
    def __init__(self, training_pairs: List[Tuple[List[int], int]], 
                 vocab, n_negative: int = 10):
        self.training_pairs = training_pairs
        self.vocab = vocab
        self.n_negative = n_negative
        # Find max context length for padding
        self.max_context_len = max(len(context) for context, _ in training_pairs)
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx):
        context, target = self.training_pairs[idx]
        
        # Pad context to max length
        padded_context = context + [0] * (self.max_context_len - len(context))
        
        # Sample negative examples
        negative_indices = self.vocab.sample_negative(self.n_negative, {target})
        
        return (
            torch.tensor(padded_context, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(negative_indices, dtype=torch.long)
        )


class Word2VecTrainer:
    """
    Trainer for Word2Vec models (CBOW and Skip-gram).
    
    Handles:
    - Training loop
    - Loss computation
    - Checkpointing
    - Logging and monitoring
    """
    
    def __init__(self, model, vocab, config: Dict, device=None):
        """
        Initialize trainer.
        
        Args:
            model: CBOW or Skip-gram model
            vocab: Vocabulary object
            config: Training configuration
            device: Device to train on (CPU or CUDA)
        """
        self.model = model
        self.vocab = vocab
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 0.001)
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'epoch_times': [],
            'learning_rates': []
        }
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def train_cbow_epoch(self, data_loader) -> float:
        """Train one epoch for CBOW model."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for context, target, negatives in tqdm(data_loader, desc="Training"):
            context = context.to(self.device)
            target = target.to(self.device)
            negatives = negatives.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass - get context representation
            output = self.model(context)
            
            # Get embeddings for loss computation
            # For CBOW, we use the averaged context embedding
            batch_size = context.size(0)
            context_embeds = self.model.embeddings(context)
            avg_context = torch.mean(context_embeds, dim=1)
            
            # Get target and negative embeddings
            target_embeds = self.model.linear.weight[target]
            negative_embeds = self.model.linear.weight[negatives]
            
            # Compute negative sampling loss
            positive_score = torch.sum(avg_context * target_embeds, dim=1)
            positive_loss = -torch.nn.functional.logsigmoid(positive_score)
            
            negative_score = torch.bmm(
                negative_embeds,
                avg_context.unsqueeze(2)
            ).squeeze(2)
            negative_loss = -torch.sum(
                torch.nn.functional.logsigmoid(-negative_score), dim=1
            )
            
            loss = torch.mean(positive_loss + negative_loss)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def train_skipgram_epoch(self, data_loader) -> float:
        """Train one epoch for Skip-gram model."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for target, context, negatives in tqdm(data_loader, desc="Training"):
            target = target.to(self.device)
            context = context.to(self.device)
            negatives = negatives.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass (computes loss internally)
            loss = self.model(target, context, negatives)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, data_loader, epochs: int, model_type: str = 'cbow',
              save_dir: str = None) -> Dict:
        """
        Train the model for specified number of epochs.
        
        Args:
            data_loader: DataLoader for training data
            epochs: Number of epochs to train
            model_type: 'cbow' or 'skipgram'
            save_dir: Directory to save checkpoints
        
        Returns:
            dict: Training history
        """
        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Model type: {model_type}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train one epoch
            if model_type == 'cbow':
                epoch_loss = self.train_cbow_epoch(data_loader)
            else:
                epoch_loss = self.train_skipgram_epoch(data_loader)
            
            epoch_time = time.time() - start_time
            
            # Record history
            self.history['train_loss'].append(epoch_loss)
            self.history['epoch_times'].append(epoch_time)
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Log progress
            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Loss: {epoch_loss:.4f} - "
                    f"Time: {epoch_time:.2f}s"
                )
            
            # Save checkpoint
            if save_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(save_dir, epoch)
        
        logger.info("Training completed!")
        return self.history
    
    def save_checkpoint(self, save_dir: str, epoch: int):
        """Save model checkpoint."""
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_model(self, filepath: str):
        """Save final model."""
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"Model saved: {filepath}")
    
    def save_history(self, filepath: str):
        """Save training history."""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"History saved: {filepath}")


def prepare_cbow_data(sentences: List[List[int]], window_size: int):
    """Prepare training data for CBOW."""
    from .cbow import create_training_pairs
    pairs = create_training_pairs(sentences, window_size)
    return pairs


def prepare_skipgram_data(sentences: List[List[int]], window_size: int):
    """Prepare training data for Skip-gram."""
    from .skipgram import create_skipgram_pairs
    pairs = create_skipgram_pairs(sentences, window_size)
    return pairs


if __name__ == "__main__":
    print("Word2Vec Trainer Module")
    print("Use this module through the main training script")
