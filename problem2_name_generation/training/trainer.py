"""
Training Module for Character-Level RNN Name Generation
Dhruva Kumar Kaushal (B22AI017)
NLU Assignment - Problem 2

Complete training pipeline for all three RNN models.
Includes data loading, training loop, validation, and checkpointing.

Original implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import logging
import json
import time
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.base_model import CharacterDataset, collate_fn


class Trainer:
    """
    Trainer class for RNN name generation models.
    
    Handles:
    - Data loading and splitting
    - Training loop with backpropagation
    - Validation and early stopping
    - Checkpointing and logging
    """
    
    def __init__(self, model, config: Dict, vocab, device: str = 'cpu'):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration dictionary
            vocab: CharacterVocabulary instance
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.vocab = vocab
        self.device = device
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Loss function (ignore padding)
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.char2idx[vocab.PAD_TOKEN])
        
        # Optimizer
        self.optimizer = self._setup_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._setup_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        
        self.logger.info(f"Trainer initialized for {model.__class__.__name__}")
        self.logger.info(f"Device: {device}")
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.0001)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_name = self.config.get('scheduler', 'reduce_on_plateau')
        
        if scheduler_name == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.get('scheduler_factor', 0.5),
                patience=self.config.get('scheduler_patience', 5)
            )
        else:
            return None
    
    def prepare_data(self, data_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load and prepare data for training.
        
        Args:
            data_path: Path to indian_names.json
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        self.logger.info("Loading and preparing data...")
        
        # Load names
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, list):
            names = data
        elif isinstance(data, dict) and 'names' in data:
            names = data['names']
        else:
            raise ValueError(f"Unexpected data format in {data_path}")
        
        self.logger.info(f"Loaded {len(names)} names")
        
        # Encode names
        encoded_names = []
        for name_entry in names:
            name = name_entry['name']
            encoded = self.vocab.encode(name, add_special_tokens=True)
            encoded_names.append(encoded)
        
        self.logger.info(f"Encoded {len(encoded_names)} names")
        
        # Create dataset
        dataset = CharacterDataset(encoded_names)
        
        # Split into train/val/test
        val_split = self.config.get('validation_split', 0.15)
        test_split = self.config.get('test_split', 0.15)
        
        n_total = len(dataset)
        n_test = int(n_total * test_split)
        n_val = int(n_total * val_split)
        n_train = n_total - n_test - n_val
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.config.get('seed', 42))
        )
        
        self.logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Create data loaders
        batch_size = self.config.get('batch_size', 64)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for inputs, targets, lengths in progress_bar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, _ = self.model(inputs)
            
            # Compute loss
            # Reshape for cross entropy: (batch_size * seq_len, vocab_size)
            logits = logits.view(-1, len(self.vocab))
            targets = targets.view(-1)
            
            loss = self.criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / n_batches
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for inputs, targets, lengths in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                logits, _ = self.model(inputs)
                
                # Compute loss
                logits = logits.view(-1, len(self.vocab))
                targets = targets.view(-1)
                
                loss = self.criterion(logits, targets)
                
                total_loss += loss.item()
                n_batches += 1
        
        avg_loss = total_loss / n_batches
        return avg_loss
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              output_dir: str) -> Dict:
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            output_dir: Directory to save outputs
            
        Returns:
            Training history dictionary
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        num_epochs = self.config.get('num_epochs', 50)
        patience = self.config.get('patience', 10)
        save_every = self.config.get('save_every', 5)
        
        self.logger.info("=" * 80)
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Log progress
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs} - "
                           f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = output_path / f"checkpoint_epoch_{epoch + 1}.pt"
                self.model.save_checkpoint(
                    str(checkpoint_path),
                    epoch + 1,
                    self.optimizer.state_dict(),
                    val_loss,
                    metadata={'train_loss': train_loss}
                )
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                best_model_path = output_path / "best_model.pt"
                self.model.save_checkpoint(
                    str(best_model_path),
                    epoch + 1,
                    self.optimizer.state_dict(),
                    val_loss,
                    metadata={'train_loss': train_loss, 'best_epoch': epoch + 1}
                )
                self.logger.info(f"New best model saved (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.config.get('early_stopping', True) and self.patience_counter >= patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        elapsed_time = time.time() - start_time
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1,
            'training_time_seconds': elapsed_time
        }
        
        history_path = output_path / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.logger.info("=" * 80)
        self.logger.info(f"Training complete! Time: {elapsed_time:.2f}s")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info("=" * 80)
        
        return history
