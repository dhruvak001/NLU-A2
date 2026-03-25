"""
Base Model Class for Character-Level RNN Name Generation
Dhruva Kumar Kaushal (B22AI017)
NLU Assignment - Problem 2

This module defines the base class for all RNN models.
Provides common functionality for training, generation, and evaluation.

Original implementation following PyTorch best practices.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
import logging


class BaseNameGenerationModel(nn.Module, ABC):
    """
    Abstract base class for name generation models.
    
    All RNN architectures (Vanilla RNN, BLSTM, RNN+Attention) inherit from this.
    Provides common interface for training and generation.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 pad_idx: int, sos_idx: int, eos_idx: int):
        """
        Initialize base model.
        
        Args:
            vocab_size: Size of character vocabulary
            embedding_dim: Dimension of character embeddings
            hidden_dim: Dimension of hidden states
            pad_idx: Index of padding token
            sos_idx: Index of start-of-sequence token
            eos_idx: Index of end-of-sequence token
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Character embedding layer (shared across all models)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )
        
        # Output projection layer (shared across all models)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        self.logger.info(f"Initialized {self.__class__.__name__} "
                        f"(vocab={vocab_size}, emb={embedding_dim}, hidden={hidden_dim})")
    
    @abstractmethod
    def forward(self, input_seq: torch.Tensor, 
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_seq: Input sequence tensor (batch_size, seq_len)
            hidden: Optional initial hidden state
            
        Returns:
            Tuple of (output_logits, final_hidden)
        """
        pass
    
    @abstractmethod
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """
        Initialize hidden state.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial hidden state tensor
        """
        pass
    
    def generate(self, start_token: int = None, max_length: int = 15,
                 temperature: float = 1.0, top_k: int = 0, top_p: float = 0.9,
                 device: str = 'cpu') -> List[int]:
        """
        Generate a name character-by-character.
        
        Args:
            start_token: Starting token (defaults to SOS)
            max_length: Maximum generation length
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (0 = disabled)
            top_p: Nucleus sampling threshold
            device: Device to run on
            
        Returns:
            List of generated character indices
        """
        self.eval()
        
        if start_token is None:
            start_token = self.sos_idx
        
        generated = [start_token]
        hidden = self.init_hidden(1)
        
        # Move hidden state to device (handle both single tensor and tuple)
        if isinstance(hidden, tuple):
            hidden = tuple(h.to(device) for h in hidden)
        else:
            hidden = hidden.to(device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Prepare input
                input_tensor = torch.tensor([[generated[-1]]], device=device)
                
                # Forward pass
                output, hidden = self.forward(input_tensor, hidden)
                
                # Get logits for next character
                logits = output[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits[top_k_indices] = top_k_logits
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[indices_to_remove] = float('-inf')
                
                # Sample from distribution
                probs = torch.softmax(logits, dim=-1)
                next_char = torch.multinomial(probs, num_samples=1).item()
                
                generated.append(next_char)
                
                # Stop if EOS token generated
                if next_char == self.eos_idx:
                    break
        
        return generated
    
    def generate_batch(self, batch_size: int, max_length: int = 15,
                      temperature: float = 1.0, device: str = 'cpu') -> List[List[int]]:
        """
        Generate multiple names in batch.
        
        Args:
            batch_size: Number of names to generate
            max_length: Maximum length per name
            temperature: Sampling temperature
            device: Device to run on
            
        Returns:
            List of generated sequences
        """
        generated_sequences = []
        
        for _ in range(batch_size):
            sequence = self.generate(
                max_length=max_length,
                temperature=temperature,
                device=device
            )
            generated_sequences.append(sequence)
        
        return generated_sequences
    
    def get_model_info(self) -> Dict:
        """
        Get model information and statistics.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }
    
    def save_checkpoint(self, path: str, epoch: int, optimizer_state: Dict,
                       loss: float, metadata: Dict = None):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            optimizer_state: Optimizer state dict
            loss: Current loss value
            metadata: Additional metadata to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer_state,
            'loss': loss,
            'model_info': self.get_model_info(),
            'metadata': metadata or {}
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    @classmethod
    def load_checkpoint(cls, path: str, vocab_size: int, embedding_dim: int,
                       hidden_dim: int, pad_idx: int, sos_idx: int, eos_idx: int,
                       **kwargs) -> Tuple['BaseNameGenerationModel', Dict]:
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint file
            vocab_size, embedding_dim, hidden_dim: Model dimensions
            pad_idx, sos_idx, eos_idx: Special token indices
            **kwargs: Additional model-specific parameters
            
        Returns:
            Tuple of (model, checkpoint_info)
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        # Create model instance
        model = cls(vocab_size, embedding_dim, hidden_dim,
                   pad_idx, sos_idx, eos_idx, **kwargs)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint


class CharacterDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for character-level name data.
    """
    
    def __init__(self, names: List[List[int]], max_length: Optional[int] = None):
        """
        Initialize dataset.
        
        Args:
            names: List of encoded name sequences
            max_length: Maximum sequence length (for padding)
        """
        self.names = names
        
        if max_length is None:
            self.max_length = max(len(name) for name in names)
        else:
            self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.names)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get item at index.
        
        Returns:
            Tuple of (input_seq, target_seq, length)
        """
        name = self.names[idx]
        length = len(name)
        
        # Input: all chars except last (EOS)
        # Target: all chars except first (SOS)
        input_seq = name[:-1]
        target_seq = name[1:]
        
        # Pad if necessary
        if len(input_seq) < self.max_length - 1:
            input_seq = input_seq + [0] * (self.max_length - 1 - len(input_seq))
            target_seq = target_seq + [0] * (self.max_length - 1 - len(target_seq))
        
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long),
            length - 1  # Subtract 1 because we removed one token
        )


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    
    Args:
        batch: List of tuples from CharacterDataset
        
    Returns:
        Tuple of (input_batch, target_batch, lengths)
    """
    inputs, targets, lengths = zip(*batch)
    
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return inputs, targets, lengths
