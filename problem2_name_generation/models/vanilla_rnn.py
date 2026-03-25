"""
Vanilla RNN Model for Name Generation
Dhruva Kumar Kaushal (B22AI017)
NLU Assignment - Problem 2

Simple RNN implementation for character-level name generation.
Serves as baseline for comparison with more advanced architectures.

Original implementation from scratch.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.base_model import BaseNameGenerationModel


class VanillaRNN(BaseNameGenerationModel):
    """
    Vanilla RNN for character-level name generation.
    
    Simple recurrent architecture with:
    - Character embeddings
    - Multi-layer RNN
    - Dropout regularization
    - Linear output projection
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int, dropout: float, pad_idx: int, sos_idx: int, eos_idx: int):
        """
        Initialize Vanilla RNN model.
        
        Args:
            vocab_size: Size of character vocabulary
            embedding_dim: Dimension of character embeddings
            hidden_dim: Dimension of hidden states
            num_layers: Number of RNN layers
            dropout: Dropout probability
            pad_idx: Padding token index
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index
        """
        super().__init__(vocab_size, embedding_dim, hidden_dim, pad_idx, sos_idx, eos_idx)
        
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        self.logger.info(f"Vanilla RNN: {num_layers} layers, {hidden_dim} hidden units")
    
    def forward(self, input_seq: torch.Tensor, 
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_seq: Input sequence tensor (batch_size, seq_len)
            hidden: Optional initial hidden state (num_layers, batch_size, hidden_dim)
            
        Returns:
            Tuple of (output_logits, final_hidden)
            - output_logits: (batch_size, seq_len, vocab_size)
            - final_hidden: (num_layers, batch_size, hidden_dim)
        """
        batch_size = input_seq.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size).to(input_seq.device)
        
        # Embed input characters
        embedded = self.embedding(input_seq)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Pass through RNN
        rnn_output, hidden = self.rnn(embedded, hidden)  # (batch_size, seq_len, hidden_dim)
        
        # Apply dropout
        rnn_output = self.dropout(rnn_output)
        
        # Project to vocabulary size
        logits = self.output_projection(rnn_output)  # (batch_size, seq_len, vocab_size)
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """
        Initialize hidden state with zeros.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial hidden state tensor (num_layers, batch_size, hidden_dim)
        """
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)
