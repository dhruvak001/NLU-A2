"""
Bidirectional LSTM Model for Name Generation
Dhruva Kumar Kaushal (B22AI017)
NLU Assignment - Problem 2

BLSTM implementation for character-level name generation.
Captures context from both forward and backward directions.

Original implementation from scratch.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.base_model import BaseNameGenerationModel


class BLSTM(BaseNameGenerationModel):
    """
    Bidirectional LSTM for character-level name generation.
    
    Features:
    - Character embeddings
    - Bidirectional LSTM layers
    - Dropout regularization
    - Linear output projection
    
    Better at capturing long-range dependencies than vanilla RNN.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int, dropout: float, pad_idx: int, sos_idx: int, eos_idx: int):
        """
        Initialize Bidirectional LSTM model.
        
        Args:
            vocab_size: Size of character vocabulary
            embedding_dim: Dimension of character embeddings
            hidden_dim: Dimension of hidden states (will be split for bidirectional)
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            pad_idx: Padding token index
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index
        """
        super().__init__(vocab_size, embedding_dim, hidden_dim, pad_idx, sos_idx, eos_idx)
        
        self.num_layers = num_layers
        
        # Bidirectional LSTM - each direction has hidden_dim // 2
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,  # Split for bidirectional
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        self.logger.info(f"Bidirectional LSTM: {num_layers} layers, "
                        f"{hidden_dim} hidden units (bidirectional)")
    
    def forward(self, input_seq: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> \
                Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            input_seq: Input sequence tensor (batch_size, seq_len)
            hidden: Optional initial hidden state tuple (h_0, c_0)
                   Each: (num_layers * 2, batch_size, hidden_dim // 2)
            
        Returns:
            Tuple of (output_logits, final_hidden)
            - output_logits: (batch_size, seq_len, vocab_size)
            - final_hidden: Tuple of (h_n, c_n)
        """
        batch_size = input_seq.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size)
            hidden = (hidden[0].to(input_seq.device), hidden[1].to(input_seq.device))
        
        # Embed input characters
        embedded = self.embedding(input_seq)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Pass through LSTM
        # Output shape: (batch_size, seq_len, hidden_dim)
        # Hidden shape: (num_layers * 2, batch_size, hidden_dim // 2)
        lstm_output, hidden = self.lstm(embedded, hidden)
        
        # Apply dropout
        lstm_output = self.dropout(lstm_output)
        
        # Project to vocabulary size
        logits = self.output_projection(lstm_output)  # (batch_size, seq_len, vocab_size)
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden and cell states with zeros.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Tuple of (h_0, c_0) tensors
            Each: (num_layers * 2, batch_size, hidden_dim // 2)
        """
        # LSTM has both hidden and cell states
        # Multiply num_layers by 2 for bidirectional
        h_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2)
        c_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2)
        
        return (h_0, c_0)
