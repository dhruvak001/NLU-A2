"""
RNN with Attention Mechanism for Name Generation
Dhruva Kumar Kaushal (B22AI017)
NLU Assignment - Problem 2

RNN with attention mechanism for character-level name generation.
Attention allows the model to focus on relevant parts of the input.

Original implementation from scratch following Bahdanau et al. (2015).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.base_model import BaseNameGenerationModel


class Attention(nn.Module):
    """
    Attention mechanism for RNN.
    
    Computes attention weights over encoder outputs to create
    context-aware representations.
    """
    
    def __init__(self, hidden_dim: int, attention_dim: int):
        """
        Initialize attention mechanism.
        
        Args:
            hidden_dim: Dimension of hidden states
            attention_dim: Dimension of attention layer
        """
        super().__init__()
        
        # Attention projection layers
        self.attn = nn.Linear(hidden_dim * 2, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)
    
    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights.
        
        Args:
            hidden: Current hidden state (batch_size, hidden_dim)
            encoder_outputs: All encoder outputs (batch_size, seq_len, hidden_dim)
            
        Returns:
            Attention weights (batch_size, seq_len)
        """
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # Repeat hidden state for all time steps
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_dim)
        
        # Concatenate hidden and encoder outputs
        combined = torch.cat((hidden, encoder_outputs), dim=2)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Compute energy
        energy = torch.tanh(self.attn(combined))  # (batch_size, seq_len, attention_dim)
        energy = self.v(energy).squeeze(2)  # (batch_size, seq_len)
        
        # Compute attention weights
        attention_weights = F.softmax(energy, dim=1)
        
        return attention_weights


class RNNWithAttention(BaseNameGenerationModel):
    """
    RNN with attention mechanism for character-level name generation.
    
    Features:
    - Character embeddings
    - Multi-layer GRU (faster than LSTM)
    - Attention mechanism over encoder outputs
    - Context-aware output projection
    - Dropout regularization
    
    Most sophisticated architecture among the three models.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int, attention_dim: int, dropout: float,
                 pad_idx: int, sos_idx: int, eos_idx: int):
        """
        Initialize RNN with Attention model.
        
        Args:
            vocab_size: Size of character vocabulary
            embedding_dim: Dimension of character embeddings
            hidden_dim: Dimension of hidden states
            num_layers: Number of GRU layers
            attention_dim: Dimension of attention layer
            dropout: Dropout probability
            pad_idx: Padding token index
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index
        """
        super().__init__(vocab_size, embedding_dim, hidden_dim, pad_idx, sos_idx, eos_idx)
        
        self.num_layers = num_layers
        self.attention_dim = attention_dim
        
        # GRU layer (faster than LSTM, similar performance)
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = Attention(hidden_dim, attention_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output projection (takes concatenated RNN output and context)
        self.output_projection = nn.Linear(hidden_dim * 2, vocab_size)
        
        self.logger.info(f"RNN with Attention: {num_layers} layers, "
                        f"{hidden_dim} hidden units, {attention_dim} attention dim")
    
    def forward(self, input_seq: torch.Tensor,
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model with attention.
        
        Args:
            input_seq: Input sequence tensor (batch_size, seq_len)
            hidden: Optional initial hidden state (num_layers, batch_size, hidden_dim)
            
        Returns:
            Tuple of (output_logits, final_hidden)
            - output_logits: (batch_size, seq_len, vocab_size)
            - final_hidden: (num_layers, batch_size, hidden_dim)
        """
        batch_size = input_seq.size(0)
        seq_len = input_seq.size(1)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size).to(input_seq.device)
        
        # Embed input characters
        embedded = self.embedding(input_seq)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Pass through GRU
        rnn_output, hidden = self.rnn(embedded, hidden)  # (batch_size, seq_len, hidden_dim)
        
        # Compute attention for each time step
        attention_outputs = []
        
        for t in range(seq_len):
            # Get hidden state at time t (from last layer)
            current_hidden = hidden[-1]  # (batch_size, hidden_dim)
            
            # Compute attention weights
            attn_weights = self.attention(current_hidden, rnn_output)  # (batch_size, seq_len)
            
            # Compute context vector as weighted sum of encoder outputs
            context = torch.bmm(attn_weights.unsqueeze(1), rnn_output)  # (batch_size, 1, hidden_dim)
            context = context.squeeze(1)  # (batch_size, hidden_dim)
            
            # Concatenate RNN output and context
            output_t = rnn_output[:, t, :]  # (batch_size, hidden_dim)
            combined = torch.cat((output_t, context), dim=1)  # (batch_size, hidden_dim * 2)
            
            attention_outputs.append(combined)
        
        # Stack outputs
        attention_output = torch.stack(attention_outputs, dim=1)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Apply dropout
        attention_output = self.dropout(attention_output)
        
        # Project to vocabulary size
        logits = self.output_projection(attention_output)  # (batch_size, seq_len, vocab_size)
        
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
