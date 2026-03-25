"""
CBOW (Continuous Bag of Words) Implementation
=============================================

This module implements CBOW Word2Vec from scratch using PyTorch.

CBOW predicts a target word from its context words. Given surrounding words,
it learns to predict the center word.

Architecture:
    Context Words → Embedding → Average → Output Layer → Softmax

Author: Dhruva Kumar Kaushal (B22AI017)
Date: March 2026

Reference: Mikolov et al., "Efficient Estimation of Word Representations in Vector Space", 2013
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CBOW(nn.Module):
    """
    Continuous Bag of Words (CBOW) model implementation.
    
    The model consists of:
    1. Input embedding matrix: Maps word indices to dense vectors
    2. Context combination: Averages context word embeddings
    3. Output layer: Projects to vocabulary size
    4. Loss: Negative sampling loss (efficient alternative to softmax)
    
    Mathematical formulation:
        h = (1/C) * Σ(v_c) for context words c
        y = W_out @ h
        p = softmax(y) or negative_sampling_loss(y, target)
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initialize CBOW model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
        """
        super(CBOW, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Input embedding layer
        # Maps word index to dense embedding vector
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Output layer
        # Projects from embedding dimension to vocabulary size
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
        # Initialize weights with small random values
        # This is important for breaking symmetry and effective learning
        self._init_weights()
        
        logger.info(f"CBOW model initialized: vocab_size={vocab_size}, embedding_dim={embedding_dim}")
    
    def _init_weights(self):
        """
        Initialize weights with uniform distribution.
        
        Following the original Word2Vec implementation:
        - Embeddings initialized uniformly in [-0.5/dim, 0.5/dim]
        - Linear layer initialized similarly
        """
        init_range = 0.5 / self.embedding_dim
        self.embeddings.weight.data.uniform_(-init_range, init_range)
        self.linear.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.zero_()
    
    def forward(self, context_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CBOW model.
        
        Args:
            context_indices: Tensor of shape (batch_size, context_size)
                           Contains indices of context words
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, vocab_size)
        
        Process:
            1. Look up embeddings for all context words
            2. Average the context word embeddings
            3. Project to vocabulary size through linear layer
        """
        # Get embeddings for context words
        # Shape: (batch_size, context_size, embedding_dim)
        context_embeds = self.embeddings(context_indices)
        
        # Average context embeddings
        # Shape: (batch_size, embedding_dim)
        # This is the key operation in CBOW: combining context into single representation
        avg_embed = torch.mean(context_embeds, dim=1)
        
        # Project to vocabulary size
        # Shape: (batch_size, vocab_size)
        output = self.linear(avg_embed)
        
        return output
    
    def get_word_embedding(self, word_idx: int) -> np.ndarray:
        """
        Get embedding vector for a specific word.
        
        Args:
            word_idx: Index of the word
            
        Returns:
            np.ndarray: Embedding vector of shape (embedding_dim,)
        """
        with torch.no_grad():
            embedding = self.embeddings.weight[word_idx].cpu().numpy()
        return embedding
    
    def get_all_embeddings(self) -> np.ndarray:
        """
        Get all word embeddings as numpy array.
        
        Returns:
            np.ndarray: Embedding matrix of shape (vocab_size, embedding_dim)
        """
        with torch.no_grad():
            embeddings = self.embeddings.weight.cpu().numpy()
        return embeddings
    
    def save_embeddings(self, filepath: str, vocab=None):
        """
        Save embeddings to file.
        
        Args:
            filepath: Path to save embeddings
            vocab: Vocabulary object (optional, for saving with words)
        """
        embeddings = self.get_all_embeddings()
        
        if vocab:
            # Save in word2vec format: word followed by embedding values
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"{self.vocab_size} {self.embedding_dim}\n")
                for idx in range(self.vocab_size):
                    word = vocab.get_index_word(idx)
                    embed_str = ' '.join(f"{val:.6f}" for val in embeddings[idx])
                    f.write(f"{word} {embed_str}\n")
        else:
            # Save as numpy array
            np.save(filepath, embeddings)
        
        logger.info(f"Embeddings saved to {filepath}")
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count trainable parameters in the model.
        
        Returns:
            dict: Parameter counts by component and total
        """
        embedding_params = self.embeddings.weight.numel()
        linear_weight_params = self.linear.weight.numel()
        linear_bias_params = self.linear.bias.numel()
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        params_dict = {
            'embedding_layer': embedding_params,
            'output_weight': linear_weight_params,
            'output_bias': linear_bias_params,
            'total': total_params
        }
        
        return params_dict


class NegativeSamplingLoss(nn.Module):
    """
    Negative Sampling Loss for efficient Word2Vec training.
    
    Instead of computing softmax over entire vocabulary (expensive),
    we use negative sampling which only considers:
    - The target word (positive sample)
    - K randomly sampled negative words
    
    Loss function:
        L = -log(σ(s_pos)) - Σ log(σ(-s_neg_i))
    
    where:
        σ is sigmoid function
        s_pos is score for positive (target) word
        s_neg_i is score for negative sample i
    
    Reference: Mikolov et al., "Distributed Representations of Words and Phrases", 2013
    """
    
    def __init__(self):
        """Initialize negative sampling loss."""
        super(NegativeSamplingLoss, self).__init__()
    
    def forward(self, input_vectors: torch.Tensor, 
                output_vectors: torch.Tensor,
                negative_vectors: torch.Tensor) -> torch.Tensor:
        """
        Compute negative sampling loss.
        
        Args:
            input_vectors: Context embeddings (batch_size, embedding_dim)
            output_vectors: Target word embeddings (batch_size, embedding_dim)
            negative_vectors: Negative sample embeddings (batch_size, n_negative, embedding_dim)
        
        Returns:
            torch.Tensor: Scalar loss value
        
        Mathematics:
            For each training example:
            1. Positive score: dot(context, target)
            2. Negative scores: dot(context, negative_i) for each negative sample
            3. Loss: -log(sigmoid(pos_score)) - sum(log(sigmoid(-neg_scores)))
        """
        batch_size = input_vectors.size(0)
        
        # Compute positive score
        # Shape: (batch_size,)
        # This is the dot product between context and target embeddings
        positive_score = torch.sum(input_vectors * output_vectors, dim=1)
        
        # Apply sigmoid and log to positive scores
        # We want to maximize this (minimize negative log)
        positive_loss = -F.logsigmoid(positive_score)
        
        # Compute negative scores
        # Shape: (batch_size, n_negative)
        # For each negative sample, compute dot product with context
        negative_score = torch.bmm(negative_vectors, input_vectors.unsqueeze(2)).squeeze(2)
        
        # Apply sigmoid and log to negative scores (with negative sign)
        # We want to minimize these scores
        negative_loss = -torch.sum(F.logsigmoid(-negative_score), dim=1)
        
        # Total loss is sum of positive and negative losses
        loss = torch.mean(positive_loss + negative_loss)
        
        return loss


def create_training_pairs(sentences: List[List[int]], 
                         window_size: int) -> List[Tuple[List[int], int]]:
    """
    Create CBOW training pairs from sentences.
    
    For each word in each sentence, we create a training example where:
    - Input: Context words (words around target within window)
    - Output: Target word (center word)
    
    Args:
        sentences: List of sentences (each sentence is list of word indices)
        window_size: Context window size (number of words on each side)
    
    Returns:
        list: List of (context_indices, target_index) pairs
    
    Example:
        Sentence: "the quick brown fox jumps"
        Window size: 2
        For "brown" (index 2):
            Context: ["the", "quick", "fox", "jumps"]
            Target: "brown"
    """
    training_pairs = []
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # For each position in sentence
        for center_pos in range(sentence_length):
            target_word = sentence[center_pos]
            
            # Collect context words within window
            context = []
            
            # Look at words before center
            for offset in range(-window_size, 0):
                pos = center_pos + offset
                if 0 <= pos < sentence_length:
                    context.append(sentence[pos])
            
            # Look at words after center
            for offset in range(1, window_size + 1):
                pos = center_pos + offset
                if 0 <= pos < sentence_length:
                    context.append(sentence[pos])
            
            # Only add if we have context words
            if context:
                training_pairs.append((context, target_word))
    
    return training_pairs


def pad_context(contexts: List[List[int]], max_length: int, pad_value: int = 0) -> torch.Tensor:
    """
    Pad context sequences to same length for batching.
    
    Args:
        contexts: List of context word lists (variable length)
        max_length: Maximum length to pad to
        pad_value: Value to use for padding
    
    Returns:
        torch.Tensor: Padded tensor of shape (batch_size, max_length)
    """
    batch_size = len(contexts)
    padded = torch.full((batch_size, max_length), pad_value, dtype=torch.long)
    
    for i, context in enumerate(contexts):
        length = min(len(context), max_length)
        padded[i, :length] = torch.tensor(context[:length])
    
    return padded


if __name__ == "__main__":
    # Test CBOW model
    print("\n" + "="*60)
    print("TESTING CBOW MODEL")
    print("="*60)
    
    # Create a small test model
    vocab_size = 1000
    embedding_dim = 100
    
    model = CBOW(vocab_size, embedding_dim)
    
    # Count parameters
    params = model.count_parameters()
    print("\nModel Parameters:")
    for component, count in params.items():
        print(f"  {component}: {count:,}")
    
    # Test forward pass
    batch_size = 32
    context_size = 4
    context_indices = torch.randint(0, vocab_size, (batch_size, context_size))
    
    output = model(context_indices)
    print(f"\nInput shape: {context_indices.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test embedding extraction
    word_idx = 42
    embedding = model.get_word_embedding(word_idx)
    print(f"\nEmbedding for word {word_idx}: shape {embedding.shape}")
    
    print("="*60)
