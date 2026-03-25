"""
Skip-gram Implementation with Negative Sampling
==============================================

This module implements Skip-gram Word2Vec from scratch using PyTorch.

Skip-gram predicts context words from a target word. Given a center word,
it learns to predict the surrounding words.

Architecture:
    Target Word → Embedding → Output Layer → Context Words

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


class SkipGram(nn.Module):
    """
    Skip-gram model implementation.
    
    Skip-gram is the inverse of CBOW:
    - Input: Target (center) word
    - Output: Context words
    
    The model learns by trying to predict context words given the target word.
    This often works better than CBOW for smaller datasets and rare words.
    
    Mathematical formulation:
        For target word w_t and context word w_c:
        p(w_c | w_t) = exp(v_c · v_t) / Σ exp(v_i · v_t)
        
        With negative sampling, we optimize:
        log σ(v_c · v_t) + Σ log σ(-v_n · v_t) for negative samples n
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initialize Skip-gram model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
        """
        super(SkipGram, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Input embeddings (for target words)
        # These are the embeddings we'll use for similarity and analogies
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Output embeddings (for context words)
        # Separate embedding matrix for output helps with learning
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Skip-gram model initialized: vocab_size={vocab_size}, embedding_dim={embedding_dim}")
    
    def _init_weights(self):
        """
        Initialize weights with uniform distribution.
        
        Using the same initialization strategy as CBOW for consistency.
        """
        init_range = 0.5 / self.embedding_dim
        self.embeddings.weight.data.uniform_(-init_range, init_range)
        self.output_embeddings.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, target_indices: torch.Tensor, 
                context_indices: torch.Tensor,
                negative_indices: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of Skip-gram model.
        
        Args:
            target_indices: Tensor of shape (batch_size,) - target word indices
            context_indices: Tensor of shape (batch_size,) - context word indices
            negative_indices: Tensor of shape (batch_size, n_negative) - negative samples
        
        Returns:
            torch.Tensor: Loss value
        
        Note: Unlike CBOW, Skip-gram forward pass computes loss directly
        because we use negative sampling rather than softmax.
        """
        # Get target embeddings
        # Shape: (batch_size, embedding_dim)
        target_embeds = self.embeddings(target_indices)
        
        # Get context embeddings
        # Shape: (batch_size, embedding_dim)
        context_embeds = self.output_embeddings(context_indices)
        
        # Compute positive score (target · context)
        # Shape: (batch_size,)
        positive_score = torch.sum(target_embeds * context_embeds, dim=1)
        positive_loss = -F.logsigmoid(positive_score)
        
        # Compute negative scores if negative samples provided
        if negative_indices is not None:
            # Get negative embeddings
            # Shape: (batch_size, n_negative, embedding_dim)
            negative_embeds = self.output_embeddings(negative_indices)
            
            # Compute negative scores
            # Shape: (batch_size, n_negative)
            negative_score = torch.bmm(
                negative_embeds, 
                target_embeds.unsqueeze(2)
            ).squeeze(2)
            
            # Negative loss
            negative_loss = -torch.sum(F.logsigmoid(-negative_score), dim=1)
            
            # Total loss
            loss = torch.mean(positive_loss + negative_loss)
        else:
            loss = torch.mean(positive_loss)
        
        return loss
    
    def get_input_embedding(self, word_idx: int) -> np.ndarray:
        """
        Get input embedding vector for a specific word.
        
        This is the embedding we use for word similarity and analogies.
        
        Args:
            word_idx: Index of the word
            
        Returns:
            np.ndarray: Embedding vector of shape (embedding_dim,)
        """
        with torch.no_grad():
            embedding = self.embeddings.weight[word_idx].cpu().numpy()
        return embedding
    
    def get_output_embedding(self, word_idx: int) -> np.ndarray:
        """
        Get output embedding vector for a specific word.
        
        Args:
            word_idx: Index of the word
            
        Returns:
            np.ndarray: Embedding vector of shape (embedding_dim,)
        """
        with torch.no_grad():
            embedding = self.output_embeddings.weight[word_idx].cpu().numpy()
        return embedding
    
    def get_all_embeddings(self, use_input: bool = True) -> np.ndarray:
        """
        Get all word embeddings as numpy array.
        
        Args:
            use_input: If True, return input embeddings; else output embeddings
        
        Returns:
            np.ndarray: Embedding matrix of shape (vocab_size, embedding_dim)
        """
        with torch.no_grad():
            if use_input:
                embeddings = self.embeddings.weight.cpu().numpy()
            else:
                embeddings = self.output_embeddings.weight.cpu().numpy()
        return embeddings
    
    def get_averaged_embeddings(self) -> np.ndarray:
        """
        Get averaged input and output embeddings.
        
        Some implementations average input and output embeddings for final representation.
        
        Returns:
            np.ndarray: Averaged embedding matrix of shape (vocab_size, embedding_dim)
        """
        with torch.no_grad():
            input_embeds = self.embeddings.weight.cpu().numpy()
            output_embeds = self.output_embeddings.weight.cpu().numpy()
            averaged = (input_embeds + output_embeds) / 2
        return averaged
    
    def save_embeddings(self, filepath: str, vocab=None, use_input: bool = True):
        """
        Save embeddings to file.
        
        Args:
            filepath: Path to save embeddings
            vocab: Vocabulary object (optional)
            use_input: If True, save input embeddings; else output embeddings
        """
        embeddings = self.get_all_embeddings(use_input=use_input)
        
        if vocab:
            # Save in word2vec format
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
        input_embedding_params = self.embeddings.weight.numel()
        output_embedding_params = self.output_embeddings.weight.numel()
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        params_dict = {
            'input_embeddings': input_embedding_params,
            'output_embeddings': output_embedding_params,
            'total': total_params
        }
        
        return params_dict


def create_skipgram_pairs(sentences: List[List[int]], 
                          window_size: int) -> List[Tuple[int, int]]:
    """
    Create Skip-gram training pairs from sentences.
    
    For Skip-gram, each training example consists of:
    - Input: Target (center) word
    - Output: One context word
    
    Unlike CBOW, Skip-gram creates multiple training pairs for each word:
    one pair for each context word.
    
    Args:
        sentences: List of sentences (each sentence is list of word indices)
        window_size: Context window size
    
    Returns:
        list: List of (target_index, context_index) pairs
    
    Example:
        Sentence: "the quick brown fox jumps"
        Window size: 2
        For "brown" (index 2):
            Pairs: (brown, the), (brown, quick), (brown, fox), (brown, jumps)
    """
    training_pairs = []
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # For each position in sentence
        for center_pos in range(sentence_length):
            target_word = sentence[center_pos]
            
            # Look at context words within window
            # Create one pair for each context word
            for offset in range(-window_size, window_size + 1):
                if offset == 0:  # Skip the center word itself
                    continue
                
                context_pos = center_pos + offset
                
                # Check if position is valid
                if 0 <= context_pos < sentence_length:
                    context_word = sentence[context_pos]
                    training_pairs.append((target_word, context_word))
    
    return training_pairs


class SkipGramDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Skip-gram training.
    
    Handles:
    - Loading training pairs
    - Generating negative samples on-the-fly
    - Batching
    """
    
    def __init__(self, training_pairs: List[Tuple[int, int]], 
                 vocab, n_negative: int = 10):
        """
        Initialize dataset.
        
        Args:
            training_pairs: List of (target, context) pairs
            vocab: Vocabulary object for negative sampling
            n_negative: Number of negative samples per example
        """
        self.training_pairs = training_pairs
        self.vocab = vocab
        self.n_negative = n_negative
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.training_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training example.
        
        Args:
            idx: Index of example
        
        Returns:
            tuple: (target_idx, context_idx, negative_indices)
        """
        target_idx, context_idx = self.training_pairs[idx]
        
        # Sample negative examples
        # Exclude target and context from negative samples
        exclude = {target_idx, context_idx}
        negative_indices = self.vocab.sample_negative(self.n_negative, exclude)
        
        return (
            torch.tensor(target_idx, dtype=torch.long),
            torch.tensor(context_idx, dtype=torch.long),
            torch.tensor(negative_indices, dtype=torch.long)
        )


if __name__ == "__main__":
    # Test Skip-gram model
    print("\n" + "="*60)
    print("TESTING SKIP-GRAM MODEL")
    print("="*60)
    
    # Create a small test model
    vocab_size = 1000
    embedding_dim = 100
    
    model = SkipGram(vocab_size, embedding_dim)
    
    # Count parameters
    params = model.count_parameters()
    print("\nModel Parameters:")
    for component, count in params.items():
        print(f"  {component}: {count:,}")
    
    # Test forward pass
    batch_size = 32
    n_negative = 10
    
    target_indices = torch.randint(0, vocab_size, (batch_size,))
    context_indices = torch.randint(0, vocab_size, (batch_size,))
    negative_indices = torch.randint(0, vocab_size, (batch_size, n_negative))
    
    loss = model(target_indices, context_indices, negative_indices)
    print(f"\nTarget shape: {target_indices.shape}")
    print(f"Context shape: {context_indices.shape}")
    print(f"Negative shape: {negative_indices.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test embedding extraction
    word_idx = 42
    embedding = model.get_input_embedding(word_idx)
    print(f"\nEmbedding for word {word_idx}: shape {embedding.shape}")
    
    print("="*60)
