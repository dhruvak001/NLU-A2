"""
Vocabulary Builder for Word2Vec
================================

This module builds and manages the vocabulary for Word2Vec training.
It handles word-to-index mappings, frequency counting, and vocabulary pruning.

Author: Dhruva Kumar Kaushal (B22AI017)
Date: March 2026
"""

import numpy as np
from collections import Counter
from typing import List, Dict, Set
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Vocabulary:
    """
    Vocabulary class for Word2Vec.
    
    Manages:
    - Word to index mapping
    - Index to word mapping
    - Word frequencies
    - Negative sampling distribution
    """
    
    def __init__(self, min_count: int = 5):
        """
        Initialize vocabulary builder.
        
        Args:
            min_count: Minimum frequency for a word to be included in vocabulary
        """
        self.min_count = min_count
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        self.vocab_size = 0
        self.total_words = 0
        
        # For negative sampling - computed lazily
        self._negative_sampling_probs = None
    
    def build_vocabulary(self, sentences: List[List[str]]):
        """
        Build vocabulary from tokenized sentences.
        
        This function:
        1. Counts word frequencies
        2. Filters words below min_count threshold
        3. Creates word↔index bidirectional mappings
        4. Prepares negative sampling distribution
        
        Args:
            sentences: List of tokenized sentences (each sentence is list of tokens)
        """
        logger.info("Building vocabulary...")
        
        # Count all words
        for sentence in sentences:
            self.word_counts.update(sentence)
            self.total_words += len(sentence)
        
        logger.info(f"Total words before filtering: {len(self.word_counts)}")
        
        # Filter by minimum count
        filtered_words = {word: count for word, count in self.word_counts.items() 
                         if count >= self.min_count}
        
        logger.info(f"Words after filtering (min_count={self.min_count}): {len(filtered_words)}")
        
        # Create mappings (sorted by frequency for consistency)
        # Index 0 is reserved for unknown/padding
        sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
        
        self.word2idx = {'<UNK>': 0}
        self.idx2word = {0: '<UNK>'}
        
        for idx, (word, count) in enumerate(sorted_words, start=1):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        # Update word_counts to only include vocabulary words
        self.word_counts = Counter(filtered_words)
        self.vocab_size = len(self.word2idx)
        
        logger.info(f"Vocabulary built with {self.vocab_size} words")
        logger.info(f"Most common words: {self.word_counts.most_common(10)}")
    
    def get_word_index(self, word: str) -> int:
        """
        Get index for a word.
        
        Args:
            word: Word to look up
            
        Returns:
            int: Index of the word (0 for unknown words)
        """
        return self.word2idx.get(word, 0)
    
    def get_index_word(self, idx: int) -> str:
        """
        Get word for an index.
        
        Args:
            idx: Index to look up
            
        Returns:
            str: Word at that index ('<UNK>' if index not found)
        """
        return self.idx2word.get(idx, '<UNK>')
    
    def get_word_count(self, word: str) -> int:
        """
        Get frequency count for a word.
        
        Args:
            word: Word to look up
            
        Returns:
            int: Frequency count (0 if not in vocabulary)
        """
        return self.word_counts.get(word, 0)
    
    def encode_sentence(self, sentence: List[str]) -> List[int]:
        """
        Convert sentence (list of words) to list of indices.
        
        Args:
            sentence: List of word tokens
            
        Returns:
            list: List of word indices
        """
        return [self.get_word_index(word) for word in sentence]
    
    def decode_indices(self, indices: List[int]) -> List[str]:
        """
        Convert list of indices back to words.
        
        Args:
            indices: List of word indices
            
        Returns:
            list: List of words
        """
        return [self.get_index_word(idx) for idx in indices]
    
    def get_negative_sampling_probs(self) -> np.ndarray:
        """
        Compute negative sampling probability distribution.
        
        Uses the formula: P(w) = count(w)^(3/4) / Z
        This gives more weight to rare words during negative sampling.
        
        Returns:
            np.ndarray: Probability distribution over vocabulary
        """
        if self._negative_sampling_probs is not None:
            return self._negative_sampling_probs
        
        logger.info("Computing negative sampling distribution...")
        
        # Initialize probability array
        probs = np.zeros(self.vocab_size)
        
        # Compute probabilities using 3/4 power
        for word, count in self.word_counts.items():
            idx = self.word2idx[word]
            probs[idx] = count ** 0.75
        
        # Normalize to get probability distribution
        probs = probs / np.sum(probs)
        
        self._negative_sampling_probs = probs
        logger.info("Negative sampling distribution computed")
        
        return self._negative_sampling_probs
    
    def sample_negative(self, n_samples: int, exclude_indices: Set[int] = None) -> List[int]:
        """
        Sample negative examples for negative sampling.
        
        Args:
            n_samples: Number of negative samples to draw
            exclude_indices: Set of indices to exclude from sampling
            
        Returns:
            list: List of sampled negative indices
        """
        probs = self.get_negative_sampling_probs()
        
        if exclude_indices:
            # Create a copy and zero out excluded indices
            probs = probs.copy()
            for idx in exclude_indices:
                if 0 <= idx < len(probs):
                    probs[idx] = 0
            # Renormalize
            probs = probs / np.sum(probs)
        
        # Sample without replacement
        samples = np.random.choice(self.vocab_size, size=n_samples, 
                                  replace=False, p=probs)
        
        return samples.tolist()
    
    def save(self, filepath: str):
        """
        Save vocabulary to file.
        
        Args:
            filepath: Path to save vocabulary
        """
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_counts': dict(self.word_counts),
            'vocab_size': self.vocab_size,
            'total_words': self.total_words,
            'min_count': self.min_count
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        
        logger.info(f"Vocabulary saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'Vocabulary':
        """
        Load vocabulary from file.
        
        Args:
            filepath: Path to vocabulary file
            
        Returns:
            Vocabulary: Loaded vocabulary object
        """
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        
        vocab = cls(min_count=vocab_data['min_count'])
        vocab.word2idx = vocab_data['word2idx']
        vocab.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        vocab.word_counts = Counter(vocab_data['word_counts'])
        vocab.vocab_size = vocab_data['vocab_size']
        vocab.total_words = vocab_data['total_words']
        
        logger.info(f"Vocabulary loaded from {filepath}")
        logger.info(f"Vocabulary size: {vocab.vocab_size}")
        
        return vocab
    
    def get_statistics(self) -> Dict:
        """
        Get vocabulary statistics.
        
        Returns:
            dict: Dictionary containing vocabulary statistics
        """
        return {
            'vocab_size': self.vocab_size,
            'total_words': self.total_words,
            'min_count': self.min_count,
            'most_common': self.word_counts.most_common(20),
            'least_common': self.word_counts.most_common()[-20:] if len(self.word_counts) >= 20 else []
        }


def load_corpus(corpus_file: str) -> List[List[str]]:
    """
    Load corpus from file.
    
    Args:
        corpus_file: Path to corpus file (one sentence per line)
        
    Returns:
        list: List of tokenized sentences
    """
    sentences = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                sentences.append(tokens)
    
    return sentences


if __name__ == "__main__":
    # Test vocabulary building
    corpus_file = 'data/processed/corpus.txt'
    sentences = load_corpus(corpus_file)
    
    vocab = Vocabulary(min_count=5)
    vocab.build_vocabulary(sentences)
    
    print("\n" + "="*60)
    print("VOCABULARY STATISTICS")
    print("="*60)
    stats = vocab.get_statistics()
    print(f"Vocabulary Size: {stats['vocab_size']}")
    print(f"Total Words: {stats['total_words']}")
    print(f"\nTop 10 Most Common Words:")
    for word, count in stats['most_common'][:10]:
        print(f"  {word}: {count}")
    print("="*60)
    
    # Save vocabulary
    vocab.save('data/processed/vocabulary.pkl')
