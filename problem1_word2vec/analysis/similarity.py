"""
Semantic Analysis Tools - Similarity and Nearest Neighbors
=========================================================

This module implements cosine similarity computation and nearest neighbor
search for Word2Vec embeddings.

Author: Dhruva Kumar Kaushal (B22AI017)
Date: March 2026
"""

import numpy as np
from typing import List, Dict, Tuple
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Cosine similarity measures the cosine of the angle between two vectors.
    It ranges from -1 (opposite) to 1 (identical).
    
    Formula:
        similarity = (A · B) / (||A|| × ||B||)
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        float: Cosine similarity score between -1 and 1
    """
    # Compute dot product
    dot_product = np.dot(vec1, vec2)
    
    # Compute norms (magnitudes)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Return cosine similarity
    return dot_product / (norm1 * norm2)


def batch_cosine_similarity(query_vec: np.ndarray, 
                           matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and all vectors in a matrix.
    
    This is more efficient than computing similarities one by one.
    
    Args:
        query_vec: Query vector of shape (embedding_dim,)
        matrix: Matrix of vectors of shape (n_vectors, embedding_dim)
    
    Returns:
        np.ndarray: Array of similarities of shape (n_vectors,)
    """
    # Normalize query vector
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    
    # Normalize all vectors in matrix
    matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix_normalized = matrix / (matrix_norms + 1e-10)
    
    # Compute dot products (cosine similarities)
    similarities = np.dot(matrix_normalized, query_norm)
    
    return similarities


class SemanticAnalyzer:
    """
    Semantic analysis tool for Word2Vec embeddings.
    
    Provides methods for:
    - Finding nearest neighbors
    - Computing word similarities
    - Analyzing semantic relationships
    """
    
    def __init__(self, embeddings: np.ndarray, vocab):
        """
        Initialize semantic analyzer.
        
        Args:
            embeddings: Embedding matrix of shape (vocab_size, embedding_dim)
            vocab: Vocabulary object with word-index mappings
        """
        self.embeddings = embeddings
        self.vocab = vocab
        self.vocab_size, self.embedding_dim = embeddings.shape
        
        logger.info(f"SemanticAnalyzer initialized with {self.vocab_size} words")
    
    def get_word_vector(self, word: str) -> np.ndarray:
        """
        Get embedding vector for a word.
        
        Args:
            word: Word to look up
        
        Returns:
            np.ndarray: Embedding vector, or None if word not in vocabulary
        """
        idx = self.vocab.get_word_index(word)
        if idx == 0 and word != '<UNK>':  # Not in vocabulary
            return None
        return self.embeddings[idx]
    
    def word_similarity(self, word1: str, word2: str) -> float:
        """
        Compute similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
        
        Returns:
            float: Similarity score (0-1), or None if either word not found
        """
        vec1 = self.get_word_vector(word1)
        vec2 = self.get_word_vector(word2)
        
        if vec1 is None or vec2 is None:
            logger.warning(f"One or both words not found: {word1}, {word2}")
            return None
        
        return cosine_similarity(vec1, vec2)
    
    def find_nearest_neighbors(self, word: str, k: int = 5, 
                              exclude_self: bool = True) -> List[Tuple[str, float]]:
        """
        Find k nearest neighbors for a word.
        
        Args:
            word: Query word
            k: Number of neighbors to return
            exclude_self: Whether to exclude the word itself from results
        
        Returns:
            list: List of (word, similarity) tuples, sorted by similarity (descending)
        """
        # Get word vector
        word_vec = self.get_word_vector(word)
        if word_vec is None:
            logger.error(f"Word not in vocabulary: {word}")
            return []
        
        # Compute similarities with all words
        similarities = batch_cosine_similarity(word_vec, self.embeddings)
        
        # Get word index for exclusion
        word_idx = self.vocab.get_word_index(word)
        
        # Sort by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Collect top k neighbors
        neighbors = []
        for idx in sorted_indices:
            # Skip if it's the word itself and we want to exclude it
            if exclude_self and idx == word_idx:
                continue
            
            # Skip unknown token
            if idx == 0:
                continue
            
            neighbor_word = self.vocab.get_index_word(idx)
            similarity = similarities[idx]
            
            neighbors.append((neighbor_word, float(similarity)))
            
            if len(neighbors) >= k:
                break
        
        return neighbors
    
    def analyze_word(self, word: str, k: int = 5) -> Dict:
        """
        Comprehensive analysis of a word.
        
        Args:
            word: Word to analyze
            k: Number of neighbors to find
        
        Returns:
            dict: Analysis results including neighbors and statistics
        """
        neighbors = self.find_nearest_neighbors(word, k)
        
        word_vec = self.get_word_vector(word)
        if word_vec is None:
            return {'error': 'Word not in vocabulary', 'word': word}
        
        analysis = {
            'word': word,
            'in_vocabulary': True,
            'word_index': self.vocab.get_word_index(word),
            'frequency': self.vocab.get_word_count(word),
            'nearest_neighbors': neighbors,
            'embedding_norm': float(np.linalg.norm(word_vec)),
            'embedding_mean': float(np.mean(word_vec)),
            'embedding_std': float(np.std(word_vec))
        }
        
        return analysis
    
    def analyze_multiple_words(self, words: List[str], k: int = 5) -> Dict:
        """
        Analyze multiple words.
        
        Args:
            words: List of words to analyze
            k: Number of neighbors per word
        
        Returns:
            dict: Analysis results for all words
        """
        results = {}
        
        for word in words:
            logger.info(f"Analyzing word: {word}")
            results[word] = self.analyze_word(word, k)
        
        return results
    
    def save_analysis(self, results: Dict, filepath: str):
        """
        Save analysis results to JSON file.
        
        Args:
            results: Analysis results dictionary
            filepath: Path to save file
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis results saved to {filepath}")
    
    def create_similarity_matrix(self, words: List[str]) -> np.ndarray:
        """
        Create pairwise similarity matrix for a list of words.
        
        Args:
            words: List of words
        
        Returns:
            np.ndarray: Similarity matrix of shape (n_words, n_words)
        """
        n_words = len(words)
        similarity_matrix = np.zeros((n_words, n_words))
        
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                elif i < j:
                    sim = self.word_similarity(word1, word2)
                    similarity_matrix[i, j] = sim if sim is not None else 0.0
                    similarity_matrix[j, i] = similarity_matrix[i, j]
        
        return similarity_matrix


def analyze_required_words(model, vocab, output_dir: str):
    """
    Analyze the required words for the assignment.
    
    Required words: research, student, phd, exam
    
    Args:
        model: Trained Word2Vec model
        vocab: Vocabulary object
        output_dir: Directory to save results
    """
    import os
    
    # Get embeddings
    embeddings = model.get_all_embeddings()
    
    # Initialize analyzer
    analyzer = SemanticAnalyzer(embeddings, vocab)
    
    # Required words
    required_words = ["research", "student", "phd", "exam"]
    
    # Analyze
    results = analyzer.analyze_multiple_words(required_words, k=5)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'nearest_neighbors.json')
    analyzer.save_analysis(results, output_file)
    
    # Print results
    print("\n" + "="*60)
    print("NEAREST NEIGHBORS ANALYSIS")
    print("="*60)
    
    for word, analysis in results.items():
        if 'error' in analysis:
            print(f"\n❌ {word}: {analysis['error']}")
            continue
        
        print(f"\n📊 Word: {word}")
        print(f"   Frequency: {analysis['frequency']}")
        print(f"   Top 5 Nearest Neighbors:")
        
        for i, (neighbor, similarity) in enumerate(analysis['nearest_neighbors'], 1):
            print(f"      {i}. {neighbor} (similarity: {similarity:.4f})")
    
    print("="*60)
    
    return results


if __name__ == "__main__":
    print("Semantic Analysis Module")
    print("Run through main.py for full analysis")
