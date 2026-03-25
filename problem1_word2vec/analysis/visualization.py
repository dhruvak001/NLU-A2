"""
Visualization Module for Word Embeddings
Dhruva Kumar Kaushal (B22AI017)
NLU Assignment - Problem 1

This module implements dimensionality reduction and visualization techniques
for word embeddings:
- PCA (Principal Component Analysis)
- t-SNE (t-Distributed Stochastic Neighbor Embedding)

Visualizations help understand the semantic space learned by Word2Vec models.

Original implementation following best practices for embedding visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Optional, Tuple, Dict
import logging
from pathlib import Path
import json


class EmbeddingVisualizer:
    """
    Visualizes word embeddings using dimensionality reduction techniques.
    
    Implements:
    - PCA for linear dimensionality reduction
    - t-SNE for non-linear dimensionality reduction
    - Interactive and static plots
    
    Attributes:
        embeddings (np.ndarray): Word embedding matrix
        word2idx (dict): Word to index mapping
        idx2word (dict): Index to word mapping
    """
    
    def __init__(self, embeddings: np.ndarray, word2idx: dict, idx2word: dict):
        """
        Initialize the visualizer.
        
        Args:
            embeddings: Word embedding matrix (vocab_size x embedding_dim)
            word2idx: Word to index mapping
            idx2word: Index to word mapping
        """
        self.embeddings = embeddings
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.logger = logging.getLogger(__name__)
        
        # Set visualization style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        self.logger.info(f"Initialized EmbeddingVisualizer with {len(word2idx)} words")
    
    def select_words_for_visualization(self, 
                                      method: str = 'frequency',
                                      n_words: int = 100,
                                      custom_words: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Select a subset of words for visualization.
        
        Args:
            method: Selection method ('frequency', 'random', 'custom')
            n_words: Number of words to select
            custom_words: List of specific words to visualize (for 'custom' method)
            
        Returns:
            Tuple of (selected_embeddings, selected_words)
        """
        if method == 'custom' and custom_words:
            # Use custom word list
            indices = []
            words = []
            for word in custom_words:
                if word in self.word2idx:
                    indices.append(self.word2idx[word])
                    words.append(word)
                else:
                    self.logger.warning(f"Word '{word}' not in vocabulary")
            
            selected_embeddings = self.embeddings[indices]
            
        elif method == 'random':
            # Random selection
            indices = np.random.choice(len(self.embeddings), size=min(n_words, len(self.embeddings)), 
                                      replace=False)
            selected_embeddings = self.embeddings[indices]
            words = [self.idx2word[idx] for idx in indices]
            
        else:  # Default: frequency-based
            # Select most frequent words (assuming they appear first in vocabulary)
            n_select = min(n_words, len(self.embeddings))
            selected_embeddings = self.embeddings[:n_select]
            words = [self.idx2word[i] for i in range(n_select)]
        
        self.logger.info(f"Selected {len(words)} words for visualization using '{method}' method")
        return selected_embeddings, words
    
    def visualize_pca(self, 
                      n_words: int = 100,
                      custom_words: Optional[List[str]] = None,
                      output_path: Optional[Path] = None,
                      show_plot: bool = True) -> Dict:
        """
        Visualize embeddings using PCA (2D projection).
        
        PCA finds the principal components that explain maximum variance.
        Good for getting a quick overview of the embedding space.
        
        Args:
            n_words: Number of words to visualize
            custom_words: Optional list of specific words to visualize
            output_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Dictionary containing PCA results and statistics
        """
        self.logger.info("Performing PCA visualization...")
        
        # Select words
        method = 'custom' if custom_words else 'frequency'
        embeddings_subset, words = self.select_words_for_visualization(
            method=method, n_words=n_words, custom_words=custom_words
        )
        
        # Apply PCA
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(embeddings_subset)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot points
        scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                           alpha=0.6, s=50, c=range(len(words)), 
                           cmap='viridis')
        
        # Add word labels
        for i, word in enumerate(words):
            ax.annotate(word, (coords_2d[i, 0], coords_2d[i, 1]),
                       fontsize=8, alpha=0.7,
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel(f'First Principal Component (Variance: {pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'Second Principal Component (Variance: {pca.explained_variance_ratio_[1]:.2%})')
        ax.set_title(f'PCA Visualization of Word Embeddings\n'
                    f'({len(words)} words, Total Variance Explained: '
                    f'{sum(pca.explained_variance_ratio_):.2%})')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved PCA plot to {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Return statistics
        results = {
            'method': 'PCA',
            'n_components': 2,
            'n_words': len(words),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'total_variance_explained': float(sum(pca.explained_variance_ratio_)),
            'coordinates': coords_2d.tolist(),
            'words': words
        }
        
        return results
    
    def visualize_tsne(self,
                       n_words: int = 100,
                       custom_words: Optional[List[str]] = None,
                       perplexity: int = 30,
                       n_iter: int = 1000,
                       output_path: Optional[Path] = None,
                       show_plot: bool = True) -> Dict:
        """
        Visualize embeddings using t-SNE (2D projection).
        
        t-SNE is better at preserving local structure and revealing clusters.
        More computationally expensive than PCA but often gives better insights.
        
        Args:
            n_words: Number of words to visualize
            custom_words: Optional list of specific words to visualize
            perplexity: t-SNE perplexity parameter (typical range: 5-50)
            n_iter: Number of iterations for optimization
            output_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Dictionary containing t-SNE results and statistics
        """
        self.logger.info(f"Performing t-SNE visualization (perplexity={perplexity})...")
        
        # Select words
        method = 'custom' if custom_words else 'frequency'
        embeddings_subset, words = self.select_words_for_visualization(
            method=method, n_words=n_words, custom_words=custom_words
        )
        
        # Adjust perplexity if needed
        max_perplexity = (len(words) - 1) // 3
        if perplexity > max_perplexity:
            perplexity = max_perplexity
            self.logger.warning(f"Adjusted perplexity to {perplexity} based on sample size")
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                   random_state=42, verbose=0)
        coords_2d = tsne.fit_transform(embeddings_subset)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot points
        scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                           alpha=0.6, s=50, c=range(len(words)), 
                           cmap='plasma')
        
        # Add word labels
        for i, word in enumerate(words):
            ax.annotate(word, (coords_2d[i, 0], coords_2d[i, 1]),
                       fontsize=8, alpha=0.7,
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title(f't-SNE Visualization of Word Embeddings\n'
                    f'({len(words)} words, perplexity={perplexity}, iterations={n_iter})')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved t-SNE plot to {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Return statistics
        results = {
            'method': 't-SNE',
            'n_components': 2,
            'n_words': len(words),
            'perplexity': perplexity,
            'n_iter': n_iter,
            'coordinates': coords_2d.tolist(),
            'words': words
        }
        
        return results
    
    def create_semantic_clusters_plot(self,
                                     word_groups: Dict[str, List[str]],
                                     output_path: Optional[Path] = None,
                                     show_plot: bool = True,
                                     method: str = 'tsne') -> Dict:
        """
        Create a visualization highlighting semantic clusters.
        
        Args:
            word_groups: Dictionary mapping group names to lists of words
                        Example: {'geography': ['india', 'delhi', ...],
                                 'education': ['student', 'professor', ...]}
            output_path: Path to save the plot
            show_plot: Whether to display the plot
            method: 'pca' or 'tsne'
            
        Returns:
            Dictionary containing visualization results
        """
        self.logger.info(f"Creating semantic clusters plot using {method.upper()}...")
        
        # Collect all words and their group labels
        all_words = []
        labels = []
        colors = []
        color_map = plt.cm.get_cmap('tab10')
        
        for i, (group_name, words) in enumerate(word_groups.items()):
            for word in words:
                if word in self.word2idx:
                    all_words.append(word)
                    labels.append(group_name)
                    colors.append(color_map(i))
        
        # Get embeddings
        indices = [self.word2idx[word] for word in all_words]
        embeddings_subset = self.embeddings[indices]
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, perplexity=min(30, len(all_words) // 3), 
                          n_iter=1000, random_state=42)
        else:
            reducer = PCA(n_components=2, random_state=42)
        
        coords_2d = reducer.fit_transform(embeddings_subset)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot each group with different color
        for i, group_name in enumerate(word_groups.keys()):
            group_mask = [label == group_name for label in labels]
            group_coords = coords_2d[group_mask]
            group_words = [w for w, m in zip(all_words, group_mask) if m]
            
            ax.scatter(group_coords[:, 0], group_coords[:, 1],
                      label=group_name, alpha=0.7, s=100, color=color_map(i))
            
            # Add labels
            for j, word in enumerate(group_words):
                ax.annotate(word, (group_coords[j, 0], group_coords[j, 1]),
                          fontsize=9, alpha=0.8,
                          xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel(f'{method.upper()} Dimension 1')
        ax.set_ylabel(f'{method.upper()} Dimension 2')
        ax.set_title(f'Semantic Clusters in Word Embedding Space ({method.upper()})\n'
                    f'{len(all_words)} words across {len(word_groups)} categories')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved semantic clusters plot to {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        results = {
            'method': method,
            'n_groups': len(word_groups),
            'n_words': len(all_words),
            'groups': {name: len([w for w in words if w in self.word2idx]) 
                      for name, words in word_groups.items()},
            'coordinates': coords_2d.tolist(),
            'words': all_words,
            'labels': labels
        }
        
        return results
    
    @staticmethod
    def get_default_semantic_groups() -> Dict[str, List[str]]:
        """
        Get default semantic word groups for cluster visualization.
        
        Returns:
            Dictionary mapping category names to word lists
        """
        return {
            'Education': [
                'student', 'professor', 'teacher', 'lecture', 'university',
                'college', 'research', 'study', 'exam', 'degree', 'bachelor',
                'master', 'phd', 'thesis', 'course'
            ],
            'Technology': [
                'computer', 'software', 'algorithm', 'programming', 'code',
                'data', 'technology', 'digital', 'internet', 'network',
                'artificial', 'intelligence', 'machine', 'learning', 'python'
            ],
            'Science': [
                'science', 'physics', 'chemistry', 'biology', 'mathematics',
                'experiment', 'theory', 'research', 'laboratory', 'discovery',
                'equation', 'formula', 'analysis', 'hypothesis'
            ],
            'Geography': [
                'india', 'delhi', 'mumbai', 'bangalore', 'jodhpur',
                'country', 'city', 'capital', 'state', 'nation',
                'asia', 'america', 'europe', 'world'
            ],
            'Time': [
                'day', 'week', 'month', 'year', 'time', 'today',
                'tomorrow', 'yesterday', 'morning', 'evening', 'night',
                'past', 'present', 'future', 'century'
            ]
        }
    
    def save_visualization_results(self, results: Dict, output_path: Path):
        """
        Save visualization results to JSON file.
        
        Args:
            results: Results dictionary from visualization methods
            output_path: Path to save JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved visualization results to {output_path}")


def main():
    """
    Example usage and testing of the EmbeddingVisualizer.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("EmbeddingVisualizer module loaded successfully")
    logger.info("This module is designed to be imported, not run directly")
    logger.info("Use main.py to run the complete Word2Vec pipeline")


if __name__ == "__main__":
    main()
