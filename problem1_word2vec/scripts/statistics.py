"""
Corpus Statistics and Word Cloud Generator
Dhruva Kumar Kaushal (B22AI017)
NLU Assignment - Problem 1

This module generates comprehensive statistics about the corpus and creates
word cloud visualizations to understand the data characteristics.

Statistics include:
- Document counts and sizes
- Vocabulary statistics
- Word frequency distributions
- Language distribution
- Corpus quality metrics

Original implementation for corpus analysis and reporting.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud


class CorpusStatistics:
    """
    Generates comprehensive statistics about the text corpus.
    
    Provides insights into corpus characteristics that are important for
    understanding the quality and properties of the learned embeddings.
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialize corpus statistics generator.
        
        Args:
            data_dir: Path to directory containing corpus data
        """
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
    
    def load_corpus_data(self) -> Dict:
        """
        Load processed corpus data.
        
        Returns:
            Dictionary containing corpus data and metadata
        """
        processed_dir = self.data_dir / "processed"
        
        # Load processed text
        text_file = processed_dir / "processed_text.txt"
        sentences_file = processed_dir / "sentences.txt"
        metadata_file = processed_dir / "metadata.json"
        
        if not text_file.exists():
            self.logger.error(f"Processed text file not found: {text_file}")
            return {}
        
        # Read text data
        with open(text_file, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        sentences = []
        if sentences_file.exists():
            with open(sentences_file, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f if line.strip()]
        
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        self.logger.info(f"Loaded corpus: {len(sentences)} sentences, "
                        f"{len(full_text.split())} words")
        
        return {
            'full_text': full_text,
            'sentences': sentences,
            'metadata': metadata
        }
    
    def compute_statistics(self, corpus_data: Dict) -> Dict:
        """
        Compute comprehensive corpus statistics.
        
        Args:
            corpus_data: Dictionary containing corpus data
            
        Returns:
            Dictionary containing various statistics
        """
        self.logger.info("Computing corpus statistics...")
        
        full_text = corpus_data.get('full_text', '')
        sentences = corpus_data.get('sentences', [])
        metadata = corpus_data.get('metadata', {})
        
        # Tokenize
        words = full_text.split()
        
        # Basic counts
        n_sentences = len(sentences)
        n_words = len(words)
        n_unique_words = len(set(words))
        
        # Word lengths
        word_lengths = [len(word) for word in words]
        avg_word_length = np.mean(word_lengths) if word_lengths else 0
        
        # Sentence lengths
        sentence_lengths = [len(sent.split()) for sent in sentences]
        avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
        max_sentence_length = max(sentence_lengths) if sentence_lengths else 0
        min_sentence_length = min(sentence_lengths) if sentence_lengths else 0
        
        # Word frequency
        word_counts = Counter(words)
        most_common = word_counts.most_common(50)
        
        # Vocabulary richness (Type-Token Ratio)
        ttr = n_unique_words / n_words if n_words > 0 else 0
        
        # Hapax legomena (words appearing only once)
        hapax_count = sum(1 for count in word_counts.values() if count == 1)
        hapax_ratio = hapax_count / n_unique_words if n_unique_words > 0 else 0
        
        # Frequency distribution statistics
        frequencies = list(word_counts.values())
        frequency_stats = {
            'mean': float(np.mean(frequencies)),
            'median': float(np.median(frequencies)),
            'std': float(np.std(frequencies)),
            'min': int(np.min(frequencies)),
            'max': int(np.max(frequencies))
        }
        
        # Compile statistics
        stats = {
            'basic': {
                'n_documents': metadata.get('n_documents', 0),
                'n_sentences': n_sentences,
                'n_words': n_words,
                'n_unique_words': n_unique_words,
                'type_token_ratio': float(ttr)
            },
            'word_statistics': {
                'avg_word_length': float(avg_word_length),
                'min_word_length': min(word_lengths) if word_lengths else 0,
                'max_word_length': max(word_lengths) if word_lengths else 0
            },
            'sentence_statistics': {
                'avg_sentence_length': float(avg_sentence_length),
                'min_sentence_length': int(min_sentence_length),
                'max_sentence_length': int(max_sentence_length),
                'std_sentence_length': float(np.std(sentence_lengths)) if sentence_lengths else 0
            },
            'vocabulary': {
                'hapax_legomena_count': hapax_count,
                'hapax_ratio': float(hapax_ratio),
                'most_common_words': [
                    {'word': word, 'count': count} for word, count in most_common
                ]
            },
            'frequency_distribution': frequency_stats,
            'metadata': metadata
        }
        
        self.logger.info(f"Statistics computed: {n_words} words, {n_unique_words} unique")
        
        return stats
    
    def generate_word_cloud(self, 
                           corpus_data: Dict, 
                           output_path: Path,
                           max_words: int = 200,
                           width: int = 1600,
                           height: int = 800) -> None:
        """
        Generate and save a word cloud visualization.
        
        Args:
            corpus_data: Dictionary containing corpus data
            output_path: Path to save word cloud image
            max_words: Maximum number of words to include
            width: Image width in pixels
            height: Image height in pixels
        """
        self.logger.info("Generating word cloud...")
        
        full_text = corpus_data.get('full_text', '')
        
        if not full_text:
            self.logger.warning("No text available for word cloud")
            return
        
        # Create word cloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            max_words=max_words,
            background_color='white',
            colormap='viridis',
            relative_scaling=0.5,
            min_font_size=10
        ).generate(full_text)
        
        # Create plot
        plt.figure(figsize=(20, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - IIT Jodhpur Corpus', 
                 fontsize=20, pad=20)
        plt.tight_layout(pad=0)
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Word cloud saved to {output_path}")
    
    def generate_frequency_plots(self,
                                corpus_data: Dict,
                                output_dir: Path) -> None:
        """
        Generate frequency distribution plots.
        
        Args:
            corpus_data: Dictionary containing corpus data
            output_dir: Directory to save plots
        """
        self.logger.info("Generating frequency plots...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        full_text = corpus_data.get('full_text', '')
        words = full_text.split()
        word_counts = Counter(words)
        
        # Plot 1: Top 30 most common words
        fig, ax = plt.subplots(figsize=(12, 6))
        most_common = word_counts.most_common(30)
        words_plot = [w for w, _ in most_common]
        counts_plot = [c for _, c in most_common]
        
        ax.barh(words_plot, counts_plot, color='steelblue')
        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_title('Top 30 Most Frequent Words', fontsize=14)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_dir / 'top_words.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Word frequency distribution (Zipf's law)
        fig, ax = plt.subplots(figsize=(10, 6))
        frequencies = sorted(word_counts.values(), reverse=True)
        ranks = range(1, len(frequencies) + 1)
        
        ax.loglog(ranks, frequencies, 'b-', alpha=0.7)
        ax.set_xlabel('Rank (log scale)', fontsize=12)
        ax.set_ylabel('Frequency (log scale)', fontsize=12)
        ax.set_title("Word Frequency Distribution (Zipf's Law)", fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'zipf_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Sentence length distribution
        sentences = corpus_data.get('sentences', [])
        sentence_lengths = [len(sent.split()) for sent in sentences]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(sentence_lengths, bins=50, color='coral', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Sentence Length (words)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Sentence Lengths', fontsize=14)
        ax.axvline(np.mean(sentence_lengths), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(sentence_lengths):.1f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'sentence_length_dist.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Frequency plots saved to {output_dir}")
    
    def save_statistics(self, stats: Dict, output_path: Path) -> None:
        """
        Save statistics to JSON file.
        
        Args:
            stats: Statistics dictionary
            output_path: Path to save JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Statistics saved to {output_path}")
    
    def generate_report(self, stats: Dict) -> str:
        """
        Generate a human-readable statistics report.
        
        Args:
            stats: Statistics dictionary
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("CORPUS STATISTICS REPORT")
        report.append("IIT Jodhpur Web Corpus")
        report.append("=" * 80)
        report.append("")
        
        # Basic statistics
        basic = stats.get('basic', {})
        report.append("BASIC STATISTICS")
        report.append("-" * 80)
        report.append(f"Number of documents: {basic.get('n_documents', 0):,}")
        report.append(f"Number of sentences: {basic.get('n_sentences', 0):,}")
        report.append(f"Total words: {basic.get('n_words', 0):,}")
        report.append(f"Unique words: {basic.get('n_unique_words', 0):,}")
        report.append(f"Type-Token Ratio: {basic.get('type_token_ratio', 0):.4f}")
        report.append("")
        
        # Word statistics
        word_stats = stats.get('word_statistics', {})
        report.append("WORD STATISTICS")
        report.append("-" * 80)
        report.append(f"Average word length: {word_stats.get('avg_word_length', 0):.2f} characters")
        report.append(f"Min word length: {word_stats.get('min_word_length', 0)}")
        report.append(f"Max word length: {word_stats.get('max_word_length', 0)}")
        report.append("")
        
        # Sentence statistics
        sent_stats = stats.get('sentence_statistics', {})
        report.append("SENTENCE STATISTICS")
        report.append("-" * 80)
        report.append(f"Average sentence length: {sent_stats.get('avg_sentence_length', 0):.2f} words")
        report.append(f"Min sentence length: {sent_stats.get('min_sentence_length', 0)} words")
        report.append(f"Max sentence length: {sent_stats.get('max_sentence_length', 0)} words")
        report.append(f"Std deviation: {sent_stats.get('std_sentence_length', 0):.2f}")
        report.append("")
        
        # Vocabulary
        vocab = stats.get('vocabulary', {})
        report.append("VOCABULARY ANALYSIS")
        report.append("-" * 80)
        report.append(f"Hapax legomena (words appearing once): {vocab.get('hapax_legomena_count', 0):,}")
        report.append(f"Hapax ratio: {vocab.get('hapax_ratio', 0):.2%}")
        report.append("")
        report.append("Top 20 Most Common Words:")
        for i, word_info in enumerate(vocab.get('most_common_words', [])[:20], 1):
            report.append(f"  {i:2d}. {word_info['word']:20s} - {word_info['count']:6,} occurrences")
        report.append("")
        
        # Frequency distribution
        freq_dist = stats.get('frequency_distribution', {})
        report.append("FREQUENCY DISTRIBUTION")
        report.append("-" * 80)
        report.append(f"Mean frequency: {freq_dist.get('mean', 0):.2f}")
        report.append(f"Median frequency: {freq_dist.get('median', 0):.2f}")
        report.append(f"Std deviation: {freq_dist.get('std', 0):.2f}")
        report.append(f"Min frequency: {freq_dist.get('min', 0)}")
        report.append(f"Max frequency: {freq_dist.get('max', 0)}")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """
    Main function for standalone execution.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate corpus statistics')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='outputs/statistics',
                       help='Path to output directory')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create statistics generator
    stats_gen = CorpusStatistics(Path(args.data_dir))
    
    # Load corpus
    corpus_data = stats_gen.load_corpus_data()
    
    if not corpus_data:
        print("Error: Could not load corpus data")
        return
    
    # Compute statistics
    stats = stats_gen.compute_statistics(corpus_data)
    
    # Generate outputs
    output_dir = Path(args.output_dir)
    stats_gen.save_statistics(stats, output_dir / 'corpus_statistics.json')
    
    # Generate report
    report = stats_gen.generate_report(stats)
    print(report)
    
    # Save report
    with open(output_dir / 'corpus_report.txt', 'w') as f:
        f.write(report)
    
    # Generate visualizations
    stats_gen.generate_word_cloud(corpus_data, output_dir / 'wordcloud.png')
    stats_gen.generate_frequency_plots(corpus_data, output_dir)
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
