"""
Text Preprocessing Pipeline for Word2Vec
========================================

This module implements a comprehensive preprocessing pipeline to clean and
prepare the scraped text data for Word2Vec training.

Author: Dhruva Kumar Kaushal (B22AI017)
Date: March 2026

Preprocessing Steps:
1. Language detection and filtering (English only)
2. Noise removal (boilerplate, navigation, etc.)
3. Tokenization (sentence and word level)
4. Lowercasing
5. Punctuation handling
6. Special character removal
7. Corpus assembly and validation
"""

import os
import re
import json
from typing import List, Dict, Set, Tuple
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from langdetect import detect, LangDetectException
from tqdm import tqdm
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Comprehensive text preprocessing class for Word2Vec corpus preparation.
    
    This class handles all preprocessing steps from raw text to clean,
    tokenized corpus ready for Word2Vec training.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config: Dictionary containing preprocessing parameters
        """
        self.config = config or {}
        self.min_sentence_length = self.config.get('min_sentence_length', 3)
        self.max_sentence_length = self.config.get('max_sentence_length', 100)
        self.lowercase = self.config.get('lowercase', True)
        self.remove_numbers = self.config.get('remove_numbers', True)
        
        # Boilerplate patterns to remove (common across IIT Jodhpur pages)
        self.boilerplate_patterns = [
            r'skip to (?:main )?content',
            r'home\s*\|\s*about\s*\|\s*contact',
            r'copyright.*iit jodhpur',
            r'quick links',
            r'latest news',
            r'all rights reserved',
            r'privacy policy',
            r'terms (?:and|&) conditions',
            r'social media',
            r'follow us',
            r'facebook\s*twitter\s*instagram',
        ]
        
        # Compile regex patterns for efficiency
        self.boilerplate_regex = [re.compile(pattern, re.IGNORECASE) 
                                  for pattern in self.boilerplate_patterns]
        
        # Statistics tracking
        self.stats = {
            'total_docs': 0,
            'filtered_non_english': 0,
            'total_sentences': 0,
            'total_tokens': 0,
            'removed_boilerplate_lines': 0
        }
    
    def detect_language(self, text: str, min_length: int = 50) -> str:
        """
        Detect the language of text.
        
        Args:
            text: Input text
            min_length: Minimum text length for reliable detection
            
        Returns:
            str: Language code ('en' for English) or 'unknown'
        """
        if len(text) < min_length:
            return 'unknown'
        
        try:
            return detect(text)
        except LangDetectException:
            return 'unknown'
    
    def filter_english_only(self, text: str) -> str:
        """
        Filter text to keep only English content.
        
        Processes at sentence level to handle code-mixed text.
        
        Args:
            text: Input text
            
        Returns:
            str: Filtered text containing only English sentences
        """
        # Check language at document level first
        doc_lang = self.detect_language(text)
        if doc_lang != 'en':
            return ''
        
        # If document is English, keep all sentences
        # (sentence-level detection can be too strict)
        return text
    
    def remove_boilerplate(self, text: str) -> str:
        """
        Remove boilerplate text common in web pages.
        
        This includes navigation menus, headers, footers, and other
        repetitive content that doesn't contribute to semantic learning.
        
        Args:
            text: Input text
            
        Returns:
            str: Text with boilerplate removed
        """
        # If text has no newlines, just return it after basic cleaning
        if '\n' not in text:
            # Remove obvious boilerplate phrases
            for pattern in self.boilerplate_regex:
                text = pattern.sub('', text)
            return text
        
        lines = text.split('\n')
        cleaned_lines = []
        removed_count = 0
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check against boilerplate patterns
            is_boilerplate = False
            for pattern in self.boilerplate_regex:
                if pattern.search(line):
                    is_boilerplate = True
                    removed_count += 1
                    break
            
            if not is_boilerplate:
                # Additional heuristics
                # Remove lines that are mostly non-alphabetic
                alpha_ratio = sum(c.isalpha() for c in line) / (len(line) + 1)
                if alpha_ratio > 0.5:
                    cleaned_lines.append(line)
                else:
                    removed_count += 1
        
        self.stats['removed_boilerplate_lines'] += removed_count
        return '\n'.join(cleaned_lines)
    
    def remove_urls_and_emails(self, text: str) -> str:
        """
        Remove URLs and email addresses from text.
        
        Args:
            text: Input text
            
        Returns:
            str: Text with URLs and emails removed
        """
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),])+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Apply various cleaning operations to text.
        
        Args:
            text: Input text
            
        Returns:
            str: Cleaned text
        """
        # Remove special markers from web scraping
        text = re.sub(r'###\d+\$\$\$_\w+_%%%\d+!!!', '', text)
        text = re.sub(r'arrow_downward\s*️?', '', text)
        
        # Remove URLs and emails
        text = self.remove_urls_and_emails(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove excessive punctuation (multiple consecutive punctuation marks)
        text = re.sub(r'[.!?]{2,}', '.', text)
        text = re.sub(r'[,;:]{2,}', ',', text)
        
        return text.strip()
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            list: List of sentences
        """
        sentences = sent_tokenize(text)
        
        # Filter sentences by length
        filtered_sentences = []
        for sentence in sentences:
            word_count = len(sentence.split())
            if self.min_sentence_length <= word_count <= self.max_sentence_length:
                filtered_sentences.append(sentence)
        
        return filtered_sentences
    
    def tokenize_words(self, sentence: str) -> List[str]:
        """
        Tokenize a sentence into words.
        
        Args:
            sentence: Input sentence
            
        Returns:
            list: List of word tokens
        """
        # Use NLTK word tokenizer
        tokens = word_tokenize(sentence)
        
        # Clean each token
        cleaned_tokens = []
        for token in tokens:
            # Lowercase if configured
            if self.lowercase:
                token = token.lower()
            
            # Remove pure punctuation tokens
            if token.isalpha() or (token.isalnum() and any(c.isalpha() for c in token)):
                # Keep tokens that are alphabetic or alphanumeric with at least one letter
                if len(token) >= 2:  # Minimum token length
                    cleaned_tokens.append(token)
        
        return cleaned_tokens
    
    def process_document(self, text: str) -> List[List[str]]:
        """
        Process a single document through the complete pipeline.
        
        Args:
            text: Raw document text
            
        Returns:
            list: List of tokenized sentences (each sentence is a list of tokens)
        """
        # Step 1: Filter English only
        text = self.filter_english_only(text)
        if not text:
            return []
        
        # Step 2: Remove boilerplate
        text = self.remove_boilerplate(text)
        
        # Step 3: Clean text
        text = self.clean_text(text)
        
        # Step 4: Tokenize into sentences
        sentences = self.tokenize_sentences(text)
        
        # Step 5: Tokenize each sentence into words
        tokenized_sentences = []
        for sentence in sentences:
            tokens = self.tokenize_words(sentence)
            if tokens:  # Only keep non-empty sentences
                tokenized_sentences.append(tokens)
                self.stats['total_tokens'] += len(tokens)
        
        self.stats['total_sentences'] += len(tokenized_sentences)
        
        return tokenized_sentences
    
    def process_corpus(self, input_dir: str) -> List[List[str]]:
        """
        Process all documents in the corpus.
        
        Args:
            input_dir: Directory containing raw text files
            
        Returns:
            list: List of all tokenized sentences from all documents
        """
        all_sentences = []
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(input_dir):
            txt_files = [f for f in files if f.endswith('.txt')]
            
            for filename in tqdm(txt_files, desc=f"Processing {os.path.basename(root)}"):
                filepath = os.path.join(root, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    # Check if English
                    lang = self.detect_language(text)
                    if lang != 'en':
                        self.stats['filtered_non_english'] += 1
                        logger.warning(f"Filtered non-English document: {filename}")
                        continue
                    
                    # Process document
                    tokenized_sentences = self.process_document(text)
                    all_sentences.extend(tokenized_sentences)
                    self.stats['total_docs'] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
        
        return all_sentences
    
    def save_corpus(self, sentences: List[List[str]], output_file: str):
        """
        Save processed corpus to file.
        
        Format: One sentence per line, tokens separated by spaces.
        
        Args:
            sentences: List of tokenized sentences
            output_file: Path to output file
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(' '.join(sentence) + '\n')
        
        logger.info(f"Corpus saved to {output_file}")
    
    def compute_statistics(self, sentences: List[List[str]]) -> Dict:
        """
        Compute statistics about the processed corpus.
        
        Args:
            sentences: List of tokenized sentences
            
        Returns:
            dict: Statistics including token counts, vocabulary size, etc.
        """
        # Count all tokens
        all_tokens = [token for sentence in sentences for token in sentence]
        token_counts = Counter(all_tokens)
        
        stats = {
            'total_documents': self.stats['total_docs'],
            'filtered_non_english': self.stats['filtered_non_english'],
            'total_sentences': len(sentences),
            'total_tokens': len(all_tokens),
            'vocabulary_size': len(token_counts),
            'unique_tokens': len(token_counts),
            'avg_sentence_length': len(all_tokens) / len(sentences) if sentences else 0,
            'most_common_tokens': token_counts.most_common(50),
            'removed_boilerplate_lines': self.stats['removed_boilerplate_lines']
        }
        
        return stats
    
    def save_statistics(self, stats: Dict, output_file: str):
        """
        Save corpus statistics to JSON file.
        
        Args:
            stats: Statistics dictionary
            output_file: Path to output file
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved to {output_file}")


def main():
    """
    Main function to run the preprocessing pipeline.
    """
    import yaml
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(config['preprocessing'])
    
    # Process corpus
    print("\n" + "="*60)
    print("PREPROCESSING CORPUS")
    print("="*60)
    
    input_dir = 'data/raw'
    sentences = preprocessor.process_corpus(input_dir)
    
    # Save processed corpus
    output_file = 'data/processed/corpus.txt'
    preprocessor.save_corpus(sentences, output_file)
    
    # Compute and save statistics
    stats = preprocessor.compute_statistics(sentences)
    preprocessor.save_statistics(stats, 'data/statistics/corpus_stats.json')
    
    # Print summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Total documents processed: {stats['total_documents']}")
    print(f"Total sentences: {stats['total_sentences']}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Vocabulary size: {stats['vocabulary_size']:,}")
    print(f"Average sentence length: {stats['avg_sentence_length']:.2f} tokens")
    print(f"Output file: {output_file}")
    print("="*60)


if __name__ == "__main__":
    main()
