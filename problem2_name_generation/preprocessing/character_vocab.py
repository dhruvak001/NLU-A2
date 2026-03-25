"""
Character Vocabulary Builder
Dhruva Kumar Kaushal (B22AI017)
NLU Assignment - Problem 2

This module builds a character-level vocabulary from the name dataset.
Includes special tokens for sequence modeling.

Original implementation for character-level language modeling.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Set
from collections import Counter


class CharacterVocabulary:
    """
    Builds and manages character-level vocabulary for name generation.
    
    Handles:
    - Character extraction from names
    - Special token management (<PAD>, <SOS>, <EOS>, <UNK>)
    - Character-to-index and index-to-character mappings
    - Vocabulary statistics
    """
    
    # Special tokens
    PAD_TOKEN = '<PAD>'
    SOS_TOKEN = '<SOS>'  # Start of sequence
    EOS_TOKEN = '<EOS>'  # End of sequence
    UNK_TOKEN = '<UNK>'  # Unknown character
    
    def __init__(self, lowercase: bool = True, max_vocab_size: int = 100):
        """
        Initialize character vocabulary builder.
        
        Args:
            lowercase: Whether to convert all characters to lowercase
            max_vocab_size: Maximum vocabulary size (including special tokens)
        """
        self.lowercase = lowercase
        self.max_vocab_size = max_vocab_size
        self.logger = logging.getLogger(__name__)
        
        # Vocabulary mappings
        self.char2idx = {}
        self.idx2char = {}
        self.char_counts = Counter()
        
        # Initialize with special tokens
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Initialize special tokens in vocabulary."""
        special_tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        
        for idx, token in enumerate(special_tokens):
            self.char2idx[token] = idx
            self.idx2char[idx] = token
        
        self.logger.info(f"Initialized {len(special_tokens)} special tokens")
    
    def build_vocab(self, names: List[str]) -> Dict:
        """
        Build vocabulary from list of names.
        
        Args:
            names: List of name strings
            
        Returns:
            Dictionary containing vocabulary statistics
        """
        self.logger.info(f"Building vocabulary from {len(names)} names...")
        
        # Extract all characters
        for name in names:
            if self.lowercase:
                name = name.lower()
            
            for char in name:
                self.char_counts[char] += 1
        
        # Get most common characters (excluding special tokens)
        n_special = len([t for t in self.char2idx if t.startswith('<')])
        max_regular_chars = self.max_vocab_size - n_special
        
        most_common = self.char_counts.most_common(max_regular_chars)
        
        # Add to vocabulary
        current_idx = len(self.char2idx)
        for char, count in most_common:
            if char not in self.char2idx:
                self.char2idx[char] = current_idx
                self.idx2char[current_idx] = char
                current_idx += 1
        
        # Calculate statistics
        stats = {
            'vocab_size': len(self.char2idx),
            'n_special_tokens': n_special,
            'n_special': n_special,
            'n_regular_chars': len(self.char2idx) - n_special,
            'total_chars_seen': sum(self.char_counts.values()),
            'unique_chars_seen': len(self.char_counts),
            'most_common': most_common[:20]
        }
        
        self.logger.info(f"Vocabulary built: {stats['vocab_size']} characters "
                        f"({stats['n_regular_chars']} regular + {stats['n_special_tokens']} special)")
        
        return stats
    
    def build_from_dataset(self, dataset_path: Path) -> Dict:
        """
        Build vocabulary from JSON dataset file.
        
        Args:
            dataset_path: Path to JSON file containing name data
            
        Returns:
            Dictionary containing vocabulary statistics
        """
        self.logger.info(f"Loading dataset from {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract names
        if isinstance(data, list):
            # If data is list of name dicts
            names = [item['name'] for item in data if 'name' in item]
        else:
            # If data is a dictionary
            names = data.get('names', [])
        
        return self.build_vocab(names)
    
    def encode(self, name: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a name into sequence of character indices.
        
        Args:
            name: Name string to encode
            add_special_tokens: Whether to add <SOS> and <EOS> tokens
            
        Returns:
            List of character indices
        """
        if self.lowercase:
            name = name.lower()
        
        # Convert characters to indices
        indices = []
        
        if add_special_tokens:
            indices.append(self.char2idx[self.SOS_TOKEN])
        
        for char in name:
            if char in self.char2idx:
                indices.append(self.char2idx[char])
            else:
                indices.append(self.char2idx[self.UNK_TOKEN])
        
        if add_special_tokens:
            indices.append(self.char2idx[self.EOS_TOKEN])
        
        return indices
    
    def decode(self, indices: List[int], remove_special_tokens: bool = True) -> str:
        """
        Decode sequence of indices back to name string.
        
        Args:
            indices: List of character indices
            remove_special_tokens: Whether to remove special tokens from output
            
        Returns:
            Decoded name string
        """
        chars = []
        
        for idx in indices:
            if idx in self.idx2char:
                char = self.idx2char[idx]
                
                if remove_special_tokens and char.startswith('<') and char.endswith('>'):
                    continue
                
                chars.append(char)
        
        return ''.join(chars)
    
    def encode_batch(self, names: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """
        Encode batch of names.
        
        Args:
            names: List of name strings
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of encoded sequences
        """
        return [self.encode(name, add_special_tokens) for name in names]
    
    def decode_batch(self, sequences: List[List[int]], 
                     remove_special_tokens: bool = True) -> List[str]:
        """
        Decode batch of sequences.
        
        Args:
            sequences: List of encoded sequences
            remove_special_tokens: Whether to remove special tokens
            
        Returns:
            List of decoded name strings
        """
        return [self.decode(seq, remove_special_tokens) for seq in sequences]
    
    def pad_sequence(self, sequence: List[int], max_length: int, 
                     padding: str = 'post') -> List[int]:
        """
        Pad sequence to specified length.
        
        Args:
            sequence: Sequence to pad
            max_length: Target length
            padding: 'pre' or 'post' padding
            
        Returns:
            Padded sequence
        """
        pad_idx = self.char2idx[self.PAD_TOKEN]
        
        if len(sequence) >= max_length:
            return sequence[:max_length]
        
        padding_length = max_length - len(sequence)
        padding_vec = [pad_idx] * padding_length
        
        if padding == 'post':
            return sequence + padding_vec
        else:
            return padding_vec + sequence
    
    def get_vocab_stats(self) -> Dict:
        """
        Get vocabulary statistics.
        
        Returns:
            Dictionary containing various statistics
        """
        special_tokens = [char for char in self.char2idx if char.startswith('<')]
        regular_chars = [char for char in self.char2idx if not char.startswith('<')]
        
        # Categorize characters
        letters = [c for c in regular_chars if c.isalpha()]
        digits = [c for c in regular_chars if c.isdigit()]
        special_chars = [c for c in regular_chars if not c.isalnum()]
        
        stats = {
            'total_vocab_size': len(self.char2idx),
            'special_tokens': len(special_tokens),
            'regular_chars': len(regular_chars),
            'letters': len(letters),
            'digits': len(digits),
            'special_characters': len(special_chars),
            'special_token_list': special_tokens,
            'sample_chars': regular_chars[:50]
        }
        
        return stats
    
    def save(self, output_path: Path):
        """
        Save vocabulary to JSON file.
        
        Args:
            output_path: Path to save vocabulary
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        vocab_data = {
            'char2idx': self.char2idx,
            'idx2char': {int(k): v for k, v in self.idx2char.items()},
            'char_counts': dict(self.char_counts.most_common()),
            'config': {
                'lowercase': self.lowercase,
                'max_vocab_size': self.max_vocab_size,
                'vocab_size': len(self.char2idx)
            },
            'special_tokens': {
                'PAD': self.PAD_TOKEN,
                'SOS': self.SOS_TOKEN,
                'EOS': self.EOS_TOKEN,
                'UNK': self.UNK_TOKEN
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Vocabulary saved to {output_path}")
    
    @classmethod
    def load(cls, vocab_path: Path) -> 'CharacterVocabulary':
        """
        Load vocabulary from JSON file.
        
        Args:
            vocab_path: Path to vocabulary file
            
        Returns:
            CharacterVocabulary instance
        """
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        config = vocab_data.get('config', {})
        vocab = cls(
            lowercase=config.get('lowercase', True),
            max_vocab_size=config.get('max_vocab_size', 100)
        )
        
        # Load mappings
        vocab.char2idx = vocab_data['char2idx']
        vocab.idx2char = {int(k): v for k, v in vocab_data['idx2char'].items()}
        vocab.char_counts = Counter(vocab_data.get('char_counts', {}))
        
        return vocab
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.char2idx)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"CharacterVocabulary(size={len(self)}, lowercase={self.lowercase})"


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build character vocabulary')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to name dataset JSON file')
    parser.add_argument('--output', type=str, default='data/processed/char_vocab.json',
                       help='Output vocabulary file path')
    parser.add_argument('--lowercase', action='store_true', default=True,
                       help='Convert to lowercase')
    parser.add_argument('--max-vocab-size', type=int, default=100,
                       help='Maximum vocabulary size')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Build vocabulary
    vocab = CharacterVocabulary(
        lowercase=args.lowercase,
        max_vocab_size=args.max_vocab_size
    )
    
    stats = vocab.build_from_dataset(Path(args.data_path))
    
    # Display statistics
    print("\nVocabulary Statistics:")
    print(json.dumps(stats, indent=2))
    
    print("\nVocabulary Info:")
    vocab_stats = vocab.get_vocab_stats()
    print(json.dumps(vocab_stats, indent=2))
    
    # Save vocabulary
    vocab.save(Path(args.output))
    
    # Test encoding/decoding
    print("\nTesting encoding/decoding:")
    test_names = ['Aarav', 'Priya', 'Kumar']
    for name in test_names:
        encoded = vocab.encode(name)
        decoded = vocab.decode(encoded)
        print(f"{name} -> {encoded} -> {decoded}")


if __name__ == "__main__":
    main()
