"""
Analogy Testing Module for Word2Vec Models
Dhruva Kumar Kaushal (B22AI017)
NLU Assignment - Problem 1

This module implements word analogy testing using the classic vector arithmetic
approach: king - man + woman ≈ queen

The analogy task tests the model's ability to capture semantic relationships.
For analogy (a, b, c, d): "a is to b as c is to d"
We compute: embedding(b) - embedding(a) + embedding(c) ≈ embedding(d)

Original implementation following Word2Vec paper's evaluation methodology.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path
import json


class AnalogyTester:
    """
    Tests word embeddings on analogy tasks using vector arithmetic.
    
    Implements the evaluation methodology from Mikolov et al. (2013):
    "Efficient Estimation of Word Representations in Vector Space"
    
    Attributes:
        embeddings (np.ndarray): Word embedding matrix (vocab_size x embedding_dim)
        word2idx (dict): Mapping from words to indices
        idx2word (dict): Mapping from indices to words
    """
    
    def __init__(self, embeddings: np.ndarray, word2idx: dict, idx2word: dict):
        """
        Initialize the analogy tester.
        
        Args:
            embeddings: Word embedding matrix
            word2idx: Word to index mapping
            idx2word: Index to word mapping
        """
        self.embeddings = embeddings
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.logger = logging.getLogger(__name__)
        
        # Normalize embeddings for faster cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        self.normalized_embeddings = embeddings / norms
        
        self.logger.info(f"Initialized AnalogyTester with {len(word2idx)} words")
    
    def solve_analogy(self, a: str, b: str, c: str, top_k: int = 5, 
                     exclude_input: bool = True) -> List[Tuple[str, float]]:
        """
        Solve the analogy: "a is to b as c is to ?"
        
        Uses vector arithmetic: b - a + c ≈ d
        
        Args:
            a: First word (e.g., "king")
            b: Second word (e.g., "man")
            c: Third word (e.g., "queen")
            top_k: Number of candidates to return
            exclude_input: Whether to exclude input words from results
            
        Returns:
            List of (word, similarity) tuples, sorted by similarity
            
        Example:
            >>> tester.solve_analogy("king", "man", "woman")
            [("queen", 0.85), ("princess", 0.78), ...]
        """
        # Check if all words are in vocabulary
        missing_words = []
        for word in [a, b, c]:
            if word not in self.word2idx:
                missing_words.append(word)
        
        if missing_words:
            self.logger.warning(f"Words not in vocabulary: {missing_words}")
            return []
        
        # Get embeddings for input words
        idx_a = self.word2idx[a]
        idx_b = self.word2idx[b]
        idx_c = self.word2idx[c]
        
        # Vector arithmetic: b - a + c
        target_vector = (self.normalized_embeddings[idx_b] - 
                        self.normalized_embeddings[idx_a] + 
                        self.normalized_embeddings[idx_c])
        
        # Normalize the target vector
        target_norm = np.linalg.norm(target_vector)
        if target_norm > 0:
            target_vector = target_vector / target_norm
        
        # Compute cosine similarities with all words
        similarities = np.dot(self.normalized_embeddings, target_vector)
        
        # Exclude input words if requested
        if exclude_input:
            similarities[idx_a] = -np.inf
            similarities[idx_b] = -np.inf
            similarities[idx_c] = -np.inf
        
        # Get top-k most similar words
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (self.idx2word[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def test_analogy_set(self, analogies: List[Tuple[str, str, str, str]],
                         top_k: int = 5) -> Dict:
        """
        Test a set of analogies and compute accuracy.
        
        Args:
            analogies: List of (a, b, c, d) tuples where d is the expected answer
            top_k: Consider top-k predictions for accuracy (accuracy@k)
            
        Returns:
            Dictionary containing:
                - total: Total number of analogies tested
                - correct_at_1: Accuracy when considering top-1 prediction
                - correct_at_k: Accuracy when considering top-k predictions
                - results: Detailed results for each analogy
        """
        results = []
        correct_at_1 = 0
        correct_at_k = 0
        total = 0
        skipped = 0
        
        for a, b, c, expected_d in analogies:
            # Solve the analogy
            predictions = self.solve_analogy(a, b, c, top_k=top_k)
            
            if not predictions:
                skipped += 1
                self.logger.debug(f"Skipped: {a} - {b} + {c} (words not in vocab)")
                continue
            
            total += 1
            
            # Check if expected answer is in predictions
            predicted_words = [word for word, _ in predictions]
            
            if predicted_words and predicted_words[0].lower() == expected_d.lower():
                correct_at_1 += 1
                correct_at_k += 1
            elif any(word.lower() == expected_d.lower() for word in predicted_words):
                correct_at_k += 1
            
            results.append({
                'analogy': f"{a} - {b} + {c} = {expected_d}",
                'predictions': predictions,
                'correct': predicted_words[0].lower() == expected_d.lower() if predicted_words else False
            })
        
        # Calculate accuracies
        accuracy_at_1 = correct_at_1 / total if total > 0 else 0
        accuracy_at_k = correct_at_k / total if total > 0 else 0
        
        summary = {
            'total_tested': total,
            'skipped': skipped,
            'correct_at_1': correct_at_1,
            'correct_at_k': correct_at_k,
            'accuracy_at_1': accuracy_at_1,
            'accuracy_at_k': accuracy_at_k,
            'top_k': top_k,
            'results': results
        }
        
        self.logger.info(f"Analogy test: {correct_at_1}/{total} correct (top-1), "
                        f"{correct_at_k}/{total} correct (top-{top_k})")
        
        return summary
    
    @staticmethod
    def create_analogy_dataset() -> List[Tuple[str, str, str, str]]:
        """
        Create a diverse set of analogies for testing.
        
        Returns comprehensive analogies covering:
        - Gender relationships
        - Geography (capital cities, countries)
        - Comparative/superlative forms
        - Verb tenses
        - Academic/technical relationships
        
        Returns:
            List of (a, b, c, d) tuples representing analogies
        """
        analogies = [
            # Gender relationships
            ("king", "man", "woman", "queen"),
            ("prince", "boy", "girl", "princess"),
            ("brother", "man", "woman", "sister"),
            ("father", "man", "woman", "mother"),
            ("uncle", "man", "woman", "aunt"),
            ("nephew", "man", "woman", "niece"),
            ("husband", "man", "woman", "wife"),
            ("actor", "man", "woman", "actress"),
            
            # Geography - Capital cities
            ("delhi", "india", "france", "paris"),
            ("washington", "usa", "uk", "london"),
            ("beijing", "china", "japan", "tokyo"),
            ("moscow", "russia", "germany", "berlin"),
            ("ottawa", "canada", "australia", "canberra"),
            
            # Geography - Country relationships
            ("india", "indian", "france", "french"),
            ("japan", "japanese", "china", "chinese"),
            ("america", "american", "canada", "canadian"),
            
            # Comparative/Superlative
            ("good", "better", "bad", "worse"),
            ("big", "bigger", "small", "smaller"),
            ("fast", "faster", "slow", "slower"),
            ("high", "higher", "low", "lower"),
            
            # Verb tenses
            ("go", "went", "do", "did"),
            ("run", "running", "walk", "walking"),
            ("see", "saw", "eat", "ate"),
            ("write", "wrote", "read", "read"),
            
            # Academic relationships (relevant to IIT context)
            ("student", "university", "patient", "hospital"),
            ("professor", "university", "doctor", "hospital"),
            ("research", "science", "experiment", "laboratory"),
            ("lecture", "classroom", "experiment", "laboratory"),
            ("bachelor", "undergraduate", "master", "graduate"),
            
            # Technical relationships
            ("code", "programming", "equation", "mathematics"),
            ("algorithm", "computer", "theorem", "mathematics"),
            ("python", "programming", "calculus", "mathematics"),
            
            # Plural forms
            ("cat", "cats", "dog", "dogs"),
            ("book", "books", "paper", "papers"),
            ("student", "students", "professor", "professors"),
            
            # Opposites
            ("hot", "cold", "fast", "slow"),
            ("dark", "light", "hard", "soft"),
            ("easy", "difficult", "simple", "complex"),
        ]
        
        return analogies
    
    def save_results(self, results: Dict, output_path: Path):
        """
        Save analogy test results to JSON file.
        
        Args:
            results: Results dictionary from test_analogy_set
            output_path: Path to save JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved analogy results to {output_path}")
    
    def generate_report(self, results: Dict) -> str:
        """
        Generate a human-readable report from analogy test results.
        
        Args:
            results: Results dictionary from test_analogy_set
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("WORD ANALOGY TEST RESULTS")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        report.append(f"Total analogies tested: {results['total_tested']}")
        report.append(f"Skipped (words not in vocab): {results['skipped']}")
        report.append(f"Correct predictions (top-1): {results['correct_at_1']}/{results['total_tested']}")
        report.append(f"Correct predictions (top-{results['top_k']}): {results['correct_at_k']}/{results['total_tested']}")
        report.append(f"Accuracy @ 1: {results['accuracy_at_1']:.2%}")
        report.append(f"Accuracy @ {results['top_k']}: {results['accuracy_at_k']:.2%}")
        report.append("")
        
        # Sample results
        report.append("=" * 80)
        report.append("SAMPLE PREDICTIONS (First 10)")
        report.append("=" * 80)
        report.append("")
        
        for i, result in enumerate(results['results'][:10], 1):
            report.append(f"{i}. {result['analogy']}")
            report.append(f"   Predicted: {result['predictions'][0][0]} "
                         f"(similarity: {result['predictions'][0][1]:.3f})")
            report.append(f"   Status: {'✓ CORRECT' if result['correct'] else '✗ INCORRECT'}")
            report.append("")
        
        return "\n".join(report)


def main():
    """
    Example usage and testing of the AnalogyTester.
    """
    # This is for testing purposes only
    # In actual use, this module is imported by main.py
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("AnalogyTester module loaded successfully")
    logger.info("This module is designed to be imported, not run directly")
    logger.info("Use main.py to run the complete Word2Vec pipeline")


if __name__ == "__main__":
    main()
