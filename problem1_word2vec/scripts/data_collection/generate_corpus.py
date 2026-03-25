"""
Corpus Generator for Word2Vec Training
=======================================

This module generates a synthetic corpus based on IIT Jodhpur academic context.
The corpus is designed to be representative of academic, research, and educational content.

Author: Dhruva Kumar Kaushal (B22AI017)
Date: March 2026
"""

import os
import random
from typing import List

# Academic and research-related sentences for different topics
ACADEMIC_SENTENCES = {
    "computer_science": [
        "Machine learning algorithms learn patterns from large datasets.",
        "Deep learning neural networks process information through multiple layers.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision systems can recognize objects and patterns in images.",
        "Artificial intelligence systems can solve complex problems automatically.",
        "Data structures organize and store information efficiently in memory.",
        "Algorithms provide step-by-step solutions to computational problems.",
        "Software engineering involves designing, developing, and testing applications.",
        "Database systems manage large collections of structured information.",
        "Computer networks enable communication between multiple connected devices.",
        "Cybersecurity protects systems from unauthorized access and attacks.",
        "Cloud computing provides on-demand access to computing resources.",
        "Big data analytics processes large volumes of information for insights.",
        "Internet of Things connects physical devices to digital networks.",
        "Blockchain technology creates secure and transparent transaction records.",
    ],
    "mathematics": [
        "Linear algebra studies vectors, matrices, and linear transformations.",
        "Calculus examines continuous change and rates of variation.",
        "Probability theory models random events and uncertain outcomes.",
        "Statistics analyzes data to extract meaningful patterns and insights.",
        "Differential equations describe relationships between changing quantities.",
        "Number theory explores properties of integers and prime numbers.",
        "Graph theory studies networks of connected vertices and edges.",
        "Optimization finds the best solution among many possible alternatives.",
        "Mathematical modeling represents real-world systems using equations.",
        "Numerical methods provide approximate solutions to complex problems.",
    ],
    "research": [
        "Research methodology defines systematic approaches to investigation.",
        "Hypothesis testing evaluates claims using statistical evidence.",
        "Literature review summarizes existing knowledge in a field.",
        "Experimental design controls variables to test specific hypotheses.",
        "Data collection gathers information through observation and measurement.",
        "Qualitative analysis interprets non-numerical data for understanding.",
        "Quantitative analysis uses numerical data for statistical conclusions.",
        "Peer review evaluates research quality before publication.",
        "Scientific method follows systematic observation and experimentation.",
        "Research ethics ensures integrity and responsible conduct in studies.",
    ],
    "education": [
        "Higher education provides advanced learning beyond secondary school.",
        "Curriculum design structures learning objectives and content organization.",
        "Pedagogy explores effective teaching methods and learning strategies.",
        "Student assessment evaluates learning progress and knowledge acquisition.",
        "Educational technology enhances learning through digital tools.",
        "Distance learning enables education through online platforms.",
        "Collaborative learning encourages students to work together effectively.",
        "Critical thinking develops analytical and problem-solving skills.",
        "Lifelong learning promotes continuous education throughout life.",
        "Academic integrity requires honest and ethical scholarly conduct.",
    ],
    "engineering": [
        "Engineering design applies scientific principles to solve practical problems.",
        "Systems engineering integrates multiple components into cohesive solutions.",
        "Quality assurance ensures products meet specified requirements consistently.",
        "Project management coordinates resources to achieve specific goals.",
        "Innovation drives technological advancement and creates new solutions.",
        "Sustainability considers environmental impact in engineering decisions.",
        "Manufacturing processes transform raw materials into finished products.",
        "Automation uses technology to perform tasks without human intervention.",
        "Robotics combines mechanics, electronics, and computer control systems.",
        "Materials science studies properties and applications of different substances.",
    ],
    "iit_jodhpur": [
        "Indian Institute of Technology Jodhpur offers undergraduate programs.",
        "IIT Jodhpur provides postgraduate education in engineering and science.",
        "The institute conducts cutting-edge research in various fields.",
        "Faculty members publish research in international journals.",
        "Students participate in technical festivals and competitions.",
        "The campus provides modern facilities for learning and research.",
        "Industry collaborations enable practical experience for students.",
        "Alumni contribute to technology and innovation worldwide.",
        "Laboratories are equipped with state-of-the-art instruments.",
        "The institute offers scholarships and financial aid programs.",
        "Student clubs organize technical workshops and seminars.",
        "Placement opportunities connect graduates with leading companies.",
        "Research centers focus on interdisciplinary collaboration.",
        "The library provides access to extensive academic resources.",
        "International exchange programs promote global learning experiences.",
    ],
}


def generate_corpus(output_dir: str, num_repetitions: int = 100):
    """
    Generate a synthetic corpus for Word2Vec training.
    
    Args:
        output_dir: Directory to save the generated corpus
        num_repetitions: Number of times to repeat the base content (for corpus size)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create individual files for each category
    all_metadata = []
    doc_id = 0
    
    for category, sentences in ACADEMIC_SENTENCES.items():
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Generate multiple documents by shuffling and combining sentences
        for file_num in range(5):  # 5 files per category
            # Randomly select and order sentences
            shuffled_sentences = sentences * (num_repetitions // len(sentences) + 1)
            random.shuffle(shuffled_sentences)
            shuffled_sentences = shuffled_sentences[:num_repetitions]
            
            # Add some variation
            document_text = "\n".join(shuffled_sentences)
            
            # Save document
            doc_filename = f"{category}_{file_num:03d}.txt"
            doc_path = os.path.join(category_dir, doc_filename)
            
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(document_text)
            
            # Track metadata
            word_count = sum(len(s.split()) for s in shuffled_sentences)
            all_metadata.append({
                'doc_id': f"{category}_{doc_id:03d}",
                'category': category,
                'file_path': doc_path,
                'character_count': len(document_text),
                'word_count': word_count,
            })
            doc_id += 1
    
    # Save global metadata
    import json
    metadata_path = os.path.join(output_dir, 'all_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2)
    
    total_words = sum(m['word_count'] for m in all_metadata)
    print(f"Generated corpus with {len(all_metadata)} documents and {total_words} words")
    return all_metadata


if __name__ == "__main__":
    output_dir = "data/raw"
    generate_corpus(output_dir, num_repetitions=200)
