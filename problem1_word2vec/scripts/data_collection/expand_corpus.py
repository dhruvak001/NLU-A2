"""
Extended Data Collection Script
================================

This script expands the corpus by:
1. Using the existing collected data
2. Fetching additional linked pages from department websites  
3. Creating variations through different URL patterns

Author: Dhruva Kumar Kaushal (B22AI017)
"""

import requests
from bs4 import BeautifulSoup
import time
import json
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_and_save(url, category_dir, doc_id, category):
    """Fetch URL and save content."""
    try:
        logger.info(f"Fetching: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'button', 'form']):
            tag.decompose()
        
        text = soup.get_text(separator=' ', strip=True)
        text = ' '.join(text.split())
        
        word_count = len(text.split())
        
        if word_count < 100:
            return None
        
        filename = f"{category}_{doc_id:03d}.txt"
        filepath = os.path.join(category_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.info(f"✓ Saved: {word_count:,} words")
        time.sleep(2)
        
        return {
            'doc_id': f"{category}_{doc_id:03d}",
            'category': category,
            'url': url,
            'filepath': filepath,
            'word_count': word_count,
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        return None


def expand_corpus():
    """Expand the corpus with additional sources."""
    
    output_dir = 'data/raw'
    
    # Extended URL list - more comprehensive
    extended_sources = {
        'academic_regulations': [
            'https://www.iitj.ac.in',
            'https://www.iitj.ac.in/main/en/introduction',
            'https://www.iitj.ac.in/main/en/vision-mission',
        ],
        'departments': [
            'https://www.iitj.ac.in/computer-science-engineering',
            'https://www.iitj.ac.in/electrical-engineering',
            'https://www.iitj.ac.in/mechanical-engineering',
            'https://www.iitj.ac.in/chemistry',
            'https://www.iitj.ac.in/physics',
            'https://www.iitj.ac.in/mathematics',
            'https://www.iitj.ac.in/humanities-social-sciences',
            'https://www.iitj.ac.in/bioscience-bioengineering',
            'https://www.iitj.ac.in/metallurgical-materials-engineering',
            'https://www.iitj.ac.in/computer-science-engineering/en/about',
            'https://www.iitj.ac.in/electrical-engineering/en/about',
            'https://www.iitj.ac.in/mechanical-engineering/en/about',
        ],
        'research': [
            'https://www.iitj.ac.in/main/en/research-highlight',
            'https://www.iitj.ac.in/computer-science-engineering/en/Research-Highlights',
            'https://www.iitj.ac.in/electrical-engineering/en/Research-Highlights',
            'https://www.iitj.ac.in/mechanical-engineering/en/Research-Highlights',
        ],
        'faculty': [
            'https://www.iitj.ac.in/computer-science-engineering/en/people',
            'https://www.iitj.ac.in/electrical-engineering/en/people',
            'https://www.iitj.ac.in/mechanical-engineering/en/people',
        ],
        'courses': [
            'https://www.iitj.ac.in/computer-science-engineering/en/courses',
            'https://www.iitj.ac.in/electrical-engineering/en/courses',
            'https://www.iitj.ac.in/mechanical-engineering/en/courses',
        ],
        'about': [
            'https://www.iitj.ac.in/office-of-students/en/campus-life',
            'https://www.iitj.ac.in/main/en/iitj',
            'https://www.iitj.ac.in/main/en/director',
            'https://www.iitj.ac.in/main/en/chairman',
        ]
    }
    
    all_metadata = []
    doc_id_counter = 0
    
    print("\n" + "="*70)
    print("EXPANDING CORPUS WITH ADDITIONAL SOURCES")
    print("="*70 + "\n")
    
    for category, urls in extended_sources.items():
        print(f"[{category.upper()}]")
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        for url in urls:
            metadata = fetch_and_save(url, category_dir, doc_id_counter, category)
            if metadata:
                all_metadata.append(metadata)
                doc_id_counter += 1
    
    # Update metadata
    metadata_file = os.path.join(output_dir, 'all_metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2)
    
    total_words = sum(m['word_count'] for m in all_metadata)
    
    print("\n" + "="*70)
    print(f"✓ Total documents: {len(all_metadata)}")
    print(f"✓ Total words: {total_words:,}")
    print(f"✓ Avg words/doc: {total_words//len(all_metadata) if all_metadata else 0:,}")
    print("="*70 + "\n")
    
    return all_metadata


if __name__ == '__main__':
    expand_corpus()
