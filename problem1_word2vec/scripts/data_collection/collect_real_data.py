"""
Comprehensive Data Collection Script for IIT Jodhpur
===================================================

This script collects real textual data from multiple IIT Jodhpur sources
to create a sufficient corpus for Word2Vec training.

Author: Dhruva Kumar Kaushal (B22AI017)
Date: March 2026
"""

import requests
from bs4 import BeautifulSoup
import time
import json
import os
from datetime import datetime
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_and_save(url, category_dir, doc_id, category):
    """Fetch URL content and save if sufficient."""
    try:
        logger.info(f"Fetching: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
            'Accept': 'text/html,application/xhtml+xml',
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'button', 'form']):
            tag.decompose()
        
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        text = ' '.join(text.split())  # Normalize whitespace
        
        word_count = len(text.split())
        
        if word_count < 100:
            logger.warning(f"Insufficient content ({word_count} words) from {url}")
            return None
        
        # Save
        filename = f"{category}_{doc_id:03d}.txt"
        filepath = os.path.join(category_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.info(f"✓ Saved {filename}: {word_count:,} words")
        
        time.sleep(2)  # Rate limiting
        
        return {
            'doc_id': f"{category}_{doc_id:03d}",
            'category': category,
            'url': url,
            'filepath': filepath,
            'word_count': word_count,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return None


def collect_all_data():
    """Collect comprehensive data from IIT Jodhpur sources."""
    
    output_dir = 'data/raw'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define all URLs to scrape
    data_sources = {
        'academic_regulations': [
            'https://www.iitj.ac.in',
            'https://old.iitj.ac.in/uploaded_docs/UG_Ordinances_Rules.pdf',  # Will try
            'https://www.iitj.ac.in/academics',
        ],
        'departments': [
            'https://www.iitj.ac.in/computer-science-engineering',
            'https://www.iitj.ac.in/electrical-engineering',
            'https://www.iitj.ac.in/mechanical-engineering',
            'https://www.iitj.ac.in/chemistry',
            'https://www.iitj.ac.in/physics',
            'https://www.iitj.ac.in/mathematics',
        ],
        'research': [
            'https://www.iitj.ac.in/research',
            'https://www.iitj.ac.in/centers-excellence',
        ],
        'faculty': [
            'https://www.iitj.ac.in/faculty',
            'https://www.iitj.ac.in/computer-science-engineering/en/people',
        ],
        'courses': [
            'https://www.iitj.ac.in/academics/undergraduate',
            'https://www.iitj.ac.in/academics/postgraduate',
        ],
        'about': [
            'https://www.iitj.ac.in/about',
            'https://www.iitj.ac.in/main/en/introduction',
            'https://www.iitj.ac.in/office-of-students/en/campus-life',
        ]
    }
    
    all_metadata = []
    doc_id = 0
    
    print("="*70)
    print("STARTING COMPREHENSIVE DATA COLLECTION FROM IIT JODHPUR")
    print("="*70)
    
    for category, urls in data_sources.items():
        print(f"\n[{category.upper()}]")
        
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        for url in urls:
            metadata = fetch_and_save(url, category_dir, doc_id, category)
            if metadata:
                all_metadata.append(metadata)
                doc_id += 1
    
    # Save metadata
    metadata_file = os.path.join(output_dir, 'all_metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2)
    
    total_words = sum(m['word_count'] for m in all_metadata)
    
    print("\n" + "="*70)
    print("DATA COLLECTION COMPLETE!")
    print(f"Total documents: {len(all_metadata)}")
    print(f"Total words: {total_words:,}")
    print(f"Average words per document: {total_words//len(all_metadata) if all_metadata else 0:,}")
    print("="*70)
    
    return all_metadata


if __name__ == '__main__':
    collect_all_data()
