"""
Enhanced Web Scraper for IIT Jodhpur Data Collection
====================================================

This module implements an improved web scraper to collect comprehensive textual data
from IIT Jodhpur website sources.

Author: Dhruva Kumar Kaushal (B22AI017)
Date: March 2026
"""

import requests
from bs4 import BeautifulSoup
import time
import json
import os
from typing import List, Dict
from datetime import datetime
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedIITJodhpurScraper:
    """Enhanced scraper with better content extraction."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        
    def extract_main_content(self, html: str) -> str:
        """Extract main textual content from HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 
                            'aside', 'iframe', 'noscript', 'button', 'form']):
            element.decompose()
        
        # Try to find main content area
        main_content = None
        for selector in ['main', 'article', '[role="main"]', '.content', '#content', '.main-content']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.body if soup.body else soup
        
        # Extract text
        text = main_content.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = ' '.join(lines)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def fetch_url(self, url: str) -> str:
        """Fetch content from URL."""
        try:
            logger.info(f"Fetching: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            time.sleep(2)  # Rate limiting
            return response.text
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return ""
    
    def scrape_sources(self, output_dir: str) -> List[Dict]:
        """Scrape all configured sources."""
        
        # Define comprehensive source URLs
        sources = {
            'academic_regulations': [
                'https://www.iitj.ac.in/academics/regulations',
                'https://www.iitj.ac.in/academics/programs',
                'https://www.iitj.ac.in/academics/academic-calendar',
                'https://www.iitj.ac.in/academics',
            ],
            'departments': [
                'https://www.iitj.ac.in/department/index.php?id=1',  # CSE
                'https://www.iitj.ac.in/department/index.php?id=2',  # EE
                'https://www.iitj.ac.in/department/index.php?id=3',  # ME
                'https://www.iitj.ac.in/department/index.php?id=5',  # Chemistry
                'https://www.iitj.ac.in/department/index.php?id=6',  # Physics
                'https://www.iitj.ac.in/department/index.php?id=7',  # Mathematics
                'https://www.iitj.ac.in/departments',
            ],
            'research': [
                'https://www.iitj.ac.in/research',
                'https://www.iitj.ac.in/research/publications',
                'https://www.iitj.ac.in/research/centers',
                'https://www.iitj.ac.in/research/projects',
            ],
            'faculty': [
                'https://www.iitj.ac.in/faculty',
            ],
            'courses': [
                'https://www.iitj.ac.in/academics/courses',
                'https://www.iitj.ac.in/academics/undergraduate',
                'https://www.iitj.ac.in/academics/postgraduate',
            ],
            'about': [
                'https://www.iitj.ac.in/about',
                'https://www.iitj.ac.in/about/vision-mission',
                'https://www.iitj.ac.in/about/campus',
            ],
        }
        
        all_metadata = []
        doc_id = 0
        
        for category, urls in sources.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Scraping category: {category}")
            logger.info(f"{'='*60}")
            
            category_dir = os.path.join(output_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            
            for url in tqdm(urls, desc=f"Processing {category}"):
                html = self.fetch_url(url)
                if not html:
                    continue
                
                text = self.extract_main_content(html)
                
                # Only save if we got meaningful content
                if len(text.split()) < 50:
                    logger.warning(f"Skipping {url} - insufficient content")
                    continue
                
                # Save to file
                filename = f"{category}_{doc_id:03d}.txt"
                filepath = os.path.join(category_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                word_count = len(text.split())
                logger.info(f"Saved {filename}: {word_count} words")
                
                all_metadata.append({
                    'doc_id': f"{category}_{doc_id:03d}",
                    'category': category,
                    'url': url,
                    'file_path': filepath,
                    'word_count': word_count,
                    'timestamp': datetime.now().isoformat()
                })
                
                doc_id += 1
        
        # Save metadata
        metadata_file = os.path.join(output_dir, 'all_metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, indent=2)
        
        total_words = sum(m['word_count'] for m in all_metadata)
        logger.info(f"\n{'='*60}")
        logger.info(f"Scraping complete!")
        logger.info(f"Total documents: {len(all_metadata)}")
        logger.info(f"Total words: {total_words:,}")
        logger.info(f"{'='*60}")
        
        return all_metadata


def main():
    """Run the enhanced scraper."""
    scraper = EnhancedIITJodhpurScraper()
    output_dir = 'data/raw'
    os.makedirs(output_dir, exist_ok=True)
    scraper.scrape_sources(output_dir)


if __name__ == '__main__':
    main()
