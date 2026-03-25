"""
Web Scraper for IIT Jodhpur Data Collection
============================================

This module implements a comprehensive web scraper to collect textual data
from various IIT Jodhpur website sources including academic regulations,
department pages, faculty profiles, research pages, and course information.

Author: Dhruva Kumar Kaushal (B22AI017)
Date: March 2026

Key Features:
- Respects robots.txt
- Implements rate limiting
- Handles errors gracefully
- Saves metadata for each document
- Supports multiple source types
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import time
import json
import os
from typing import List, Dict, Set
from datetime import datetime
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IITJodhpurScraper:
    """
    Main scraper class for collecting data from IIT Jodhpur website.
    
    This class handles:
    - URL discovery and crawling
    - HTML content extraction
    - Robots.txt compliance
    - Rate limiting
    - Error handling and retries
    - Metadata collection
    """
    
    def __init__(self, base_url: str = "https://www.iitj.ac.in", 
                 rate_limit: float = 2.0, max_retries: int = 3):
        """
        Initialize the scraper.
        
        Args:
            base_url: Base URL of IIT Jodhpur website
            rate_limit: Seconds to wait between requests (respect server)
            max_retries: Maximum number of retry attempts for failed requests
        """
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.visited_urls: Set[str] = set()
        self.session = requests.Session()
        
        # Set headers to identify as legitimate scraper
        self.session.headers.update({
            'User-Agent': 'IIT Jodhpur Academic Research Bot (Student Project)',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        
        # Initialize robots.txt parser
        self.robots_parser = RobotFileParser()
        self.robots_parser.set_url(urljoin(base_url, '/robots.txt'))
        try:
            self.robots_parser.read()
            logger.info("Successfully loaded robots.txt")
        except Exception as e:
            logger.warning(f"Could not load robots.txt: {e}")
    
    def can_fetch(self, url: str) -> bool:
        """
        Check if we're allowed to fetch this URL according to robots.txt.
        
        Args:
            url: URL to check
            
        Returns:
            bool: True if allowed to fetch, False otherwise
        """
        try:
            return self.robots_parser.can_fetch("*", url)
        except:
            # If robots.txt check fails, proceed cautiously
            return True
    
    def fetch_page(self, url: str) -> Dict:
        """
        Fetch a single web page with error handling and retries.
        
        Args:
            url: URL to fetch
            
        Returns:
            dict: Contains 'success', 'content', 'error' keys
        """
        # Check if already visited
        if url in self.visited_urls:
            return {'success': False, 'error': 'Already visited'}
        
        # Check robots.txt
        if not self.can_fetch(url):
            logger.warning(f"Blocked by robots.txt: {url}")
            return {'success': False, 'error': 'Blocked by robots.txt'}
        
        # Attempt to fetch with retries
        for attempt in range(self.max_retries):
            try:
                # Rate limiting - be respectful to the server
                time.sleep(self.rate_limit)
                
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                # Mark as visited
                self.visited_urls.add(url)
                
                logger.info(f"Successfully fetched: {url}")
                return {
                    'success': True,
                    'content': response.text,
                    'url': url,
                    'status_code': response.status_code,
                    'timestamp': datetime.now().isoformat()
                }
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt == self.max_retries - 1:
                    return {'success': False, 'error': str(e), 'url': url}
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return {'success': False, 'error': 'Max retries exceeded', 'url': url}
    
    def extract_text_from_html(self, html_content: str) -> str:
        """
        Extract clean text from HTML content.
        
        This removes:
        - HTML tags
        - Scripts and styles
        - Navigation elements
        - Excessive whitespace
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            str: Extracted clean text
        """
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 
                            'aside', 'iframe', 'noscript']):
            element.decompose()
        
        # Extract text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text
    
    def extract_links(self, html_content: str, base_url: str) -> List[str]:
        """
        Extract all links from HTML content.
        
        Args:
            html_content: Raw HTML content
            base_url: Base URL for resolving relative links
            
        Returns:
            list: List of absolute URLs found in the page
        """
        soup = BeautifulSoup(html_content, 'lxml')
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Convert to absolute URL
            absolute_url = urljoin(base_url, href)
            
            # Only include links from the same domain
            if urlparse(absolute_url).netloc == urlparse(self.base_url).netloc:
                links.append(absolute_url)
        
        return list(set(links))  # Remove duplicates
    
    def scrape_url(self, url: str, output_dir: str, doc_id: str) -> Dict:
        """
        Scrape a single URL and save the content.
        
        Args:
            url: URL to scrape
            output_dir: Directory to save output
            doc_id: Unique document identifier
            
        Returns:
            dict: Metadata about the scraped document
        """
        result = self.fetch_page(url)
        
        if not result['success']:
            return {'success': False, 'url': url, 'error': result.get('error')}
        
        # Extract text
        text = self.extract_text_from_html(result['content'])
        
        # Save text to file
        os.makedirs(output_dir, exist_ok=True)
        text_file = os.path.join(output_dir, f"{doc_id}.txt")
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Create metadata
        metadata = {
            'doc_id': doc_id,
            'url': url,
            'timestamp': result['timestamp'],
            'character_count': len(text),
            'word_count': len(text.split()),
            'file_path': text_file
        }
        
        logger.info(f"Saved document {doc_id}: {metadata['word_count']} words")
        
        return metadata
    
    def scrape_source(self, source_config: Dict, output_dir: str) -> List[Dict]:
        """
        Scrape all pages from a specific source.
        
        Args:
            source_config: Configuration for the source (name, URL, etc.)
            output_dir: Directory to save outputs
            
        Returns:
            list: List of metadata dicts for all scraped documents
        """
        source_name = source_config['name']
        start_url = source_config['url']
        
        logger.info(f"Starting to scrape source: {source_name}")
        
        # Create source-specific directory
        source_dir = os.path.join(output_dir, source_name)
        os.makedirs(source_dir, exist_ok=True)
        
        # Fetch start page
        result = self.fetch_page(start_url)
        if not result['success']:
            logger.error(f"Failed to fetch start page for {source_name}")
            return []
        
        # Extract links from start page
        links = self.extract_links(result['content'], start_url)
        links = [start_url] + links[:20]  # Limit to avoid excessive scraping
        
        # Scrape each link
        metadata_list = []
        for idx, link in enumerate(tqdm(links, desc=f"Scraping {source_name}")):
            doc_id = f"{source_name}_{idx:03d}"
            metadata = self.scrape_url(link, source_dir, doc_id)
            if metadata.get('success', True):
                metadata_list.append(metadata)
        
        # Save metadata for this source
        metadata_file = os.path.join(source_dir, 'metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=2)
        
        logger.info(f"Completed scraping {source_name}: {len(metadata_list)} documents")
        
        return metadata_list


def scrape_academic_regulations(output_dir: str) -> List[Dict]:
    """
    Specialized scraper for academic regulations (mandatory source).
    
    Academic regulations are critical documents that contain formal
    academic terminology, program structures, and institutional policies.
    
    Args:
        output_dir: Directory to save scraped content
        
    Returns:
        list: Metadata for scraped documents
    """
    scraper = IITJodhpurScraper()
    
    # Academic regulations URLs
    urls = [
        "https://www.iitj.ac.in/academics/regulations",
        "https://www.iitj.ac.in/academics/programs",
        "https://www.iitj.ac.in/academics/academic-calendar",
    ]
    
    regulations_dir = os.path.join(output_dir, 'academic_regulations')
    os.makedirs(regulations_dir, exist_ok=True)
    
    metadata_list = []
    for idx, url in enumerate(urls):
        doc_id = f"academic_regulations_{idx:03d}"
        metadata = scraper.scrape_url(url, regulations_dir, doc_id)
        if metadata.get('success', True):
            metadata_list.append(metadata)
    
    # Save metadata
    metadata_file = os.path.join(regulations_dir, 'metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2)
    
    return metadata_list


def main():
    """
    Main function to run the complete scraping pipeline.
    """
    import yaml
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    scraper = IITJodhpurScraper(
        base_url=config['data_collection']['base_url'],
        rate_limit=config['data_collection']['rate_limit_seconds'],
        max_retries=config['data_collection']['max_retries']
    )
    
    output_dir = 'data/raw'
    all_metadata = []
    
    # Scrape each source
    for source in config['data_collection']['sources']:
        metadata = scraper.scrape_source(source, output_dir)
        all_metadata.extend(metadata)
    
    # Save combined metadata
    combined_metadata_file = os.path.join(output_dir, 'all_metadata.json')
    with open(combined_metadata_file, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2)
    
    # Print summary
    total_docs = len(all_metadata)
    total_words = sum(m.get('word_count', 0) for m in all_metadata)
    
    print("\n" + "="*60)
    print("SCRAPING COMPLETE")
    print("="*60)
    print(f"Total documents collected: {total_docs}")
    print(f"Total words collected: {total_words:,}")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
