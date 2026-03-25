"""
Indian Name Generator using LLM
Dhruva Kumar Kaushal (B22AI017)
NLU Assignment - Problem 2

This module generates a diverse dataset of Indian names using Large Language Models.
Generates names across different:
- Regional origins (Hindi, Tamil, Telugu, Bengali, etc.)
- Genders (Male, Female)
- Lengths and complexities

Includes quality filtering and validation.

Original implementation for assignment purposes.
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter
import re

# Optional: LLM API imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class IndianNameGenerator:
    """
    Generates Indian names using LLM or seed data.
    
    Uses Large Language Models to generate diverse, culturally appropriate
    Indian names across different regions and genders.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the name generator.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # LLM configuration
        self.use_llm = config.get('use_llm', False)
        self.llm_provider = config.get('llm_provider', 'openai')
        self.model_name = config.get('model_name', 'gpt-3.5-turbo')
        self.temperature = config.get('temperature', 0.8)
        
        # Generation parameters
        self.regions = config.get('regions', ['Hindi', 'Tamil', 'Telugu'])
        self.genders = config.get('genders', {'male': 0.5, 'female': 0.5})
        self.min_length = config.get('min_length', 3)
        self.max_length = config.get('max_length', 15)
        
        # Initialize LLM client if available
        self.llm_client = None
        if self.use_llm:
            self._initialize_llm()
        
        # Seed data for fallback
        self.seed_names = self._load_seed_names()
    
    def _initialize_llm(self):
        """Initialize LLM client based on provider."""
        try:
            if self.llm_provider == 'openai' and OPENAI_AVAILABLE:
                # User should set OPENAI_API_KEY in environment
                self.llm_client = openai.OpenAI()
                self.logger.info("Initialized OpenAI client")
            elif self.llm_provider == 'anthropic' and ANTHROPIC_AVAILABLE:
                # User should set ANTHROPIC_API_KEY in environment
                self.llm_client = anthropic.Anthropic()
                self.logger.info("Initialized Anthropic client")
            else:
                self.logger.warning(f"LLM provider '{self.llm_provider}' not available. "
                                  "Using seed data instead.")
                self.use_llm = False
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM: {e}. Using seed data instead.")
            self.use_llm = False
    
    def _load_seed_names(self) -> Dict[str, List[str]]:
        """
        Load seed Indian names for fallback or augmentation.
        
        Returns:
            Dictionary mapping region to list of names
        """
        # Comprehensive seed data of authentic Indian names
        seed_names = {
            'Hindi': {
                'male': ['Aarav', 'Vivaan', 'Aditya', 'Vihaan', 'Arjun', 'Sai', 'Arnav',
                        'Ayaan', 'Krishna', 'Ishaan', 'Shaurya', 'Atharv', 'Advaith',
                        'Pranav', 'Reyansh', 'Kabir', 'Dhruv', 'Aryan', 'Yash', 'Rohan',
                        'Karan', 'Harsh', 'Varun', 'Rahul', 'Amit', 'Ravi', 'Suresh'],
                'female': ['Aadhya', 'Ananya', 'Pari', 'Anika', 'Navya', 'Diya', 'Pihu',
                          'Saanvi', 'Aaradhya', 'Sara', 'Myra', 'Kiara', 'Riya', 'Priya',
                          'Pooja', 'Neha', 'Anjali', 'Kavya', 'Nisha', 'Shreya', 'Isha',
                          'Mahira', 'Zoya', 'Anvi', 'Ira', 'Sia', 'Tara']
            },
            'Tamil': {
                'male': ['Arun', 'Kumar', 'Raj', 'Karthik', 'Surya', 'Vijay', 'Prakash',
                        'Siva', 'Murugan', 'Bharath', 'Dinesh', 'Ganesh', 'Harish',
                        'Manoj', 'Naveen', 'Praveen', 'Ramesh', 'Senthil', 'Vignesh'],
                'female': ['Priya', 'Divya', 'Lakshmi', 'Meera', 'Kavitha', 'Sangeetha',
                          'Bharathi', 'Deepa', 'Geetha', 'Janaki', 'Kamala', 'Nalini',
                          'Padma', 'Radha', 'Saraswathi', 'Uma', 'Vani', 'Yamini']
            },
            'Telugu': {
                'male': ['Venkat', 'Srinivas', 'Prasad', 'Ravi', 'Krishna', 'Ramesh',
                        'Suresh', 'Mahesh', 'Naresh', 'Rajesh', 'Vishnu', 'Balaji',
                        'Chandra', 'Murali', 'Satish', 'Vijay', 'Anil', 'Sunil'],
                'female': ['Sravani', 'Keerthi', 'Lavanya', 'Madhuri', 'Nandini', 'Pallavi',
                          'Radhika', 'Sailaja', 'Tanuja', 'Usha', 'Vasudha', 'Vidya',
                          'Anjali', 'Bhavana', 'Chandana', 'Deepika', 'Hema', 'Jyothi']
            },
            'Bengali': {
                'male': ['Abir', 'Arnab', 'Anirban', 'Biswajit', 'Debashis', 'Gourav',
                        'Indrajit', 'Jayanta', 'Kaushik', 'Mainak', 'Nilanjan', 'Partha',
                        'Rajat', 'Sanjay', 'Subhajit', 'Tapas', 'Uttam'],
                'female': ['Ananya', 'Brishti', 'Chandana', 'Debjani', 'Esha', 'Gargi',
                          'Ishani', 'Jhuma', 'Kakali', 'Lopamudra', 'Madhurima', 'Nandini',
                          'Oishi', 'Paromita', 'Ritika', 'Sanjukta', 'Tanushree', 'Urmila']
            },
            'Punjabi': {
                'male': ['Amarjit', 'Balwinder', 'Charanjit', 'Daljit', 'Gurpreet',
                        'Harjit', 'Jagjit', 'Kuldeep', 'Lakhvir', 'Manpreet', 'Navjot',
                        'Paramjit', 'Ranjit', 'Sarabjit', 'Tajinder', 'Vikram'],
                'female': ['Amrita', 'Bani', 'Channrleen', 'Dilpreet', 'Gurleen', 'Harleen',
                          'Jaspreet', 'Kirandeep', 'Manpreet', 'Navpreet', 'Prabhjot',
                          'Rajveer', 'Simran', 'Taranpreet', 'Vasundha']
            },
            'Marathi': {
                'male': ['Abhijit', 'Bharat', 'Chetan', 'Dnyanesh', 'Ganesh', 'Hemant',
                        'Jayesh', 'Kiran', 'Mangesh', 'Nitin', 'Omkar', 'Pramod',
                        'Rajendra', 'Sachin', 'Tanaji', 'Umesh', 'Vikrant'],
                'female': ['Aditi', 'Bhagyashree', 'Chaitali', 'Dipali', 'Gauri', 'Harshala',
                          'Jyoti', 'Kalpana', 'Manasi', 'Nutan', 'Pratibha', 'Rupali',
                          'Shalmali', 'Tejaswini', 'Urmila', 'Vaishali']
            },
            'Gujarati': {
                'male': ['Ananth', 'Bhavin', 'Chirag', 'Darshan', 'Gaurav', 'Hardik',
                        'Jay', 'Kaushal', 'Milan', 'Neel', 'Parth', 'Ronak', 'Sagar',
                        'Tejas', 'Uday', 'Vishal'],
                'female': ['Aarti', 'Bhavna', 'Chandni', 'Diya', 'Falguni', 'Hiral',
                          'Janvi', 'Komal', 'Meghna', 'Nidhi', 'Palak', 'Ruchi', 'Sejal',
                          'Trisha', 'Urvi', 'Vidhi']
            },
            'Kannada': {
                'male': ['Aditya', 'Bharath', 'Charan', 'Darshan', 'Ganesh', 'Harsha',
                        'Karthik', 'Manoj', 'Naveen', 'Pavan', 'Rakesh', 'Suhas',
                        'Tejas', 'Uday', 'Vinay'],
                'female': ['Anusha', 'Bhavana', 'Chandana', 'Deepika', 'Geetha', 'Harini',
                          'Kavya', 'Lakshmi', 'Madhuri', 'Nanditha', 'Pooja', 'Radhika',
                          'Sahana', 'Tejaswini', 'Varsha']
            },
            'Malayalam': {
                'male': ['Arun', 'Biju', 'Dileep', 'Gopalan', 'Hari', 'Jayakumar',
                        'Krishnan', 'Manoj', 'Naveen', 'Pradeep', 'Rajesh', 'Suresh',
                        'Unni', 'Vinod'],
                'female': ['Aparna', 'Bindu', 'Devika', 'Gowri', 'Jayasree', 'Kavya',
                          'Lakshmi', 'Maya', 'Nisha', 'Priya', 'Radha', 'Sujatha',
                          'Uma', 'Vimala']
            },
            'Odia': {
                'male': ['Akash', 'Bibhu', 'Debasis', 'Ganesh', 'Hrusikesh', 'Jagannath',
                        'Kiran', 'Manoranjan', 'Niranjan', 'Prasanna', 'Rajesh', 'Subash'],
                'female': ['Anita', 'Bandita', 'Chandrika', 'Debasmita', 'Gitanjali',
                          'Jayanti', 'Kamala', 'Lipika', 'Namita', 'Priyanka', 'Sarojini',
                          'Subhadra']
            }
        }
        
        return seed_names
    
    def generate_with_llm(self, n_names: int, region: str, gender: str) -> List[str]:
        """
        Generate names using LLM.
        
        Args:
            n_names: Number of names to generate
            region: Regional origin (e.g., 'Hindi', 'Tamil')
            gender: Gender ('male' or 'female')
            
        Returns:
            List of generated names
        """
        prompt = f"""Generate {n_names} authentic {region} {gender} names.
Requirements:
- Names should be culturally appropriate and realistic
- Each name should be on a new line
- Names should be between {self.min_length} and {self.max_length} characters
- Include both traditional and modern names
- No explanations, just the names

Examples of {region} names: {', '.join(random.sample(self.seed_names[region][gender], min(5, len(self.seed_names[region][gender]))))}

Generate {n_names} more unique names:"""
        
        try:
            if self.llm_provider == 'openai':
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=500
                )
                generated_text = response.choices[0].message.content
            elif self.llm_provider == 'anthropic':
                response = self.llm_client.messages.create(
                    model=self.model_name,
                    max_tokens=500,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                generated_text = response.content[0].text
            else:
                return []
            
            # Extract names from response
            names = [line.strip() for line in generated_text.split('\n') if line.strip()]
            # Remove numbering if present
            names = [re.sub(r'^\d+[\.\)]\s*', '', name) for name in names]
            # Remove empty strings
            names = [name for name in names if name]
            
            return names
            
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            return []
    
    def generate_from_seeds(self, n_names: int, region: str, gender: str) -> List[str]:
        """
        Generate names by varying seed names.
        
        Args:
            n_names: Number of names to generate
            region: Regional origin
            gender: Gender
            
        Returns:
            List of generated names
        """
        seed_list = self.seed_names.get(region, {}).get(gender, [])
        
        if not seed_list:
            return []
        
        # If we need more names than seeds, repeat with variations
        names = []
        while len(names) < n_names:
            # Sample from seeds
            sampled = random.sample(seed_list, min(len(seed_list), n_names - len(names)))
            names.extend(sampled)
        
        return names[:n_names]
    
    def validate_name(self, name: str) -> bool:
        """
        Validate if a name meets quality criteria.
        
        Args:
            name: Name to validate
            
        Returns:
            True if name is valid
        """
        # Remove extra whitespace
        name = name.strip()
        
        # Check length
        if len(name) < self.min_length or len(name) > self.max_length:
            return False
        
        # Check if mostly alphabetic
        alpha_ratio = sum(c.isalpha() for c in name) / len(name)
        if alpha_ratio < 0.8:
            return False
        
        # Check for reasonable structure (starts with capital, etc.)
        if not name[0].isupper():
            return False
        
        return True
    
    def generate_dataset(self, n_total: int, output_path: Path) -> Dict:
        """
        Generate complete dataset of Indian names.
        
        Args:
            n_total: Total number of names to generate
            output_path: Path to save generated names
            
        Returns:
            Dictionary containing generated data and statistics
        """
        self.logger.info(f"Generating {n_total} Indian names...")
        
        # Calculate distribution across regions and genders
        n_per_region = n_total // len(self.regions)
        
        all_names = []
        stats = {
            'total_requested': n_total,
            'by_region': {},
            'by_gender': {'male': 0, 'female': 0}
        }
        
        for region in self.regions:
            self.logger.info(f"Generating names for region: {region}")
            region_names = {'male': [], 'female': []}
            
            for gender, ratio in self.genders.items():
                n_names = int(n_per_region * ratio)
                
                # Generate names
                if self.use_llm:
                    names = self.generate_with_llm(n_names, region, gender)
                    if not names:  # Fallback to seeds if LLM fails
                        names = self.generate_from_seeds(n_names, region, gender)
                else:
                    names = self.generate_from_seeds(n_names, region, gender)
                
                # Validate and filter
                valid_names = [name for name in names if self.validate_name(name)]
                
                # Add to collection
                for name in valid_names:
                    all_names.append({
                        'name': name,
                        'region': region,
                        'gender': gender,
                        'length': len(name)
                    })
                    region_names[gender].append(name)
                
                stats['by_gender'][gender] += len(valid_names)
                
                # Rate limiting for API calls
                if self.use_llm:
                    time.sleep(0.5)
            
            stats['by_region'][region] = {
                'male': len(region_names['male']),
                'female': len(region_names['female']),
                'total': len(region_names['male']) + len(region_names['female'])
            }
        
        stats['total_generated'] = len(all_names)
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_names, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Generated {len(all_names)} names, saved to {output_path}")
        
        # Save statistics
        stats_path = output_path.parent / 'generation_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return {'names': all_names, 'statistics': stats}


def main():
    """Main function for standalone execution."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Generate Indian names dataset')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--n-names', type=int, default=5000,
                       help='Number of names to generate')
    parser.add_argument('--output', type=str, default='data/raw/indian_names.json',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config.get('data_generation', {})
    
    # Create generator
    generator = IndianNameGenerator(data_config)
    
    # Generate dataset
    result = generator.generate_dataset(args.n_names, Path(args.output))
    
    print(f"\nGeneration complete!")
    print(f"Total names: {result['statistics']['total_generated']}")
    print(f"By region: {json.dumps(result['statistics']['by_region'], indent=2)}")
    print(f"By gender: {json.dumps(result['statistics']['by_gender'], indent=2)}")


if __name__ == "__main__":
    main()
