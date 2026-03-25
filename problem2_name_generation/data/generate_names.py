"""
Indian Name Generator for RNN Training
======================================

This module generates realistic Indian names from various regions and languages
for character-level name generation training.

Author: Dhruva Kumar Kaushal (B22AI017)
Date: March 2026
"""

import random
import os


class IndianNameGenerator:
    """Generate diverse Indian names from various linguistic backgrounds."""
    
    def __init__(self):
        # Common Indian name patterns from different regions
        self.first_names_male = [
            # North Indian (Hindi/Punjabi)
            "Aarav", "Aditya", "Arjun", "Aryan", "Ayaan", "Dhruv", "Ishaan", "Kabir",
            "Karan", "Krishnan", "Lakshman", "Manish", "Mohit", "Nakul", "Nikhil",
            "Pranav", "Rahul", "Raj", "Rohan", "Sahil", "Sanjay", "Shiv", "Varun",
            "Vikram", "Virat", "Yash", "Abhishek", "Ankit", "Deepak", "Gaurav",
            # South Indian (Tamil/Telugu/Kannada)
            "Aakash", "Arun", "Balaji", "Ganesh", "Harish", "Karthik", "Kumar",
            "Manoj", "Murali", "Nagesh", "Prakash", "Praveen", "Raghav", "Rajan",
            "Ramesh", "Sanjith", "Suresh", "Venkat", "Vijay", "Vishnu",
            # Bengali/Eastern
            "Amit", "Anirban", "Anirudh", "Arnab", "Debashis", "Indrajit", "Kaushik",
            "Koushik", "Samir", "Sandeep", "Sourav", "Subhash", "Sumit", "Tapan",
            # Western (Gujarati/Marathi)
            "Ashok", "Bharat", "Chirag", "Darshan", "Hitesh", "Jayesh", "Mahesh",
            "Mukesh", "Paresh", "Piyush", "Rajesh", "Ritesh", "Sunil", "Umesh",
        ]
        
        self.first_names_female = [
            # North Indian
            "Aadhya", "Aanya", "Aditi", "Ananya", "Anjali", "Arya", "Diya", "Ishita",
            "Kavya", "Kiara", "Myra", "Navya", "Nisha", "Pari", "Priya", "Riya",
            "Saanvi", "Sara", "Shreya", "Tanya", "Vanya", "Zara", "Anushka", "Divya",
            # South Indian
            "Amala", "Bhavana", "Deepa", "Geetha", "Janani", "Kamala", "Lakshmi",
            "Meena", "Nandini", "Padma", "Radha", "Revathi", "Sangeetha", "Uma",
            "Vani", "Vidya",
            # Bengali/Eastern
            "Anindita", "Ankita", "Aparajita", "Debjani", "Moumita", "Payal", "Puja",
            "Rina", "Ritu", "Sohini", "Sreeja", "Tanvi",
            # Western
            "Aarti", "Jyoti", "Kiran", "Lata", "Mira", "Pooja", "Rashmi", "Smita",
            "Sneha", "Swati", "Tejal", "Usha", "Vandana",
        ]
        
        self.last_names = [
            # Common across India
            "Agarwal", "Bansal", "Bhatia", "Chopra", "Desai", "Dutta", "Garg", "Goel",
            "Goswami", "Gupta", "Iyer", "Jain", "Joshi", "Kapur", "Kapoor", "Khan",
            "Kumar", "Malhotra", "Mehta", "Menon", "Mishra", "Nair", "Pandey", "Patel",
            "Rao", "Reddy", "Saxena", "Sengupta", "Shah", "Sharma", "Shetty", "Singh",
            "Sinha", "Tiwari", "Trivedi", "Varma", "Verma",
            # Regional variations
            "Mukherjee", "Chatterjee", "Bhattacharya", "Chakraborty", "Ganguly",
            "Kulkarni", "Patil", "Kale", "Joshi", "Deshpande",
            "Krishnan", "Subramanian", "Ramachandran", "Venkatesh", "Narasimhan",
            "Raman", "Srinivasan", "Balakrishnan",
        ]
        
        # Additional name components for variation
        self.prefixes = ["", "Dr", "Mr", "Ms", "Prof"]
        
    def generate_name(self) -> str:
        """Generate a single Indian name."""
        # Randomly choose male or female
        if random.random() < 0.5:
            first = random.choice(self.first_names_male)
        else:
            first = random.choice(self.first_names_female)
        
        last = random.choice(self.last_names)
        
        # Sometimes use middle initial or compound names
        if random.random() < 0.3:
            # Add middle initial
            middle_initial = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            return f"{first} {middle_initial} {last}"
        elif random.random() < 0.2:
            # Compound first name
            if random.random() < 0.5:
                first2 = random.choice(self.first_names_male)
            else:
                first2 = random.choice(self.first_names_female)
            return f"{first} {first2} {last}"
        else:
            return f"{first} {last}"
    
    def generate_names(self, count: int) -> list:
        """Generate multiple unique names."""
        names = set()
        attempts = 0
        max_attempts = count * 3
        
        while len(names) < count and attempts < max_attempts:
            name = self.generate_name()
            names.add(name)
            attempts += 1
        
        return sorted(list(names))
    
    def save_to_file(self, names: list, filepath: str):
        """Save names to file, one per line."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            for name in names:
                f.write(name + '\n')
        print(f"Saved {len(names)} names to {filepath}")


def main():
    """Generate training data for name generation."""
    generator = IndianNameGenerator()
    
    # Generate 1000 unique Indian names
    print("Generating 1000 unique Indian names...")
    names = generator.generate_names(1000)
    
    # Save to TrainingNames.txt
    output_path = 'data/raw/TrainingNames.txt'
    generator.save_to_file(names, output_path)
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"  Total names: {len(names)}")
    print(f"  Avg length: {sum(len(n) for n in names) / len(names):.1f} characters")
    print(f"  Shortest: {min(names, key=len)} ({len(min(names, key=len))} chars)")
    print(f"  Longest: {max(names, key=len)} ({len(max(names, key=len))} chars)")
    print(f"\nSample names:")
    for name in random.sample(names, 10):
        print(f"  {name}")


if __name__ == '__main__':
    main()
