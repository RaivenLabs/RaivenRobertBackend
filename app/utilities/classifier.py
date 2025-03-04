# utilities/classifier.py

from rapidfuzz import fuzz, process
from typing import List, Dict
import re
import jellyfish
import time
from datetime import datetime

class Classifier:
    """
    A utility class for matching company names using fuzzy matching and phonetic algorithms.
    Handles various company name formats, abbreviations, and historical names.
    """
    
    def __init__(self):
        # Known abbreviations and common variations
        self.common_abbreviations = {
            'intl': 'international',
            'corp': 'corporation',
            'inc': 'incorporated',
            'mfg': 'manufacturing',
            'bros': 'brothers',
            'tech': 'technology',
            'sys': 'systems',
            'svcs': 'services',
            'eng': 'engineering',
            'elec': 'electric',
            'amt': 'amount',
            'assoc': 'associates',
            'co': 'company',
            'ind': 'industries',
            'llc': 'limited liability company',
            'ltd': 'limited',
            'plc': 'public limited company'
        }

        # Common company prefixes to handle
        self.common_prefixes = {
            'the ',
            'a ',
            'an '
        }

        # Legal relationship indicators
        self.relationship_indicators = {
            'f/k/a',  # formerly known as
            'successor to',
            'successor-in-interest to',
            'successor by merger to',
            'successor-by-merger to',
            'd/b/a',  # doing business as
            'a/k/a',  # also known as
            'n/k/a'   # now known as
        }

        # Words that indicate division relationships
        self.division_indicators = {
            'division of',
            'subsidiary of',
            'unit of',
            'part of',
            'branch of',
            'segment of',
            'department of'
        }

    def normalize_company_name(self, name: str) -> str:
        """
        Normalize a company name by removing common prefixes, suffixes,
        and standardizing format.
        
        Args:
            name (str): Company name to normalize
            
        Returns:
            str: Normalized company name
        """
        if not name:
            return ""
        
        # Convert to lowercase
        name = name.lower()
        
        # Remove leading 'the', 'a', 'an'
        for prefix in self.common_prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
        
        # Handle historical names and relationships
        # Get primary name (before any f/k/a, d/b/a, etc.)
        for indicator in self.relationship_indicators:
            if f" {indicator} " in name:
                name = name.split(f" {indicator} ")[0]
        
        # Remove division relationships
        for indicator in self.division_indicators:
            if f" {indicator} " in name:
                name = name.split(f" {indicator} ")[0]
        
        # Remove common corporate suffixes
        suffixes = [
            ' corporation', ' corp', ' incorporated', ' inc', 
            ' company', ' co', ' limited', ' ltd', ' llc',
            ' plc', ' holdings', ' group', ' international',
            ' industries', ' industry', ' industrial'
        ]
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        
        # Replace ampersand with 'and'
        name = name.replace('&', 'and')
        
        # Remove special characters but preserve spaces
        name = re.sub(r'[^\w\s]', ' ', name)
        
        # Replace multiple spaces with single space
        name = ' '.join(name.split())
        
        return name.strip()

    def calculate_confidence(self, str1: str, str2: str) -> Dict:
        """
        Calculate match confidence between two company names using multiple
        matching strategies.
        
        Args:
            str1 (str): First company name
            str2 (str): Second company name
            
        Returns:
            Dict: Confidence score and detailed match information
        """
        # Normalize both strings
        norm1 = self.normalize_company_name(str1)
        norm2 = self.normalize_company_name(str2)
        
        # Different matching strategies with weights
        token_sort_ratio = fuzz.token_sort_ratio(norm1, norm2)
        token_set_ratio = fuzz.token_set_ratio(norm1, norm2)
        partial_ratio = fuzz.partial_ratio(norm1, norm2)
        
        # Phonetic matching
        soundex_match = jellyfish.soundex(norm1) == jellyfish.soundex(norm2)
        metaphone_match = jellyfish.metaphone(norm1) == jellyfish.metaphone(norm2)
        
        # Check for abbreviation matches
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        abbrev_matches = sum(
            1 for w1 in words1
            for w2 in words2
            if (w1.startswith(w2) and len(w1) > len(w2)) or
               (w2.startswith(w1) and len(w2) > len(w1))
        )

        # Calculate weighted base score
        base_score = (
            0.4 * token_set_ratio +    # Heaviest weight on token set
            0.3 * partial_ratio +      # Medium weight on partial
            0.3 * token_sort_ratio     # Medium weight on sort
        )
        
        # Add bonuses for additional matches
        if soundex_match:
            base_score += 7.5  # Increased from 5
        if metaphone_match:
            base_score += 7.5  # Increased from 5
        
        # Add graduated bonus for abbreviation matches
        if abbrev_matches > 0:
            base_score += min(abbrev_matches * 5, 15)  # Up to 15 point bonus
            
        # Extra bonus for exact matches after normalization
        if norm1 == norm2:
            base_score += 10
            
        # Cap at 100
        final_score = min(base_score, 100)
        
        return {
            'confidence': final_score,
            'match_details': {
                'token_sort_ratio': token_sort_ratio,
                'token_set_ratio': token_set_ratio,
                'partial_ratio': partial_ratio,
                'soundex_match': soundex_match,
                'metaphone_match': metaphone_match,
                'abbreviation_match': abbrev_matches > 0,
                'normalized1': norm1,
                'normalized2': norm2
            }
        }

    def find_matches(self, 
                    defendant: str, 
                    customer_list: List[str]) -> List[Dict]:
        """
        Find all potential matches for a defendant name from a list of customer names.
        Returns all matches with their confidence scores for filtering at the route level.
        
        Args:
            defendant (str): Defendant company name to match
            customer_list (List[str]): List of customer names to match against
            
        Returns:
            List[Dict]: Sorted list of all matches with confidence scores
        """
        matches = []
        match_time = datetime.utcnow()
        
        for customer in customer_list:
            result = self.calculate_confidence(defendant, customer)
            
            matches.append({
                'defendant': defendant,
                'customer': customer,
                'confidence': result['confidence'],
                'match_details': result['match_details'],
                'match_date': match_time
            })
        
        # Sort by confidence descending
        return sorted(matches, key=lambda x: x['confidence'], reverse=True)
