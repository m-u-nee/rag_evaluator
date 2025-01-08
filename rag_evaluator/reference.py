"""
Module for extracting and processing references from RAG model outputs.
Handles reference extraction, citation analysis, and context processing.
"""

import re
from typing import List, Dict, Union, Optional
import pandas as pd

class ReferenceExtractor:
    """Class for extracting and analyzing references from text."""
    
    def __init__(self):
        # Regex patterns for reference extraction
        self.ref_pattern = r'<ref\s+name="([^"]+)">([^<]+)<\/ref>'
        self.source_pattern = r'<\|source_start\|>(.+?)<\|source_end\|>'
        self.source_id_pattern = r'<\|source_id_start\|>(.+?)<\|source_id_end\|>'
    
    def extract_references(self, text: str, generation_id: str) -> List[Dict]:
        """
        Extract references and their context from text.
        
        Args:
            text (str): Input text containing references
            generation_id (str): Identifier for the generation
            
        Returns:
            List[Dict]: List of reference dictionaries containing statement and citation info
        """
        references = []
        
        # Find all references in the text
        for match in re.finditer(self.ref_pattern, text):
            ref_id = match.group(1)
            citation = match.group(2)
            
            # Get the grounding context
            context = self._get_grounding_context(text, match.start())
            
            reference_obj = {
                'generation_id': generation_id,
                'statement': context,
                'statement_size': len(context.split()),
                'reference_id': ref_id,
                'citation': citation,
                'citation_size': len(citation.split()),
            }
            
            references.append(reference_obj)
        
        return references

    def _get_grounding_context(self, text: str, ref_start_pos: int) -> str:
        """
        Extract grounding context before a reference based on specific rules:
        1. Stop at newline (paragraph boundary)
        2. Stop at previous reference
        3. Limit to roughly 3 sentences
        
        Args:
            text (str): Full text
            ref_start_pos (int): Starting position of the reference
            
        Returns:
            str: Extracted grounding context
        """
        # Look backwards from the reference position
        current_pos = ref_start_pos
        
        # Find boundaries
        newline_boundary = text.rfind('\n', 0, current_pos)
        prev_ref_boundary = text.rfind('</ref>', 0, current_pos)
        
        if prev_ref_boundary != -1:
            prev_ref_boundary += len('</ref>')
            
        # Take the later boundary
        effective_boundary = max(newline_boundary, prev_ref_boundary)
        
        if effective_boundary == -1:
            effective_boundary = 0
            
        # Extract context
        context = text[effective_boundary:ref_start_pos].strip()
        
        # Limit to roughly 3 sentences
        sentences = re.split(r'(?<=[.!?])\s+', context)
        if len(sentences) > 3:
            context = ' '.join(sentences[-3:])
            
        return context.strip()

    def extract_source_texts(self, text: str) -> Dict[str, str]:
        """
        Extract source texts and their IDs.
        
        Args:
            text (str): Input text containing source texts
            
        Returns:
            Dict[str, str]: Dictionary mapping source IDs to source texts
        """
        sources = {}
        
        # Extract source blocks
        source_blocks = re.findall(self.source_pattern, text, re.DOTALL)
        
        for block in source_blocks:
            # Extract source ID
            source_id_match = re.search(self.source_id_pattern, block)
            if source_id_match:
                source_id = source_id_match.group(1)
                # Remove source ID tags from text
                source_text = re.sub(self.source_id_pattern, '', block).strip()
                sources[source_id] = source_text
                
        return sources

    def clean_references(self, references: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the reference dataset by removing quotes and cleaning statement prefixes.
        
        Args:
            references (pd.DataFrame): DataFrame with reference data
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        result = references.copy()
        
        # Remove quotes from citations
        result['citation'] = result['citation'].str.replace('"', '', regex=False)
        
        # Clean statement prefixes
        prefixes_pattern = r'^(\. |\)\. |\* |\*\*\) )'
        result['statement'] = result['statement'].str.replace(prefixes_pattern, '', regex=True)
        
        return result

    def analyze_references(self, references: pd.DataFrame, sources: Dict[str, str]) -> pd.DataFrame:
        """
        Analyze references for quality and correctness.
        
        Args:
            references (pd.DataFrame): DataFrame with references
            sources (Dict[str, str]): Dictionary of source texts
            
        Returns:
            pd.DataFrame: DataFrame with added analysis columns
        """
        result = references.copy()
        
        # Add analysis columns
        result['is_valid_id'] = result['reference_id'].isin(sources.keys())
        result['citation_length'] = result['citation'].str.split().str.len()
        result['is_valid_citation'] = result['citation_length'] > 2
        
        # Check for duplicated citations
        result['is_duplicate'] = result.duplicated(subset=['generation_id', 'citation'], keep=False)
        
        return result

def process_rag_output(text: str, generation_id: str) -> Dict:
    """
    Process RAG model output to extract and analyze references.
    
    Args:
        text (str): RAG model output text
        generation_id (str): Identifier for the generation
        
    Returns:
        Dict: Dictionary containing extracted references and analysis
    """
    extractor = ReferenceExtractor()
    
    # Extract references and sources
    references = extractor.extract_references(text, generation_id)
    sources = extractor.extract_source_texts(text)
    
    # Convert to DataFrame for analysis
    ref_df = pd.DataFrame(references)
    ref_df = extractor.clean_references(ref_df)
    ref_df = extractor.analyze_references(ref_df, sources)
    
    return {
        'references': ref_df,
        'sources': sources
    }