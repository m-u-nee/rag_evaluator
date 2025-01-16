"""
RAG Evaluator Library for assessing RAG system performance metrics.
Provides automated evaluation of retrieval-augmented generation systems.
"""

import numpy as np
import re
import pandas as pd
from tqdm import tqdm
from collections import Counter
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

@dataclass
class RAGMetrics:
    """Container for RAG evaluation metrics"""
    non_hallucinated_citation: float
    valid_quote: float
    valid_identifier: float
    unduplicated_quote: float
    rag_index: float

    def __str__(self):
        return "\n".join([f"{k}: {v:.3f}" for k, v in self.__dict__.items()])

class TextAlignment:
    """Implements text alignment using Smith-Waterman algorithm"""
    def __init__(self, match: int = 2, mismatch: int = -1, gap: int = -1, edit_mark: str = "#"):
        if not (isinstance(match, int) and match > 0):
            raise ValueError("match must be a positive integer")
        if not (isinstance(mismatch, int) and mismatch <= 0):
            raise ValueError("mismatch must be a negative integer or zero")
        if not (isinstance(gap, int) and gap <= 0):
            raise ValueError("gap must be a negative integer or zero")
        if not (isinstance(edit_mark, str) and len(edit_mark) == 1):
            raise ValueError("edit_mark must be a single character")
            
        self.match = match
        self.mismatch = mismatch
        self.gap = gap
        self.edit_mark = edit_mark

    # [TextAlignment methods remain unchanged]
    def tokenize(self, text: str) -> List[str]:
        """Split text into tokens"""
        return re.findall(r'\S+', text)
    
    def mark_chars(self, word: str) -> str:
        """Mark characters for visualization"""
        return self.edit_mark * len(word)

    def align_local(self, text_a: str, text_b: str) -> Dict:
        """Perform local sequence alignment using Smith-Waterman algorithm."""
        a_orig = self.tokenize(text_a)
        b_orig = self.tokenize(text_b)
        a = [word.lower() for word in a_orig]
        b = [word.lower() for word in b_orig]
        
        n_rows = len(b) + 1
        n_cols = len(a) + 1
        matrix = np.zeros((n_rows, n_cols), dtype=np.int32)
        
        max_score = 0
        max_pos = (0, 0)
        
        for i in range(1, n_rows):
            for j in range(1, n_cols):
                if a[j-1] == b[i-1]:
                    diag = matrix[i-1, j-1] + self.match
                else:
                    diag = matrix[i-1, j-1] + self.mismatch
                    
                up = matrix[i-1, j] + self.gap
                left = matrix[i, j-1] + self.gap
                
                matrix[i, j] = max(0, diag, up, left)
                
                if matrix[i, j] > max_score:
                    max_score = matrix[i, j]
                    max_pos = (i, j)
        
        aligned_a = []
        aligned_b = []
        i, j = max_pos
        
        while matrix[i, j] > 0:
            current = matrix[i, j]
            diag = matrix[i-1, j-1]
            up = matrix[i-1, j]
            left = matrix[i, j-1]
            
            if current == up + self.gap:
                aligned_a.append(self.mark_chars(b_orig[i-1]))
                aligned_b.append(b_orig[i-1])
                i -= 1
            elif current == left + self.gap:
                aligned_a.append(a_orig[j-1])
                aligned_b.append(self.mark_chars(a_orig[j-1]))
                j -= 1
            else:
                if a[j-1] == b[i-1]:
                    aligned_a.append(a_orig[j-1])
                    aligned_b.append(b_orig[i-1])
                else:
                    aligned_a.append(self.mark_chars(b_orig[i-1]))
                    aligned_b.append(b_orig[i-1])
                    aligned_a.append(a_orig[j-1])
                    aligned_b.append(self.mark_chars(a_orig[j-1]))
                i -= 1
                j -= 1
        
        return {
            'a_edits': ' '.join(reversed(aligned_a)),
            'b_edits': ' '.join(reversed(aligned_b)),
            'score': max_score
        }
    
    def process_dataframe(self, df: pd.DataFrame, 
                         citation_col: str = 'citation', 
                         source_col: str = 'original_source',
                         batch_size: int = 1000) -> pd.DataFrame:
        """Process DataFrame to calculate alignment scores."""
        def process_row(row):
            result = self.align_local(str(row[citation_col]), str(row[source_col]))
            return result['score']
        
        result_df = df.copy()
        scores = []
        for i in tqdm(range(0, len(df), batch_size), desc="Processing alignments"):
            batch = df.iloc[i:i+batch_size]
            batch_scores = batch.apply(process_row, axis=1)
            scores.extend(batch_scores)
        
        result_df['alignment_score'] = scores
        result_df['normalized_score'] = result_df['alignment_score'] / result_df['citation_size'] / 2
        
        return result_df

class MetricsCalculator:
    """Calculator for RAG metrics"""
    def __init__(self, eval_df: pd.DataFrame,
                 response_col: str = 'generated_response',
                 text_col: str = 'text'):
        self.eval_df = eval_df
        self.response_col = response_col
        self.text_col = text_col
        self.aligner = TextAlignment(match=2, mismatch=-1, gap=-1)
        
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate all RAG metrics"""
        sources_processed = self._process_sources(self.eval_df.copy())
        references_df = self._process_references(self.eval_df, sources_processed)
        
        metrics = {}
        
        # Calculate aligned metrics
        aligned_df = self.aligner.process_dataframe(references_df)
        
        # Non-hallucinated citation (reprinting ratio)
        metrics['non_hallucinated_citation'] = np.mean(aligned_df['normalized_score'])
        
        # Valid quote (3+ words)
        metrics['valid_quote'] = np.mean(aligned_df['citation_size'] >= 3)
        
        # Valid identifier
        metrics['valid_identifier'] = np.mean(
            aligned_df['reference_id'].notna() & 
            aligned_df['reference_id'].str.len() > 0
        )
        
        # Unduplicated quote
        citation_counts = Counter(aligned_df['citation'])
        metrics['unduplicated_quote'] = np.mean([count == 1 for count in citation_counts.values()])
        
        # RAG index
        metrics['rag_index'] = np.mean(list(metrics.values()))
        
        return metrics
    
    def _process_sources(self, sources_df: pd.DataFrame) -> pd.DataFrame:
        """Process source texts from the DataFrame"""
        sources_df['source_text'] = sources_df[self.text_col].apply(extract_sources)
        sources_df = sources_df.explode('source_text')
        sources_df['source_id'] = sources_df['source_text'].apply(extract_source_id)
        sources_df['source_text'] = sources_df['source_text'].apply(clean_source_text)
        
        # Add simple numeric generation_id
        sources_df['generation_id'] = range(1, len(sources_df) + 1)
        sources_df['generation_id'] = sources_df['generation_id'].astype(str)
        
        return sources_df[['generation_id', 'source_text', 'source_id']]
    
    def _process_references(self, eval_df: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
        """Process references from responses"""
        references = []
        for idx, row in eval_df.iterrows():
            try:
                text = row[self.response_col]
                generation_id = str(idx + 1)  # Simple numeric generation_id
                answer = extract_content(text, r'<\|answer_start\|>', r'<\|answer_end\|>')
                refs = extract_references(answer, generation_id, sources)
                references.extend(refs)
            except Exception as e:
                continue
        return pd.DataFrame(references)

class RAGHallucinationEvaluator:
    """A single-line evaluator for RAG systems that calculates various metrics."""
    
    def __init__(self, match: int = 2, mismatch: int = -1, gap: int = -1):
        self.match = match
        self.mismatch = mismatch
        self.gap = gap
        
    def evaluate(self, 
                data: Union[pd.DataFrame, str, List[Dict]], 
                response_col: str = 'generated_response',
                text_col: str = 'text') -> RAGMetrics:
        """
        Evaluate RAG system performance from responses.
        
        Args:
            data: DataFrame containing responses or path to parquet file
            response_col: Column name for generated responses
            text_col: Column name for input text
            
        Returns:
            RAGMetrics object containing calculated metrics
        """
        # Load data if needed
        if isinstance(data, str):
            data = pd.read_parquet(data) 
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        
        # Initialize calculator and compute metrics
        calculator = MetricsCalculator(
            eval_df=data,
            response_col=response_col,
            text_col=text_col
        )
        
        metrics_dict = calculator.calculate_metrics()
        return RAGMetrics(**metrics_dict)

# Helper functions remain unchanged
def extract_content(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    """Extract content between tags"""
    pattern = f"{start_tag}(.*?)(?:{end_tag}|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        return content if content else None
    return None

def extract_references(text: str, generation_id: str, sources: pd.DataFrame) -> List[Dict]:
    """Extract references from text"""
    references = []
    ref_pattern = r'<ref\s+name="([^"]+)">([^<]+)<\/ref>'
    
    for match in re.finditer(ref_pattern, text):
        ref_id = match.group(1)
        citation = match.group(2)
        
        source_mask = (sources["generation_id"] == generation_id) & (sources["source_id"] == ref_id)
        if not sources[source_mask].empty:
            original_source = sources[source_mask]["source_text"].iloc[0]
            
            reference_obj = {
                'generation_id': generation_id,
                'reference_id': ref_id,
                'citation': citation,
                'citation_size': len(citation.split()),
                'original_source': original_source,
            }
            references.append(reference_obj)
            
    return references

def extract_sources(text: str) -> List[str]:
    """Extract sources from text"""
    pattern = r'<\|source_start\|>.+?<\|source_end\|>'
    return re.findall(pattern, text, re.DOTALL)

def extract_source_id(text: str) -> str:
    """Extract source ID from text"""
    id_pattern = r'<\|source_id_start\|>.+?<\|source_id_end\|>'
    source_id = re.search(id_pattern, text)
    if source_id:
        return re.sub(r'<\|.+?\|>', '', source_id.group())
    return ''

def clean_source_text(text: str) -> str:
    """Clean source text by removing tags"""
    text = re.sub(r'<\|source_id_start\|>.+?<\|source_id_end\|>', '', text)
    text = re.sub(r'<\|.+?\|>', '', text)
    return text