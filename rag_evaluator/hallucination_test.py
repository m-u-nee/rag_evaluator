



"""
RAG Evaluator Library for assessing RAG system performance metrics.
Provides automated evaluation of retrieval-augmented generation systems.
"""

import numpy as np
import re
import pandas as pd
from tqdm import tqdm
from collections import Counter
from typing import Dict, List, Optional, Union, Literal
from dataclasses import dataclass

ModelType = Literal['pleias', 'other']

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

def extract_pleias_sources(text: str) -> List[Dict[str, str]]:
    """Extract sources from Pleias format texts"""
    # Extract full source blocks
    pattern = r'<\|source_start\|>.+?<\|source_end\|>'
    sources = re.findall(pattern, text, re.DOTALL)
    
    processed_sources = []
    for source in sources:
        # Extract source ID
        id_pattern = r'<\|source_id_start\|>.+?<\|source_id_end\|>'
        source_id_match = re.search(id_pattern, source)
        source_id = ''
        if source_id_match:
            source_id = re.sub(r'<\|.+?\|>', '', source_id_match.group())
            
        # Clean source text
        clean_text = source
        # Remove source ID block first
        clean_text = re.sub(r'<\|source_id_start\|>.+?<\|source_id_end\|>', '', clean_text)
        # Remove remaining tags
        clean_text = re.sub(r'<\|.+?\|>', '', clean_text)
        
        processed_sources.append({
            'source_id': source_id,
            'source_text': clean_text.strip()
        })
        
    return processed_sources

def extract_other_sources(text: str) -> List[Dict[str, str]]:
    """Extract sources from non-Pleias format texts"""
    content_pattern = r'^.+?You can take information from the following texts:\n(.+?)\n\nFinally, your answer'
    match = re.search(content_pattern, text, re.DOTALL)
    if not match:
        return []
    
    content = match.group(1)
    # Split sources but don't filter yet
    sources = content.split('\n**')[1:]  # Skip first empty split
    
    processed_sources = []
    for source in sources:
        try:
            # Extract source ID without **
            source_id = source[:source.find('**')].strip()
            # Get everything after **
            source_text = source[source.find('**')+2:].strip()
            
            if source_id and source_text:
                processed_sources.append({
                    'source_id': source_id,
                    'source_text': source_text
                })
        except Exception as e:
            print(f"Error processing source: {str(e)}")
            continue
            
    return processed_sources

def extract_references(text: str, generation_id: str, sources: pd.DataFrame) -> List[Dict]:
    """Extract references from both XML tags and numbered formats"""
    references = []
    
    # Handle XML ref tags first
    ref_pattern = r'<ref\s+name=["\'](.*?)[\'"]\s*>(.*?)</ref>'
    xml_matches = list(re.finditer(ref_pattern, text, re.DOTALL))
    
    if not xml_matches:
        # Improved numbered citation pattern
        numbered_pattern = r'\d+\.\s+\*\*([^*]+?)\*\*\s*(.*?)(?=\d+\.\s+\*\*|$)'
        numbered_matches = re.finditer(numbered_pattern, text, re.DOTALL)
        
        for match in numbered_matches:
            try:
                ref_id = match.group(1).strip()
                # Extract actual citation from the text block
                full_text = match.group(2).strip()
                
                # Remove metadata lines and extract citation
                lines = full_text.split('\n')
                citation_text = '\n'.join(line for line in lines if line.strip() and not line.strip().startswith('(') and not line.strip().startswith('Objet:'))
                
                if not citation_text:
                    continue
                    
                source_mask = (sources["generation_id"] == generation_id) & (sources["source_id"] == ref_id)
                
                reference_obj = {
                    'generation_id': generation_id,
                    'reference_id': ref_id,
                    'citation': citation_text.strip(),
                    'citation_size': len(citation_text.strip().split()),
                    'original_source': sources[source_mask]["source_text"].iloc[0] if not sources[source_mask].empty else ""
                }
                references.append(reference_obj)
                
            except Exception as e:
                print(f"Error processing numbered reference in generation {generation_id}: {str(e)}")
                continue
    else:
        # Process XML refs
        for match in xml_matches:
            try:
                ref_id = match.group(1).strip()
                citation = match.group(2).strip()
                
                if not citation:
                    continue
                    
                source_mask = (sources["generation_id"] == generation_id) & (sources["source_id"] == ref_id)
                
                reference_obj = {
                    'generation_id': generation_id,
                    'reference_id': ref_id,
                    'citation': citation,
                    'citation_size': len(citation.split()),
                    'original_source': sources[source_mask]["source_text"].iloc[0] if not sources[source_mask].empty else ""
                }
                references.append(reference_obj)
                
            except Exception as e:
                print(f"Error processing XML reference in generation {generation_id}: {str(e)}")
                continue
    
    return references

class MetricsCalculator:
    """Calculator for RAG metrics"""
    def __init__(self, eval_df: pd.DataFrame,
                 response_col: str = 'generated_response',
                 text_col: str = 'text',
                 model_type: ModelType = 'other'):
        self.eval_df = eval_df
        self.response_col = response_col
        self.text_col = text_col
        self.model_type = model_type
        self.aligner = TextAlignment(match=2, mismatch=-1, gap=-1)
        
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate all RAG metrics"""
        sources_processed = self._process_sources(self.eval_df.copy())
        references_df = self._process_references(self.eval_df, sources_processed)
        
        metrics = {}
        
        # First get total reference count and add citation_id
        references_df['citation_id'] = range(1, len(references_df) + 1)
        total_citations = len(references_df)
        
        # Find correct citations (references that match source IDs)
        correct_citations = (references_df[['generation_id', 'reference_id', 'citation_id']]
            .merge(
                sources_processed[['generation_id', 'source_id']], 
                left_on=['generation_id', 'reference_id'],
                right_on=['generation_id', 'source_id']
            )
        )
        
        # Count hallucinated IDs (those that don't match sources)
        hallucinated_ids = total_citations - len(correct_citations)
        
        # Calculate valid_identifier as 1 - (hallucinated/total)
        metrics['valid_identifier'] = 1 - (hallucinated_ids / total_citations) if total_citations > 0 else 0
        
        print("\nValid Identifier Calculation:")
        print(f"Total citations: {total_citations}")
        print(f"Correct citations: {len(correct_citations)}")
        print(f"Hallucinated IDs: {hallucinated_ids}")
        print(f"Valid identifier score: {metrics['valid_identifier']:.3f}")
        
        # Calculate aligned metrics
        aligned_df = self.aligner.process_dataframe(references_df)
        
        # Non-hallucinated citation (reprinting ratio)
        metrics['non_hallucinated_citation'] = np.mean(aligned_df['normalized_score'])
        
        # Valid quote (3+ words)
        metrics['valid_quote'] = np.mean(aligned_df['citation_size'] >= 3)
        
        # Calculate duplicated quotes R-style
        duplicated_quotes = self._calculate_duplicated_quotes_r_style(aligned_df)
        
        # Calculate unduplicated quote ratio like in R
        metrics['unduplicated_quote'] = 1 - (duplicated_quotes / total_citations) if total_citations > 0 else 0
        
        # RAG index
        metrics['rag_index'] = np.mean(list(metrics.values()))
        
        return metrics
    
    def _calculate_duplicated_quotes_r_style(self, df: pd.DataFrame) -> int:
        """
        Calculate duplicated quotes using R-style approach:
        - Only count duplicates within same generation
        - Each duplicate pair is counted once
        """
        duplicates = 0
        for generation_id in df['generation_id'].unique():
            gen_df = df[df['generation_id'] == generation_id]
            
            # Compare each citation with others in same generation
            for idx, row in gen_df.iterrows():
                matches = gen_df[
                    (gen_df.index > idx) &  # Only look at subsequent rows to avoid double counting
                    (gen_df['citation'] == row['citation']) &  # Same citation text
                    (gen_df['citation_id'] != row['citation_id'])  # Different citation IDs
                ]
                duplicates += len(matches)
        
        return duplicates
    
    def _process_sources(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        """Process source texts from the DataFrame using specified model type"""
        all_sources = []
        
        # Process each row
        for idx, row in eval_df.iterrows():
            generation_id = str(idx + 1)
            
            # Extract sources based on model type
            if self.model_type == 'pleias':
                sources = extract_pleias_sources(row[self.text_col])
            else:
                sources = extract_other_sources(row[self.text_col])
            
            # Add to results
            for source in sources:
                all_sources.append({
                    'generation_id': generation_id,
                    'source_id': source['source_id'],
                    'source_text': source['source_text'],
                })
        
        # Convert to DataFrame
        sources_df = pd.DataFrame(all_sources)
        return sources_df[['generation_id', 'source_text', 'source_id']]
    
    def _process_references(self, eval_df: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
        """Process references from responses"""
        references = []
        total_rows = len(eval_df)
        rows_with_refs = 0
        print(f"\nProcessing references from {total_rows} rows")
        
        for idx, row in eval_df.iterrows():
            try:
                text = row[self.response_col]
                generation_id = str(idx + 1)  # Simple numeric generation_id
                answer = extract_content(text, r'<\|answer_start\|>', r'<\|answer_end\|>')
                if answer:
                    refs = extract_references(answer, generation_id, sources)
                    if refs:
                        rows_with_refs += 1
                        references.extend(refs)
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                continue
        
        print(f"\nReference extraction summary:")
        print(f"Total rows processed: {total_rows}")
        print(f"Rows with references: {rows_with_refs}")
        print(f"Total references found: {len(references)}")
        
        return pd.DataFrame(references)

class RAGHallucinationEvaluator:
    """A single-line evaluator for RAG systems that calculates various metrics."""
    
    def __init__(self, model_type: ModelType = 'other', match: int = 2, mismatch: int = -1, gap: int = -1):
        self.model_type = model_type
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
        
        print("\nProcessing Input Data:")
        print(f"Total rows: {len(data)}")
        print(f"Columns: {data.columns.tolist()}")
        
        # Initialize calculator and compute metrics
        calculator = MetricsCalculator(
            eval_df=data,
            response_col=response_col,
            text_col=text_col,
            model_type=self.model_type
        )
        
        metrics_dict = calculator.calculate_metrics()
        return RAGMetrics(**metrics_dict)




def extract_content(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    """Extract content between tags with more robust pattern matching"""
    # Make pattern more flexible
    pattern = f"{start_tag}(.*?)(?:{end_tag}|$)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        return content if content else None
    return None







def debug_evaluation(data_path: str, num_rows: int = 10):  # Process first 10 rows by default
    """Debug RAG evaluation with detailed reference matching information"""
    # Load data
    df = pd.read_parquet(data_path).head(num_rows)
    print(f"\n=== Processing first {num_rows} rows ===")
    
    rows_without_refs = []
    rows_with_no_matching_sources = []
    all_refs = []
    
    for idx, row in df.iterrows():
        print(f"\nProcessing row {idx + 1}")
        
        # Extract sources
        sources = extract_other_sources(row['text'])
        sources_df = pd.DataFrame(sources)
        if not sources_df.empty:
            sources_df['generation_id'] = str(idx + 1)
        
        # Extract references
        refs = extract_references(row['generated_response'], str(idx + 1), sources_df)
        
        if not refs:
            rows_without_refs.append(idx + 1)
            print(f"Row {idx + 1}: No references found")
            continue
            
        # Check source matches
        matching_sources = 0
        non_matching_sources = 0
        
        for ref in refs:
            if ref['original_source']:
                matching_sources += 1
            else:
                non_matching_sources += 1
                
        all_refs.extend(refs)
        
        print(f"Row {idx + 1}: Found {len(refs)} references")
        print(f"  - Matching sources: {matching_sources}")
        print(f"  - Non-matching sources: {non_matching_sources}")
        
        if matching_sources == 0:
            rows_with_no_matching_sources.append(idx + 1)
    
    # Summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total rows processed: {num_rows}")
    print(f"Rows with no references: {len(rows_without_refs)}")
    if rows_without_refs:
        print(f"  Row numbers: {rows_without_refs}")
    
    print(f"\nRows with references but no matching sources: {len(rows_with_no_matching_sources)}")
    if rows_with_no_matching_sources:
        print(f"  Row numbers: {rows_with_no_matching_sources}")
    
    total_refs = len(all_refs)
    matching_refs = sum(1 for ref in all_refs if ref['original_source'])
    print(f"\nTotal references found: {total_refs}")
    print(f"References with matching sources: {matching_refs}")
    print(f"References without matching sources: {total_refs - matching_refs}")
    
    # Sample problematic cases
    if rows_with_no_matching_sources:
        problem_row = rows_with_no_matching_sources[0]
        print(f"\n=== Sample Problem Case (Row {problem_row}) ===")
        row_data = df.iloc[problem_row - 1]
        
        sources = extract_other_sources(row_data['text'])
        print("\nExtracted Sources:")
        for i, source in enumerate(sources, 1):
            print(f"\nSource {i}:")
            print(f"ID: {source['source_id']}")
            print(f"Text preview: {source['source_text'][:100]}...")
        
        refs = extract_references(row_data['generated_response'], str(problem_row), pd.DataFrame(sources))
        print("\nExtracted References:")
        for i, ref in enumerate(refs, 1):
            print(f"\nReference {i}:")
            print(f"ID: {ref['reference_id']}")
            print(f"Citation: {ref['citation'][:100]}...")

if __name__ == "__main__":
    debug_evaluation("/Users/mattia/Downloads/generations.parquet", num_rows=10)






# def debug_evaluation(data_path: str):
#     # Load first row only
#     df = pd.read_parquet(data_path).iloc[[0]]
#     print("\n=== Input Data ===")
#     print(f"Text preview: {df['text'].iloc[0]}...")
#     print(f"Response preview: {df['generated_response'].iloc[0]}...")
    
#     print("\n=== Source Extraction ===")
#     # Extract sources
#     sources = extract_other_sources(df['text'].iloc[0])
#     print(f"Found {len(sources)} sources")
#     for i, source in enumerate(sources, 1):
#         print(f"\nSource {i}:")
#         print(f"ID: {source['source_id']}")
#         print(f"Text preview: {source['source_text'][:100]}...")

#     # Convert to DataFrame for reference extraction
#     sources_df = pd.DataFrame(sources)
#     if not sources_df.empty:
#         sources_df['generation_id'] = '1'
    
#     print("\n=== Reference Extraction ===")
#     # Extract numbered references
#     refs = extract_references(df['generated_response'].iloc[0], '1', sources_df)
#     print(f"Found {len(refs)} references")
    
#     for i, ref in enumerate(refs, 1):
#         print(f"\nReference {i}:")
#         print(f"ID: {ref['reference_id']}")
#         print(f"Citation: {ref['citation'][:100]}...")
#         print(f"Size: {ref['citation_size']} words")
#         print(f"Has matching source: {'Yes' if ref['original_source'] else 'No'}")

#     # Debug raw matches
#     response = df['generated_response'].iloc[0]
    
#     # Test XML pattern
#     xml_pattern = r'<ref\s+name=["\'](.*?)[\'"]\s*>(.*?)</ref>'
#     xml_matches = list(re.finditer(xml_pattern, response, re.DOTALL))
#     print(f"\nRaw XML matches: {len(xml_matches)}")
    
#     # Test numbered pattern
#     numbered_pattern = r'\d+\.\s+\*\*([^*]+)\*\*([^1-9]+)'
#     numbered_matches = list(re.finditer(numbered_pattern, response, re.DOTALL))
#     print(f"\nRaw numbered matches: {len(numbered_matches)}")
    
#     print("\n=== Raw Content ===")
#     print("Generated response:")
#     print("---")
#     print(response)
#     print("---")

# if __name__ == "__main__":
#     debug_evaluation("/Users/mattia/Downloads/generations.parquet")