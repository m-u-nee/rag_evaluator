"""
Utility functions for RAG evaluation, including text processing, 
data handling, and file operations.
"""

import re
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
import json
import warnings

def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing quotes.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    # Remove extra periods
    text = re.sub(r'\.{2,}', '.', text)
    return text.strip()

def extract_content_between_tags(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    """
    Extract content between specified XML-style tags.
    
    Args:
        text (str): Input text
        start_tag (str): Opening tag
        end_tag (str): Closing tag
        
    Returns:
        Optional[str]: Extracted content or None if not found
    """
    pattern = f"{start_tag}(.*?){end_tag}"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

def parse_generated_components(text: str) -> Dict[str, str]:
    """
    Parse generated text into components based on section markers.
    
    Args:
        text (str): Generated text with section markers
        
    Returns:
        Dict[str, str]: Dictionary of components
    """
    components = {
        'query_analysis': '',
        'answer_analysis': '',
        'language_quality': '',
        'reasoning_quality': ''
    }
    
    # Extract each component using regex
    for component in components.keys():
        marker = component.replace('_', ' ').title()
        pattern = f"### {marker} ###\n(.*?)(?:\n\n###|$)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            components[component] = match.group(1).strip()
            
    return components

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load dataset from various file formats.
    
    Args:
        file_path (str): Path to the dataset file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def save_results(results: Union[pd.DataFrame, Dict], 
                file_path: str,
                format: str = 'parquet') -> None:
    """
    Save evaluation results in specified format.
    
    Args:
        results: Results to save (DataFrame or Dict)
        file_path (str): Path to save results
        format (str): Output format ('parquet', 'json', or 'csv')
    """
    # Convert dict to DataFrame if needed
    if isinstance(results, dict):
        results = pd.DataFrame([results])
        
    if format == 'parquet':
        results.to_parquet(file_path)
    elif format == 'json':
        results.to_json(file_path, orient='records', lines=True)
    elif format == 'csv':
        results.to_csv(file_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {format}")

def split_text_into_chunks(text: str, 
                         chunk_size: int = 1000, 
                         overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks for processing.
    
    Args:
        text (str): Input text
        chunk_size (int): Maximum chunk size in characters
        overlap (int): Overlap size between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        
        # Adjust chunk end to nearest sentence boundary
        if end < text_len:
            # Look for sentence boundary
            boundary = text.rfind('.', start, end)
            if boundary != -1:
                end = boundary + 1
                
        chunk = text[start:end].strip()
        chunks.append(chunk)
        
        # Move start position for next chunk
        start = end - overlap
        
    return chunks

def calculate_chunk_metrics(df: pd.DataFrame, 
                          group_col: str, 
                          value_col: str) -> pd.DataFrame:
    """
    Calculate metrics for grouped chunks of data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        group_col (str): Column to group by
        value_col (str): Column to calculate metrics for
        
    Returns:
        pd.DataFrame: DataFrame with calculated metrics
    """
    metrics = df.groupby(group_col)[value_col].agg([
        'count',
        'mean',
        'std',
        'min',
        'max'
    ]).round(3)
    
    return metrics

def detect_outliers(series: pd.Series, 
                   threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR method.
    
    Args:
        series (pd.Series): Input series
        threshold (float): IQR multiplier for outlier detection
        
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    return (series < (Q1 - threshold * IQR)) | (series > (Q3 + threshold * IQR))

def safe_division(numerator: float, 
                 denominator: float, 
                 default: float = 0.0) -> float:
    """
    Safely divide numbers, handling division by zero.
    
    Args:
        numerator (float): Numerator
        denominator (float): Denominator
        default (float): Default value when denominator is zero
        
    Returns:
        float: Division result or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default

def get_source_statistics(sources: Dict[str, str]) -> Dict[str, float]:
    """
    Calculate statistics about source texts.
    
    Args:
        sources (Dict[str, str]): Dictionary of source texts
        
    Returns:
        Dict[str, float]: Dictionary of statistics
    """
    if not sources:
        return {
            'num_sources': 0,
            'avg_source_length': 0,
            'total_source_words': 0
        }
        
    lengths = [len(text.split()) for text in sources.values()]
    
    return {
        'num_sources': len(sources),
        'avg_source_length': sum(lengths) / len(lengths),
        'total_source_words': sum(lengths)
    }