import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List
import re
from nltk.tokenize import word_tokenize
import nltk

def ensure_nltk_data():
    """Ensure all required NLTK data is downloaded."""
    required_packages = ['punkt']
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            print(f"Downloading required NLTK data: {package}")
            nltk.download(package, quiet=True)

def batch_bm25_search(queries_df: pd.DataFrame,
                     sources_df: pd.DataFrame,
                     query_column: str = 'query',
                     text_column: str = 'text',
                     top_k: int = 5,
                     separator: str = '\n### Source ###\n') -> pd.DataFrame:
    """
    Perform BM25 search for each query against source texts and concatenate results.
    
    Args:
        queries_df: DataFrame containing queries
        sources_df: DataFrame containing source texts
        query_column: Name of column containing queries
        text_column: Name of column containing source texts
        top_k: Number of top results to return per query
        separator: Separator to use between concatenated source texts
        
    Returns:
        DataFrame with original queries and concatenated search results
    """
    # Ensure NLTK data is available
    ensure_nltk_data()
    
    def preprocess_text(text: str) -> str:
        """Preprocess text by lowercasing and removing extra whitespace."""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    # Preprocess and tokenize source texts
    tokenized_corpus = []
    for text in sources_df[text_column]:
        processed_text = preprocess_text(text)
        tokens = word_tokenize(processed_text)
        tokenized_corpus.append(tokens)
    
    # Initialize BM25
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Function to search and concatenate results for a single query
    def search_single_query(query: str) -> str:
        # Preprocess and tokenize query
        processed_query = preprocess_text(query)
        tokenized_query = word_tokenize(processed_query)
        
        # Get BM25 scores
        scores = bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Get top k texts
        top_texts = sources_df.iloc[top_indices][text_column].tolist()
        
        # Concatenate results with separator
        return separator.join(top_texts)
    
    # Process all queries
    results = []
    for query in queries_df[query_column]:
        concatenated_sources = search_single_query(query)
        results.append(concatenated_sources)
    
    # Create output DataFrame
    output_df = queries_df.copy()
    output_df['concatenated_sources'] = results
    
    return output_df