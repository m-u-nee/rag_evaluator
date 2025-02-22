import pandas as pd
from typing import Union, Optional, Tuple
from pathlib import Path

from .text_processing.chunker import chunk_dataframe
from .text_processing.search import batch_bm25_search
from .text_processing.format import format_search_results

def search_and_format(
    chunked_sources: pd.DataFrame,
    queries: Union[str, pd.DataFrame],
    text_column: str = 'text',
    query_column: str = 'query',
    top_k: int = 5,
    source_separator: str = '\n### Source ###\n',
    random_ids: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform BM25 search and format results using pre-chunked source data.
    
    Args:
        chunked_sources: DataFrame containing pre-chunked source texts
        queries: Path to parquet file or DataFrame containing queries
        text_column: Name of column containing source texts
        query_column: Name of column containing queries
        top_k: Number of top results to return per query
        source_separator: Separator to use between concatenated sources
        random_ids: If True, generate random IDs; if False, use sequential numbers
        
    Returns:
        Tuple of (special_tokens_df, standard_df) containing formatted search results
    """
    # Load queries if parquet path provided
    if isinstance(queries, str):
        queries_df = pd.read_parquet(queries)
    else:
        queries_df = queries.copy()
    
    # Validate required columns exist
    if text_column not in chunked_sources.columns:
        raise ValueError(f"Text column '{text_column}' not found in sources")
    if query_column not in queries_df.columns:
        raise ValueError(f"Query column '{query_column}' not found in queries")
    
    # Perform BM25 search
    search_results = batch_bm25_search(
        queries_df=queries_df,
        sources_df=chunked_sources,
        query_column=query_column,
        text_column=text_column,
        top_k=top_k,
        separator=source_separator
    )
    
    # Format results in both formats
    special_tokens_df, standard_df = format_search_results(
        df=search_results,
        query_column=query_column,
        sources_column='concatenated_sources',
        source_separator=source_separator,
        random_ids=random_ids
    )
    
    return special_tokens_df, standard_df

def process_and_search(
    sources: Union[str, pd.DataFrame],
    queries: Union[str, pd.DataFrame],
    text_column: str = 'text',
    query_column: str = 'query',
    chunk_size: int = 300,
    top_k: int = 5,
    source_separator: str = '\n### Source ###\n',
    chunk_id_column: Optional[str] = 'id',
    chunk_lang_column: Optional[str] = 'lang',
    random_ids: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end pipeline to chunk texts, perform BM25 search, and format results.
    
    Args:
        sources: Path to parquet file or DataFrame containing source texts
        queries: Path to parquet file or DataFrame containing queries
        text_column: Name of column containing source texts
        query_column: Name of column containing queries
        chunk_size: Maximum number of words per chunk
        top_k: Number of top results to return per query
        source_separator: Separator to use between concatenated sources
        chunk_id_column: Name of ID column for chunking (optional)
        chunk_lang_column: Name of language column for chunking (optional)
        random_ids: If True, generate random IDs; if False, use sequential numbers
        
    Returns:
        Tuple of (special_tokens_df, standard_df) containing formatted search results
    """
    # Load source data if parquet path provided
    if isinstance(sources, str):
        sources_df = pd.read_parquet(sources)
    else:
        sources_df = sources.copy()
    
    # Step 1: Chunk the source texts
    chunked_df = chunk_dataframe(
        df=sources_df,
        text_column=text_column,
        id_column=chunk_id_column,
        lang_column=chunk_lang_column,
        max_segment=chunk_size
    )
    
    # Steps 2-3: Search and format using the new function
    return search_and_format(
        chunked_sources=chunked_df,
        queries=queries,
        text_column=text_column,
        query_column=query_column,
        top_k=top_k,
        source_separator=source_separator,
        random_ids=random_ids
    )

def save_results(
    special_tokens_df: pd.DataFrame,
    standard_df: pd.DataFrame,
    output_prefix: Union[str, Path],
    save_format: str = 'parquet'
) -> None:
    """
    Save both result DataFrames to files.
    
    Args:
        special_tokens_df: DataFrame with special tokens format
        standard_df: DataFrame with standard format
        output_prefix: Prefix for output files (will append _special_tokens and _standard)
        save_format: Format to save ('parquet' or 'csv')
    """
    output_prefix = Path(output_prefix)
    special_tokens_path = output_prefix.parent / f"{output_prefix.stem}_special_tokens.{save_format}"
    standard_path = output_prefix.parent / f"{output_prefix.stem}_standard.{save_format}"
    
    if save_format.lower() == 'parquet':
        special_tokens_df.to_parquet(special_tokens_path, index=False)
        standard_df.to_parquet(standard_path, index=False)
    elif save_format.lower() == 'csv':
        special_tokens_df.to_csv(special_tokens_path, index=False)
        standard_df.to_csv(standard_path, index=False)
    else:
        raise ValueError("save_format must be either 'parquet' or 'csv'")

# Example usage
if __name__ == "__main__":
    # Example 1: Full pipeline with chunking (using random IDs by default)
    special_tokens_results, standard_results = process_and_search(
        sources='/path/to/shakespeare.parquet',
        queries='/path/to/queries.parquet',
        text_column='text',
        query_column='query',
    )

    # Example 2: Using pre-chunked data with sequential IDs
    chunked_df = pd.read_parquet('/path/to/chunked_data.parquet')
    special_tokens_results, standard_results = search_and_format(
        chunked_sources=chunked_df,
        queries='/path/to/queries.parquet',
        text_column='text',
        query_column='query',
        random_ids=False  # Use sequential IDs instead of random
    )

    # Save results
    save_results(
        special_tokens_results,
        standard_results,
        'output/search_results',
        save_format='parquet'
    )