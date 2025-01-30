import pandas as pd
import random
import string

def generate_random_id(length: int = 8) -> str:
    """Generate a random sequence of letters."""
    return ''.join(random.choices(string.ascii_letters, k=length))

def format_search_results(df: pd.DataFrame, 
                         query_column: str = 'query',
                         sources_column: str = 'concatenated_sources',
                         source_separator: str = '\n### Source ###\n') -> pd.DataFrame:
    """
    Format search results with special tags and random source IDs.
    
    Args:
        df: DataFrame containing queries and concatenated sources
        query_column: Name of the column containing queries
        sources_column: Name of the column containing concatenated sources
        source_separator: Separator used between sources in the sources column
        
    Returns:
        DataFrame with formatted text column
    """
    def format_single_row(query: str, sources: str) -> str:
        # Split sources into individual texts
        source_texts = [s.strip() for s in sources.split(source_separator) if s.strip()]
        
        # Format query
        formatted_text = f"<|query_start|>{query}<|query_end|>"
        
        # Format each source with a random ID
        for source in source_texts:
            source_id = generate_random_id()
            formatted_text += f"<|source_start|><|source_id_start|>{source_id}<|source_id_end|>{source}<|source_end|>"
        
        # Add source analysis tag
        formatted_text += "<|source_analysis_start|>"
        
        return formatted_text
    
    # Create output DataFrame
    output_df = df.copy()
    
    # Add formatted text column
    output_df['formatted_text'] = output_df.apply(
        lambda row: format_single_row(row[query_column], row[sources_column]), 
        axis=1
    )
    
    return output_df