import pandas as pd
import random
import string
from typing import Tuple

def generate_random_id(length: int = 8) -> str:
    """Generate a random sequence of letters."""
    return ''.join(random.choices(string.ascii_letters, k=length))

def format_search_results(df: pd.DataFrame, 
                         query_column: str = 'query',
                         sources_column: str = 'concatenated_sources',
                         source_separator: str = '\n### Source ###\n',
                         random_ids: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Format search results with special tags and configurable source IDs in two different formats.
    
    Args:
        df: DataFrame containing queries and concatenated sources
        query_column: Name of the column containing queries
        sources_column: Name of the column containing concatenated sources
        source_separator: Separator used between sources in the sources column
        random_ids: If True, generate random IDs; if False, use sequential numbers
        
    Returns:
        Tuple of (special_tokens_df, standard_df)
    """
    def format_special_tokens(query: str, sources: str) -> str:
        """Format with special tokens format."""
        source_texts = [s.strip() for s in sources.split(source_separator) if s.strip()]
        
        formatted_text = f"<|query_start|>{query}<|query_end|>"
        
        for idx, source in enumerate(source_texts, 1):
            source_id = generate_random_id() if random_ids else str(idx)
            formatted_text += f"<|source_start|><|source_id_start|>{source_id}<|source_id_end|>{source}<|source_end|>"
        
        formatted_text += "<|source_analysis_start|>"
        
        return formatted_text
    
    def format_standard(query: str, sources: str) -> str:
        """Format with standard prompt format."""
        source_texts = [s.strip() for s in sources.split(source_separator) if s.strip()]
        
        prompt = """Write a sourced summary to a query based on a series of referenced text. """
        prompt += f"""You have received this question posted by a user: {query} """
        prompt += """You answer should rely on verifiable information provided by the texts below and provide many references and footnote using the following tags: <ref name=\"[name of reference]\">\"quote from the reference\"</ref>. """
        prompt += """Ideally you should use references and short quotes for every statement provided (so potentially 5-10 depending on the number of relevant facts). """
        prompt += """The name of the reference is given before each reference between ** and **. You should absolutely use the full name (so complete bibliographic reference with journals, pages, etc. or complete url if needs be). """
        prompt += """Quotes should be exact quotes from the text and you should be very careful to not include any invented text. """
        prompt += """You should use multiple references from different texts to increase the overall verifiability. You can take information from the following texts:"""
        
        # Add sources with IDs
        for idx, source_text in enumerate(source_texts, 1):
            source_id = generate_random_id() if random_ids else str(idx)
            prompt += f"\n**{source_id}**\n{source_text}"
        
        prompt += """ Finally, your answer should be written approximately in the style of this other text: """
        prompt += """Models are a requirement for dynamic filtering of dynamics. We use the fact that the AFM itself is a fast, precise sensor that is already in place to measure the dynamics of our system without additional hardware modifications. We can record the transfer functions of both Z actuators by exciting the respective actuator and measuring the deflection of a cantilever in contact mode. Burns et al. have presented a method that enables to also measure lateral dynamics directly with the cantilever."""
        
        prompt += """ You should not copy this text or use its content but take into account its phrasing/style/structure for your own answer. Overall please completely avoid common verbal tics (like \"in conclusion\", \"in brief\") and keep a grounded descriptive answer. Your output should be structured like this:"""
        
        prompt += """
### Language of the query ###
[Very short identification of the language of the query: is it French, English, German, etc.]

### Pre-analysis ###
[A free text in the original language of the query where you analyze and rephrase the query of the the user. You also analyze briefly the references that have been submitted, checking the most appropriate for an answer based on the information and the contextual clues — some form of source criticism. Also check if the references are duplicate: in this case you can only quote the first one.]

### Referenced answer ###
[Your long answer with many references (<ref>) to the sourced texts in the original language of the query. You should quote every single reference you have singled out in the pre-analysis]"""
        
        return prompt
    
    # Create output DataFrames
    special_tokens_df = df.copy()
    standard_df = df.copy()
    
    # Add formatted columns
    special_tokens_df['formatted_text'] = special_tokens_df.apply(
        lambda row: format_special_tokens(row[query_column], row[sources_column]), 
        axis=1
    )
    
    standard_df['formatted_text'] = standard_df.apply(
        lambda row: format_standard(row[query_column], row[sources_column]), 
        axis=1
    )
    
    return special_tokens_df, standard_df