import pandas as pd
import re
from typing import List, Optional

def process_text(text: str, max_segment: int = 300) -> List[str]:
    """Hierarchically splits text into chunks based on specified delimiters and word counts."""
    def main_split(text: str, max_segment: int) -> List[str]:
        segments = text.split('.\n')
        reconciled = []
        current_segment = ""
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            combined = f"{current_segment} {segment}" if current_segment else segment
            if len(combined.split()) < max_segment:
                current_segment = f"{combined}.\n" if current_segment else f"{segment}.\n"
            else:
                if current_segment:
                    reconciled.append(current_segment)
                current_segment = f"{segment}.\n"
        if current_segment:
            reconciled.append(current_segment)
        return reconciled

    def secondary_split(reconciled: List[str], max_segment: int) -> List[str]:
        max_segment += 100
        reconciled_secondary = []
        for primary_segment in reconciled:
            if len(primary_segment.split()) > max_segment:
                segments = primary_segment.split(". ")
                current_segment = ""
                for segment in segments:
                    segment = segment.strip()
                    if not segment:
                        continue
                    combined = f"{current_segment} {segment}" if current_segment else segment
                    if len(combined.split()) < max_segment:
                        current_segment = f"{combined}. " if current_segment else f"{segment}. "
                    else:
                        if current_segment:
                            reconciled_secondary.append(current_segment)
                        current_segment = f"{segment}. "
                if current_segment:
                    reconciled_secondary.append(current_segment)
            else:
                reconciled_secondary.append(primary_segment)
        return reconciled_secondary

    def tertiary_split(reconciled: List[str], max_segment: int) -> List[str]:
        max_segment += 200
        reconciled_tertiary = []
        for secondary_segment in reconciled:
            words = secondary_segment.split()
            if len(words) > max_segment:
                for i in range(0, len(words), max_segment):
                    chunk = " ".join(words[i:i + max_segment])
                    reconciled_tertiary.append(chunk)
            else:
                reconciled_tertiary.append(secondary_segment)
        return reconciled_tertiary

    text = re.sub(r" +\n", "\n", text)
    reconciled = main_split(text, max_segment)
    reconciled = secondary_split(reconciled, max_segment)
    reconciled = tertiary_split(reconciled, max_segment)
    return reconciled

def chunk_dataframe(df: pd.DataFrame, 
                   text_column: str = 'text',
                   id_column: Optional[str] = 'id',
                   lang_column: Optional[str] = 'lang',
                   max_segment: int = 300) -> pd.DataFrame:
    """
    Processes a DataFrame to chunk the specified text column and returns a new DataFrame with chunks.
    
    Args:
        df: Input DataFrame containing text to be chunked
        text_column: Name of the column containing text to chunk
        id_column: Name of the column containing unique identifiers (optional)
        lang_column: Name of the column containing language codes (optional)
        max_segment: Maximum number of words per chunk
    
    Returns:
        DataFrame containing chunked text with additional metadata columns
    """
    new_rows = []
    
    for idx, row in df.iterrows():
        text = row.get(text_column)
        if not text:
            continue
            
        # Get optional columns if they exist
        original_id = row.get(id_column) if id_column in df.columns else f"row_{idx}"
        lang = row.get(lang_column) if lang_column in df.columns else None
        
        chunks = process_text(text, max_segment)
        
        for i, chunk in enumerate(chunks):
            new_row = {
                'id': f"{original_id}_chunk_{i}",
                'chunk_id': i,
                text_column: chunk.strip(),
                'word_count': len(chunk.strip().split())
            }
            
            # Add language if it exists
            if lang_column in df.columns:
                new_row[lang_column] = lang
                
            # Add any other columns from the original DataFrame
            for col in df.columns:
                if col not in [text_column, id_column, lang_column]:
                    new_row[col] = row[col]
                    
            new_rows.append(new_row)
    
    if not new_rows:
        return pd.DataFrame()
        
    return pd.DataFrame(new_rows)