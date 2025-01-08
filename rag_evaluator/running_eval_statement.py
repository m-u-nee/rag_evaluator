import pandas as pd
from vllm import LLM, SamplingParams
import re
import pathlib
import os

def clean_dataset(df):
    """
    Clean the dataset by removing quotes from citations and cleaning up statement prefixes.
    
    Args:
        df (pandas.DataFrame): Input DataFrame with 'citation' and 'statement' columns
        
    Returns:
        pandas.DataFrame: Cleaned DataFrame
    """
    result = df.copy()
    result['citation'] = result['citation'].str.replace('"', '', regex=False)
    prefixes_pattern = r'^(\. |\)\. |\* |\*\*\) )'
    result['statement'] = result['statement'].str.replace(prefixes_pattern, '', regex=True)
    return result

def extract_content(text, start_tag, end_tag):
    pattern = f"{start_tag}(.*?){end_tag}"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

def get_grounding_context(text, ref_start_pos):
    """
    Extract grounding context before a reference based on specific rules:
    1. Stop at newline (paragraph boundary)
    2. Stop at previous reference
    3. Limit to roughly 3 sentences
    """
    current_pos = ref_start_pos
    newline_boundary = text.rfind('\n', 0, current_pos)
    prev_ref_boundary = text.rfind('</ref>', 0, current_pos)
    
    if prev_ref_boundary != -1:
        prev_ref_boundary += len('</ref>')
    
    effective_boundary = max(newline_boundary, prev_ref_boundary)
    
    if effective_boundary == -1:
        effective_boundary = 0
    
    context = text[effective_boundary:ref_start_pos].strip()
    sentences = re.split(r'(?<=[.!?])\s+', context)
    if len(sentences) > 3:
        context = ' '.join(sentences[-3:])
    
    return context.strip()

def extract_references(text, generation_id):
    """
    Extract references with comprehensive grounding context.
    """
    references = []
    ref_pattern = r'<ref\s+name="([^"]+)">([^<]+)<\/ref>'
    
    for match in re.finditer(ref_pattern, text):
        ref_id = match.group(1)
        citation = match.group(2)
        grounding_context = get_grounding_context(text, match.start())
        
        reference_obj = {
            'generation_id': generation_id,
            'statement': grounding_context,
            'statement_size': len(grounding_context.split()),
            'reference_id': ref_id,
            'citation': citation,
            'citation_size': len(citation.split()),
        }
        references.append(reference_obj)
    
    return references

def extract_generated_components(text):
    analysis = re.search(r'### Analysis ###\n(.*?)\n\n###', text, re.DOTALL)
    analysis = analysis.group(1).strip() if analysis else ""
    
    judgement = re.search(r'### Judgement ###\n(.*?)(?:\n\n|$)', text, re.DOTALL)
    judgement = judgement.group(1).strip() if judgement else ""
    
    return {
        'analysis': analysis,
        'judgement': judgement
    }

def main():
    # Hardcoded input and output paths
    input_file = "/path/to/your/input.parquet"
    output_file = "/path/to/your/output.parquet"
    
    # Create output directory if it doesn't exist
    directory = os.path.dirname(output_file)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Load model and set sampling parameters
    llm = LLM("llama-rag-eval/llama-rag-eval", max_model_len=8128)
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=3000,
        presence_penalty=1.2,
        stop=["#END#"]
    )
    
    # Read input file
    result = pd.read_parquet(input_file)
    
    # Process references
    structured_reference_set = []
    for ind, row in result.iterrows():
        answer = extract_content(row["text"], "<|answer_start|>", "<|answer_end|>")
        references = extract_references(answer, row["chunk_id"])
        structured_reference_set.extend(references)
    
    # Convert to DataFrame and clean
    structured_reference_set = pd.DataFrame(structured_reference_set)
    structured_reference_set = clean_dataset(structured_reference_set)
    
    # Prepare texts for model
    list_texts = []
    for ind, row in structured_reference_set.iterrows():
        list_texts.append(
            f"### Statement ###\n{row['statement']}\n\n### Citation ###\n{row['citation']}\n\n### Analysis ###\n"
        )
    
    # Generate outputs
    outputs = llm.generate(list_texts, sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]
    
    # Extract components and add to DataFrame
    components = [extract_generated_components(text) for text in generated_texts]
    structured_reference_set['analysis'] = [comp['analysis'] for comp in components]
    structured_reference_set['judgement'] = [comp['judgement'] for comp in components]
    
    # Save output
    structured_reference_set.to_parquet(output_file)

if __name__ == "__main__":
    main()