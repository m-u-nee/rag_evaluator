import pandas as pd
from vllm import LLM, SamplingParams
import re
import pathlib
import os

def clean_citations(text):
    # Remove text between <ref and </ref>
    return re.sub(r'<ref.*?</ref>', '', text, flags=re.DOTALL)

def extract_content(text, start_tag, end_tag):
    pattern = f"{start_tag}(.*?){end_tag}"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_generated_components(text):
    # Extract each component using regex
    query_analysis = re.search(r'### Query analysis ###\n(.*?)\n\n###', text, re.DOTALL)
    query_analysis = query_analysis.group(1).strip() if query_analysis else ""
    
    query_adherence = re.search(r'### Query adherence ###\n(.*?)\n\n###', text, re.DOTALL)
    query_adherence = query_adherence.group(1).strip() if query_adherence else ""
    
    answer_analysis = re.search(r'### Answer analysis ###\n(.*?)\n\n###', text, re.DOTALL)
    answer_analysis = answer_analysis.group(1).strip() if answer_analysis else ""
    
    language_quality = re.search(r'### Language quality ###\n(.*?)\n\n', text, re.DOTALL)
    language_quality = language_quality.group(1).strip() if language_quality else ""
    
    reasoning_quality = re.search(r'### Reasoning quality ###\n(.*?)(?:\n\n|$)', text, re.DOTALL)
    reasoning_quality = reasoning_quality.group(1).strip() if reasoning_quality else ""
    
    return {
        'query_analysis': query_analysis,
        'query_adherence': query_adherence,
        'answer_analysis': answer_analysis,
        'language_quality': language_quality,
        'reasoning_quality': reasoning_quality
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
    
    # Extract texts and IDs
    texts = result["text"].tolist()
    list_ids = result["chunk_id"].tolist()
    
    # Process texts
    list_texts = []
    for text in texts:
        query = extract_content(text, "<|query_start|>", "<|query_end|>")
        answer = extract_content(text, "<|answer_start|>", "<|answer_end|>")
        answer = clean_citations(answer)
        
        combined_text = f'### Query ###\n{query}\n\n### Answer ###\n{answer}\n\n### Query analysis ###\n'
        list_texts.append(combined_text)
    
    # Generate outputs
    outputs = llm.generate(list_texts, sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]
    
    # Extract components
    components = [extract_generated_components(text) for text in generated_texts]
    
    # Create DataFrame with results
    df = pd.DataFrame({
        'chunk_id': list_ids,
        'original_text': list_texts,
        'analysis': generated_texts,
        'query_analysis': [comp['query_analysis'] for comp in components],
        'query_adherence': [comp['query_adherence'] for comp in components],
        'answer_analysis': [comp['answer_analysis'] for comp in components],
        'language_quality': [comp['language_quality'] for comp in components],
        'reasoning_quality': [comp['reasoning_quality'] for comp in components]
    })
    
    # Save output
    df.to_parquet(output_file)

if __name__ == "__main__":
    main()