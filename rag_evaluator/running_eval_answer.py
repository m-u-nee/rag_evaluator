import pandas as pd
from vllm import LLM, SamplingParams
import re
import pathlib
import os
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass
import numpy as np

ModelType = Literal['pleias', 'other']

@dataclass
class EvaluationResults:
    """Container for RAG evaluation results"""
    query_analysis: str
    query_adherence: str
    answer_analysis: str
    language_quality: str
    reasoning_quality: str
    raw_analysis: str

class RAGLLMEvaluator:
    def __init__(self, model_path: str, model_type: ModelType = 'other', max_model_len: int = 8128):
        """
        Initialize the RAG evaluator.
        """
        print(f"Initializing RAGLLMEvaluator with model_path={model_path}, model_type={model_type}")
        self.llm = LLM(model_path, max_model_len=max_model_len)
        self.model_type = model_type
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=3000,
            presence_penalty=1.2,
            stop=["#END#"]
        )

    @staticmethod
    def _clean_citations(text: Optional[str]) -> str:
        """Remove text between <ref and </ref>"""
        if text is None:
            print("WARNING: Input text is None in _clean_citations")
            return ""
        print(f"Cleaning citations from text (first 100 chars): {text[:100]}...")
        cleaned_text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        print(f"After cleaning citations (first 100 chars): {cleaned_text[:100]}...")
        return cleaned_text

    @staticmethod
    def _extract_pleias_content(text: str, start_tag: str, end_tag: str) -> Optional[str]:
        """Extract content between Pleias-style tags."""
        print(f"\nAttempting to extract content between tags: {start_tag} and {end_tag}")
        print(f"Input text (first 100 chars): {text[:100]}...")
        try:
            pattern = f"{re.escape(start_tag)}(.*?){re.escape(end_tag)}"
            match = re.search(pattern, text, re.DOTALL)
            if match is None:
                print(f"WARNING: No match found for pattern {pattern}")
                return None
            content = match.group(1)
            result = content.strip() if content else None
            print(f"Extracted content (first 100 chars): {result[:100] if result else 'None'}...")
            return result
        except Exception as e:
            print(f"ERROR in _extract_pleias_content: {str(e)}")
            return None
    
    @staticmethod
    def _extract_other_query(text: str) -> Optional[str]:
        """Extract query from 'other' format text."""
        print("\nAttempting to extract query from 'other' format")
        print(f"Input text (first 100 chars): {text[:100]}...")
        try:
            pattern = r"question posted by a user:\s*(.*?)\s*You answer should"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                result = match.group(1).strip()
                print(f"Extracted query: {result[:100]}...")
                return result
            print("WARNING: No query match found")
        except Exception as e:
            print(f"ERROR in _extract_other_query: {str(e)}")
        return None

    @staticmethod
    def _extract_other_answer(text: str) -> Optional[str]:
        """Extract answer from 'other' format text."""
        print("\nAttempting to extract answer from 'other' format")
        print(f"Input text (first 100 chars): {text[:100]}...")
        try:
            pattern = r"### Referenced answer ###\s*\n(.*?)(?:\n###|$)"
            matches = list(re.finditer(pattern, text, re.DOTALL))
            
            print(f"Found {len(matches)} potential answer matches")
            if matches and len(matches) > 1:
                result = matches[-1].group(1).strip()
                print(f"Using last match. Extracted answer (first 100 chars): {result[:100]}...")
                return result
            print("WARNING: No valid answer match found")
            return None
        except Exception as e:
            print(f"ERROR in _extract_other_answer: {str(e)}")
            return None

    @staticmethod
    def _extract_components(text: str) -> Dict[str, str]:
        """Extract evaluation components from generated text."""
        print("\nExtracting evaluation components")
        print(f"Input text (first 100 chars): {text[:100]}...")
        
        components = {
            'query_analysis': '',
            'query_adherence': '',
            'answer_analysis': '',
            'language_quality': '',
            'reasoning_quality': ''
        }

        try:
            patterns = {
                'query_analysis': r'### Query analysis ###\n(.*?)\n\n###',
                'query_adherence': r'### Query adherence ###\n(.*?)\n\n###',
                'answer_analysis': r'### Answer analysis ###\n(.*?)\n\n###',
                'language_quality': r'### Language quality ###\n(.*?)\n\n',
                'reasoning_quality': r'### Reasoning quality ###\n(.*?)(?:\n\n|$)'
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    components[key] = match.group(1).strip()
                    print(f"Extracted {key} (first 50 chars): {components[key][:50]}...")
                else:
                    print(f"WARNING: No match found for {key}")

        except Exception as e:
            print(f"ERROR in _extract_components: {str(e)}")

        return components

    def evaluate_single(self, query: str, answer: str) -> EvaluationResults:
        """Evaluate a single query-answer pair."""
        print(f"\nEvaluating single query-answer pair")
        print(f"Query: {query[:100]}...")
        print(f"Answer: {answer[:100]}...")
        
        clean_answer = self._clean_citations(answer)
        input_text = f'### Query ###\n{query}\n\n### Answer ###\n{clean_answer}\n\n### Query analysis ###\n'
        
        print("Generating evaluation...")
        output = self.llm.generate([input_text], self.sampling_params)[0]
        generated_text = output.outputs[0].text
        print(f"Generated evaluation text (first 100 chars): {generated_text[:100]}...")

        components = self._extract_components(generated_text)
        return EvaluationResults(
            query_analysis=components['query_analysis'],
            query_adherence=components['query_adherence'],
            answer_analysis=components['answer_analysis'],
            language_quality=components['language_quality'],
            reasoning_quality=components['reasoning_quality'],
            raw_analysis=generated_text
        )

    def evaluate_batch(self, queries: List[str], answers: List[str]) -> List[EvaluationResults]:
        """Evaluate multiple query-answer pairs."""
        print(f"\nEvaluating batch of {len(queries)} query-answer pairs")
        if len(queries) != len(answers):
            raise ValueError("Number of queries and answers must match")

        input_texts = []
        for i, (query, answer) in enumerate(zip(queries, answers)):
            print(f"\nProcessing pair {i+1}/{len(queries)}")
            clean_answer = self._clean_citations(answer)
            input_text = f'### Query ###\n{query}\n\n### Answer ###\n{clean_answer}\n\n### Query analysis ###\n'
            input_texts.append(input_text)

        print("Generating batch evaluations...")
        outputs = self.llm.generate(input_texts, self.sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]

        results = []
        for i, generated_text in enumerate(generated_texts):
            print(f"\nExtracting components for result {i+1}/{len(generated_texts)}")
            components = self._extract_components(generated_text)
            results.append(EvaluationResults(
                query_analysis=components['query_analysis'],
                query_adherence=components['query_adherence'],
                answer_analysis=components['answer_analysis'],
                language_quality=components['language_quality'],
                reasoning_quality=components['reasoning_quality'],
                raw_analysis=generated_text
            ))

        return results

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process DataFrame and evaluate query-answer pairs."""
        print(f"\nProcessing DataFrame with {len(df)} rows")
        queries = []
        answers = []

        if "text" not in df.columns or "generated_response" not in df.columns:
            raise ValueError("Missing required columns 'text' or 'generated_response'")

        print("Preparing data...")
        df["text"] = df["text"].fillna("").astype(str)
        df["generated_response"] = df["generated_response"].fillna("").astype(str)

        for idx, row in df.iterrows():
            print(f"\nProcessing row {idx+1}/{len(df)}")
            query = None
            answer = None
            
            if self.model_type == 'pleias':
                print("Using Pleias format extraction")
                query = self._extract_pleias_content(row["text"], "<|query_start|>", "<|query_end|>")
                answer = self._extract_pleias_content(row["generated_response"], "<|answer_start|>", "<|answer_end|>")
            else:
                print("Using other format extraction")
                query = self._extract_other_query(row["text"])
                answer = self._extract_other_answer(row["generated_response"])

            if query is not None and answer is not None:
                queries.append(query)
                answers.append(answer)
            else:
                print(f"WARNING: Skipping row {idx} - Could not extract query or answer")

        print(f"\nSuccessfully extracted {len(queries)} query-answer pairs")
        evaluation_results = self.evaluate_batch(queries, answers)

        print("Creating results DataFrame...")
        results_df = pd.DataFrame({
            'chunk_id': range(len(queries)),
            'query': queries,
            'answer': answers,
            'query_analysis': [r.query_analysis for r in evaluation_results],
            'query_adherence': [r.query_adherence for r in evaluation_results],
            'answer_analysis': [r.answer_analysis for r in evaluation_results],
            'language_quality': [r.language_quality for r in evaluation_results],
            'reasoning_quality': [r.reasoning_quality for r in evaluation_results],
            'raw_analysis': [r.raw_analysis for r in evaluation_results]
        })

        return results_df

    def evaluate_from_file(self, input_file: str, output_file: Optional[str] = None) -> pd.DataFrame:
        """Evaluate query-answer pairs from a parquet file."""
        print(f"\nEvaluating from file: {input_file}")
        df = pd.read_parquet(input_file)
        print(f"Loaded DataFrame with {len(df)} rows")
        
        results_df = self._process_data(df)
        metrics_df = process_evaluation_metrics(results_df)

        if output_file:
            print(f"Saving results to: {output_file}")
            directory = os.path.dirname(output_file)
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            
            results_df.to_parquet(output_file)
            
            metrics_file = output_file.replace('.parquet', '_metrics.parquet')
            metrics_df.to_parquet(metrics_file)
            
            print(f"Raw evaluation results saved to: {output_file}")
            print(f"Numeric metrics saved to: {metrics_file}")

        return metrics_df

    def evaluate_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate query-answer pairs from a DataFrame."""
        print(f"\nEvaluating from DataFrame with {len(df)} rows")
        results_df = self._process_data(df)
        return process_evaluation_metrics(results_df)
        def process_evaluation_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Process evaluation metrics and convert them to numerical scores."""
    print(f"\nProcessing evaluation metrics for DataFrame with {len(df)} rows")
    
    # Process language quality index
    print("\n=== Processing Language Quality Index ===")
    print("Initial value counts for language_quality:")
    print(df['language_quality'].value_counts())
    
    language_results = (
        df.groupby(['language_quality'])
        .size()
        .reset_index(name='n')
        .dropna(subset=['language_quality'])
    )
    print("\nGrouped language results:")
    print(language_results)

    language_results['grounding_score'] = language_results['n'] / language_results['n'].sum()
    print("\nLanguage results with grounding scores:")
    print(language_results)

    language_index = (
        language_results.assign(
            language_evaluation=lambda x: np.where(
                x['language_quality'].isin(['High', 'Correct']),
                'Positive',
                'Negative'
            )
        )
        .groupby(['language_evaluation'])
        .agg(n=('n', 'sum'))
        .reset_index()
    )
    print("\nLanguage index after categorization:")
    print(language_index)

    language_index['language_quality_index'] = language_index['n'] / language_index['n'].sum()
    print("\nFinal language quality index:")
    print(language_index)

    # Safe extraction of language quality score with fallback
    try:
        positive_language = language_index[language_index['language_evaluation'] == 'Positive']
        language_quality_score = positive_language['language_quality_index'].iloc[0] if not positive_language.empty else 0.0
        print(f"\nExtracted language quality score: {language_quality_score}")
    except Exception as e:
        print(f"\nERROR extracting language quality score: {str(e)}")
        language_quality_score = 0.0
        print(f"Using fallback language quality score: {language_quality_score}")

    # Process reasoning quality index
    print("\n=== Processing Reasoning Quality Index ===")
    print("Initial value counts for reasoning_quality:")
    print(df['reasoning_quality'].value_counts())
    
    reasoning_results = (
        df.groupby(['reasoning_quality'])
        .size()
        .reset_index(name='n')
        .dropna(subset=['reasoning_quality'])
    )
    print("\nGrouped reasoning results:")
    print(reasoning_results)

    reasoning_results['grounding_score'] = reasoning_results['n'] / reasoning_results['n'].sum()
    print("\nReasoning results with grounding scores:")
    print(reasoning_results)

    reasoning_index = (
        reasoning_results.assign(
            reasoning_quality=lambda x: np.where(
                x['reasoning_quality'].isin(['Solid', 'Slightly']),
                'Positive',
                'Negative'
            )
        )
        .groupby(['reasoning_quality'])
        .agg(n=('n', 'sum'))
        .reset_index()
    )
    print("\nReasoning index after categorization:")
    print(reasoning_index)

    reasoning_index['reasoning_quality_index'] = reasoning_index['n'] / reasoning_index['n'].sum()
    print("\nFinal reasoning quality index:")
    print(reasoning_index)

    # Safe extraction of reasoning quality score with fallback
    try:
        positive_reasoning = reasoning_index[reasoning_index['reasoning_quality'] == 'Positive']
        reasoning_quality_score = positive_reasoning['reasoning_quality_index'].iloc[0] if not positive_reasoning.empty else 0.0
        print(f"\nExtracted reasoning quality score: {reasoning_quality_score}")
    except Exception as e:
        print(f"\nERROR extracting reasoning quality score: {str(e)}")
        reasoning_quality_score = 0.0
        print(f"Using fallback reasoning quality score: {reasoning_quality_score}")

    # Process query adherence index
    print("\n=== Processing Query Adherence Index ===")
    print("Initial value counts for query_adherence:")
    print(df['query_adherence'].value_counts())
    
    query_results = (
        df.groupby(['query_adherence'])
        .size()
        .reset_index(name='n')
        .dropna(subset=['query_adherence'])
    )
    print("\nGrouped query results:")
    print(query_results)

    query_results['grounding_score'] = query_results['n'] / query_results['n'].sum()
    print("\nQuery results with grounding scores:")
    print(query_results)

    query_index = (
        query_results[query_results['query_adherence'] != 'Refusal']
        .assign(
            query_adherence=lambda x: np.where(
                x['query_adherence'].isin(['Actual answer', 'Partial answer']),
                'Positive',
                'Negative'
            )
        )
        .groupby(['query_adherence'])
        .agg(n=('n', 'sum'))
        .reset_index()
    )
    print("\nQuery index after categorization:")
    print(query_index)

    query_index['query_adherence_index'] = query_index['n'] / query_index['n'].sum()
    print("\nFinal query adherence index:")
    print(query_index)

    # Safe extraction of query adherence score with fallback
    try:
        positive_query = query_index[query_index['query_adherence'] == 'Positive']
        query_adherence_score = positive_query['query_adherence_index'].iloc[0] if not positive_query.empty else 0.0
        print(f"\nExtracted query adherence score: {query_adherence_score}")
    except Exception as e:
        print(f"\nERROR extracting query adherence score: {str(e)}")
        query_adherence_score = 0.0
        print(f"Using fallback query adherence score: {query_adherence_score}")

    # Combine indices
    print("\n=== Combining Final Indices ===")
    indices = pd.DataFrame({
        'language_quality_index': [language_quality_score],
        'reasoning_quality_index': [reasoning_quality_score],
        'query_adherence_index': [query_adherence_score]
    })
    print("\nIndividual indices:")
    print(indices)

    indices['combined_index'] = indices.mean(axis=1)
    print("\nFinal combined index:")
    print(indices)

    return indices