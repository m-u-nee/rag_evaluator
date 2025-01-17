import pandas as pd
from vllm import LLM, SamplingParams
import re
import pathlib
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

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
    def __init__(self, model_path: str, max_model_len: int = 8128):
        """
        Initialize the RAG evaluator.
        
        Args:
            model_path (str): Path to the evaluation model
            max_model_len (int): Maximum model length for processing
        """
        self.llm = LLM(model_path, max_model_len=max_model_len)
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
            return ""
        return re.sub(r'<ref.?</ref>', '', text, flags=re.DOTALL)

    @staticmethod
    def _extract_content(text: str, start_tag: str, end_tag: str) -> Optional[str]:
        """Extract content between start and end tags."""
        try:
            pattern = f"{re.escape(start_tag)}(.*?){re.escape(end_tag)}"
            match = re.search(pattern, text, re.DOTALL)
            if match is None:
                return None
            content = match.group(1)
            return content.strip() if content else None
        except Exception:
            return None

    @staticmethod
    def _extract_components(text: str) -> Dict[str, str]:
        """Extract evaluation components from generated text."""
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

        except Exception:
            pass

        return components

    def evaluate_single(self, query: str, answer: str) -> EvaluationResults:
        """Evaluate a single query-answer pair."""
        clean_answer = self._clean_citations(answer)
        input_text = f'### Query ###\n{query}\n\n### Answer ###\n{clean_answer}\n\n### Query analysis ###\n'

        output = self.llm.generate([input_text], self.sampling_params)[0]
        generated_text = output.outputs[0].text

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
        if len(queries) != len(answers):
            raise ValueError("Number of queries and answers must match")

        input_texts = []
        for query, answer in zip(queries, answers):
            clean_answer = self._clean_citations(answer)
            input_text = f'### Query ###\n{query}\n\n### Answer ###\n{clean_answer}\n\n### Query analysis ###\n'
            input_texts.append(input_text)

        outputs = self.llm.generate(input_texts, self.sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]

        results = []
        for generated_text in generated_texts:
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
        queries = []
        answers = []

        if "text" not in df.columns or "generated_response" not in df.columns:
            raise ValueError("Missing required columns 'text' or 'generated_response'")

        # Handle data preparation
        df["text"] = df["text"].fillna("").astype(str)
        df["generated_response"] = df["generated_response"].fillna("").astype(str)
        concatenated_text = df["text"] + df["generated_response"]

        for text in concatenated_text.tolist():
            query = self._extract_content(text, "<|query_start|>", "<|query_end|>")
            answer = self._extract_content(text, "<|answer_start|>", "<|answer_end|>")

            if query is not None and answer is not None:
                queries.append(query)
                answers.append(answer)

        evaluation_results = self.evaluate_batch(queries, answers)

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
        df = pd.read_parquet(input_file)
        results_df = self._process_data(df)
        metrics_df = process_evaluation_metrics(results_df)

        if output_file:
            # Handle file saving
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
        results_df = self._process_data(df)
        return process_evaluation_metrics(results_df)

def process_evaluation_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Process evaluation metrics and convert them to numerical scores."""
    # Process language quality index
    language_results = (
        df.groupby(['language_quality'])
        .size()
        .reset_index(name='n')
        .dropna(subset=['language_quality'])
    )

    language_results['grounding_score'] = language_results['n'] / language_results['n'].sum()

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

    language_index['language_quality_index'] = language_index['n'] / language_index['n'].sum()

    language_quality_score = float(
        language_index[language_index['language_evaluation'] == 'Positive']
        ['language_quality_index']
    )

    # Process reasoning quality index
    reasoning_results = (
        df.groupby(['reasoning_quality'])
        .size()
        .reset_index(name='n')
        .dropna(subset=['reasoning_quality'])
    )

    reasoning_results['grounding_score'] = reasoning_results['n'] / reasoning_results['n'].sum()

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

    reasoning_index['reasoning_quality_index'] = reasoning_index['n'] / reasoning_index['n'].sum()

    reasoning_quality_score = float(
        reasoning_index[reasoning_index['reasoning_quality'] == 'Positive']
        ['reasoning_quality_index']
    )

    # Process query adherence index
    query_results = (
        df.groupby(['query_adherence'])
        .size()
        .reset_index(name='n')
        .dropna(subset=['query_adherence'])
    )

    query_results['grounding_score'] = query_results['n'] / query_results['n'].sum()

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

    query_index['query_adherence_index'] = query_index['n'] / query_index['n'].sum()

    query_adherence_score = float(
        query_index[query_index['query_adherence'] == 'Positive']
        ['query_adherence_index']
    )

    # Combine indices
    indices = pd.DataFrame({
        'language_quality_index': [language_quality_score],
        'reasoning_quality_index': [reasoning_quality_score],
        'query_adherence_index': [query_adherence_score]
    })

    indices['combined_index'] = indices.mean(axis=1)

    return indices