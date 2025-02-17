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
        print(f"Initializing evaluator with model_type: {model_type}")
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
            print("Warning: Received None text for citation cleaning")
            return ""
        cleaned = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        print(f"Cleaned citations. Original length: {len(text)}, Cleaned length: {len(cleaned)}")
        return cleaned

    @staticmethod
    def _extract_pleias_content(text: str, start_tag: str, end_tag: str) -> Optional[str]:
        """Extract content between Pleias-style tags."""
        try:
            pattern = f"{re.escape(start_tag)}(.*?){re.escape(end_tag)}"
            match = re.search(pattern, text, re.DOTALL)
            if match is None:
                print(f"No match found for pattern between {start_tag} and {end_tag}")
                print(f"Text preview: {text[:200]}...")
                return None
            content = match.group(1)
            print(f"Extracted Pleias content. Length: {len(content) if content else 0}")
            return content.strip() if content else None
        except Exception as e:
            print(f"Error in Pleias extraction: {str(e)}")
            return None
    
    @staticmethod
    def _extract_other_query(text: str) -> Optional[str]:
        """Extract query from 'other' format text."""
        try:
            # Debugging original text
            print(f"\nAttempting to extract query from text:")
            print(f"Text length: {len(text)}")
            print(f"Text preview: {text[:300]}...")
            
            pattern = r"question posted by a user:\s*(.*?)\s*You answer should"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                query = match.group(1).strip()
                print(f"Successfully extracted query: {query[:100]}...")
                return query
            
            print("No query match found using primary pattern")
            # Try alternative pattern
            alt_pattern = r"Question:\s*(.*?)\s*(?:Answer:|$)"
            alt_match = re.search(alt_pattern, text, re.DOTALL)
            if alt_match:
                query = alt_match.group(1).strip()
                print(f"Extracted query using alternative pattern: {query[:100]}...")
                return query
                
            print("Failed to extract query with all patterns")
            return None
        except Exception as e:
            print(f"Error in query extraction: {str(e)}")
            print(f"Problematic text: {text[:200]}...")
            return None

    @staticmethod
    def _extract_other_answer(text: str) -> Optional[str]:
        """Extract answer from 'other' format text."""
        try:
            print(f"\nAttempting to extract answer from text:")
            print(f"Text length: {len(text)}")
            print(f"Text preview: {text[:300]}...")
            
            pattern = r"### Referenced answer ###\s*\n(.*?)(?:\n###|$)"
            matches = list(re.finditer(pattern, text, re.DOTALL))
            
            print(f"Found {len(matches)} answer matches in text")
            if matches:
                if len(matches) > 1:
                    print("Multiple matches found, using last one")
                answer = matches[-1].group(1).strip()
                print(f"Successfully extracted answer: {answer[:100]}...")
                return answer
            
            # Try alternative pattern
            alt_pattern = r"Answer:\s*(.*?)(?:\n\n|\Z)"
            alt_match = re.search(alt_pattern, text, re.DOTALL)
            if alt_match:
                answer = alt_match.group(1).strip()
                print(f"Extracted answer using alternative pattern: {answer[:100]}...")
                return answer
                
            print("Failed to extract answer with all patterns")
            return None
        except Exception as e:
            print(f"Error in answer extraction: {str(e)}")
            print(f"Problematic text: {text[:200]}...")
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
            print("\nExtracting components from text:")
            print(f"Text length: {len(text)}")
            print(f"Text preview: {text[:200]}...")

            patterns = {
                'query_analysis': r'### Query analysis ###\n(.*?)\n\n###',
                'query_adherence': r'### Query adherence ###\n(.*?)\n\n###',
                'answer_analysis': r'### Answer analysis ###\n(.*?)\n\n###',
                'language_quality': r'### Language quality ###\n(.*?)(?:\n\n###|\Z)',
                'reasoning_quality': r'### Reasoning quality ###\n(.*?)(?:\n\n###|\Z)'
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    components[key] = match.group(1).strip()
                    print(f"{key}: Found match of length {len(components[key])}")
                    print(f"{key} content preview: {components[key][:100]}...")
                else:
                    print(f"{key}: No match found")
                    # Try alternative patterns based on key
                    if key == 'language_quality':
                        alt_match = re.search(r'Language:\s*(.*?)(?:\n\n|\Z)', text, re.DOTALL)
                        if alt_match:
                            components[key] = alt_match.group(1).strip()
                            print(f"{key}: Found match with alternative pattern")

        except Exception as e:
            print(f"Error in component extraction: {str(e)}")
            print("Component extraction failed, using empty values")

        return components

    def evaluate_single(self, query: str, answer: str) -> EvaluationResults:
        """Evaluate a single query-answer pair."""
        print(f"\nEvaluating single pair:")
        print(f"Query length: {len(query)}")
        print(f"Query preview: {query[:200]}...")
        print(f"Answer length: {len(answer)}")
        print(f"Answer preview: {answer[:200]}...")
        
        clean_answer = self._clean_citations(answer)
        input_text = f'### Query ###\n{query}\n\n### Answer ###\n{clean_answer}\n\n### Query analysis ###\n'
        
        print(f"Generated input text length: {len(input_text)}")
        print(f"Input text preview: {input_text[:200]}...")
        
        output = self.llm.generate([input_text], self.sampling_params)[0]
        generated_text = output.outputs[0].text
        print(f"Generated output text length: {len(generated_text)}")
        print(f"Output text preview: {generated_text[:200]}...")

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
        print(f"\nEvaluating batch of {len(queries)} pairs")
        if len(queries) != len(answers):
            raise ValueError(f"Number of queries ({len(queries)}) and answers ({len(answers)}) must match")

        results = []
        for i, (query, answer) in enumerate(zip(queries, answers)):
            print(f"\nProcessing pair {i+1}/{len(queries)}")
            try:
                result = self.evaluate_single(query, answer)
                results.append(result)
                print(f"Successfully evaluated pair {i+1}")
            except Exception as e:
                print(f"Error evaluating pair {i+1}: {str(e)}")
                # Add empty result to maintain alignment
                results.append(EvaluationResults(
                    query_analysis="",
                    query_adherence="",
                    answer_analysis="",
                    language_quality="",
                    reasoning_quality="",
                    raw_analysis=""
                ))

        return results

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process DataFrame and evaluate query-answer pairs."""
        print(f"\nProcessing DataFrame with {len(df)} rows")
        print("Columns present:", df.columns.tolist())
        print("\nColumn info:")
        print(df.info())
        
        queries = []
        answers = []

        if "text" not in df.columns or "generated_response" not in df.columns:
            raise ValueError("Missing required columns 'text' or 'generated_response'")

        # Handle data preparation
        df["text"] = df["text"].fillna("").astype(str)
        df["generated_response"] = df["generated_response"].fillna("").astype(str)

        print("\nProcessing rows:")
        for idx, row in df.iterrows():
            print(f"\nProcessing row {idx}:")
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
                print(f"Row {idx}: Successfully extracted query and answer")
                print(f"Query preview: {query[:100]}...")
                print(f"Answer preview: {answer[:100]}...")
                queries.append(query)
                answers.append(answer)
            else:
                print(f"Row {idx}: Failed to extract query or answer")
                if query is None:
                    print("Query extraction failed")
                if answer is None:
                    print("Answer extraction failed")

        print(f"\nSuccessfully extracted {len(queries)} query-answer pairs")
        
        if len(queries) == 0:
            print("WARNING: No valid query-answer pairs found!")
            return pd.DataFrame()

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

        print("\nResults DataFrame info:")
        print(results_df.info())
        print("\nSample of results:")
        print(results_df.head())
        print("\nNon-null counts:\n", results_df.count())
        print("\nUnique values in quality measures:")
        for col in ['language_quality', 'reasoning_quality', 'query_adherence']:
            print(f"\n{col} unique values:", results_df[col].unique())

        return results_df

def process_evaluation_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Process evaluation metrics and convert them to numerical scores."""
    print("\nProcessing evaluation metrics")
    print("Input DataFrame info:")
    print(df.info())
    
    if df.empty:
        print("WARNING: Empty DataFrame provided to process_evaluation_metrics")
        return pd.DataFrame({
            'language_quality_index': [0.0],
            'reasoning_quality_index': [0.0],
            'query_adherence_index': [0.0],
            'combined_index': [0.0]
        })

    # Print detailed value counts for each measure
    print("\nDetailed value counts:")
    for col in ['language_quality', 'reasoning_quality', 'query_adherence']:
        print(f"\n{col} value counts:")
        print(df[col].value_counts(dropna=False))
        print(f"{col} unique values:", df[col].unique())

    # Process language quality
    print("\nProcessing language quality metrics:")
    try:
        language_results = (
            df.groupby(['language_quality'])
            .size()
            .reset_index(name='n')
            .dropna(subset=['language_quality'])
        )
        print("\nLanguage quality grouping results:")
        print(language_results)

        language_results['grounding_score'] = language_results['n'] / language_results['n'].sum()
        
        print("\nClassifying language quality values:")
        language_index = (
            language_results.assign(
                language_evaluation=lambda x: np.where(
                    x['language_quality'].str.lower().isin(['high', 'correct', 'good', 'excellent']),
                    'Positive',
                    'Negative'
                )
            )
            .groupby(['language_evaluation'])
            .agg(n=('n', 'sum'))
            .reset_index()
        )
        print("\nLanguage quality classification results:")
        print(language_index)

        language_index['language_quality_index'] = language_index['n'] / language_index['n'].sum()
        print("\nLanguage quality indices:")
        print(language_index)

        # Safe extraction of language quality score
        try:
            positive_language = language_index[language_index['language_evaluation'] == 'Positive']
            language_quality_score = positive_language['language_quality_index'].iloc[0] if not positive_language.empty else 0.0
            print(f"\nCalculated language quality score: {language_quality_score}")
        except Exception as e:
            print(f"Error calculating language quality score: {str(e)}")
            language_quality_score = 0.0
    except Exception as e:
        print(f"Error processing language quality metrics: {str(e)}")
        language_quality_score = 0.0

    # Process reasoning quality
    print("\nProcessing reasoning quality metrics:")
    try:
        reasoning_results = (
            df.groupby(['reasoning_quality'])
            .size()
            .reset_index(name='n')
            .dropna(subset=['reasoning_quality'])
        )
        print("\nReasoning quality grouping results:")
        print(reasoning_results)

        reasoning_results['grounding_score'] = reasoning_results['n'] / reasoning_results['n'].sum()
        
        print("\nClassifying reasoning quality values:")
        reasoning_index = (
            reasoning_results.assign(
                reasoning_evaluation=lambda x: np.where(
                    x['reasoning_quality'].str.lower().isin(['solid', 'good', 'excellent', 'strong']),
                    'Positive',
                    'Negative'
                )
            )
            .groupby(['reasoning_evaluation'])
            .agg(n=('n', 'sum'))
            .reset_index()
        )
        print("\nReasoning quality classification results:")
        print(reasoning_index)

        reasoning_index['reasoning_quality_index'] = reasoning_index['n'] / reasoning_index['n'].sum()
        print("\nReasoning quality indices:")
        print(reasoning_index)

        # Safe extraction of reasoning quality score
        try:
            positive_reasoning = reasoning_index[reasoning_index['reasoning_evaluation'] == 'Positive']
            reasoning_quality_score = positive_reasoning['reasoning_quality_index'].iloc[0] if not positive_reasoning.empty else 0.0
            print(f"\nCalculated reasoning quality score: {reasoning_quality_score}")
        except Exception as e:
            print(f"Error calculating reasoning quality score: {str(e)}")
            reasoning_quality_score = 0.0
    except Exception as e:
        print(f"Error processing reasoning quality metrics: {str(e)}")
        reasoning_quality_score = 0.0

    # Process query adherence
    print("\nProcessing query adherence metrics:")
    try:
        query_results = (
            df.groupby(['query_adherence'])
            .size()
            .reset_index(name='n')
            .dropna(subset=['query_adherence'])
        )
        print("\nQuery adherence grouping results:")
        print(query_results)

        query_results['grounding_score'] = query_results['n'] / query_results['n'].sum()
        
        # Filter out refusals and classify remaining responses
        print("\nClassifying query adherence values:")
        query_index = (
            query_results[~query_results['query_adherence'].str.lower().isin(['refusal', 'refused'])]
            .assign(
                adherence_evaluation=lambda x: np.where(
                    x['query_adherence'].str.lower().isin(['actual answer', 'complete answer', 'full answer', 'correct answer']),
                    'Positive',
                    'Negative'
                )
            )
            .groupby(['adherence_evaluation'])
            .agg(n=('n', 'sum'))
            .reset_index()
        )
        print("\nQuery adherence classification results:")
        print(query_index)

        query_index['query_adherence_index'] = query_index['n'] / query_index['n'].sum()
        print("\nQuery adherence indices:")
        print(query_index)

        # Safe extraction of query adherence score
        try:
            positive_query = query_index[query_index['adherence_evaluation'] == 'Positive']
            query_adherence_score = positive_query['query_adherence_index'].iloc[0] if not positive_query.empty else 0.0
            print(f"\nCalculated query adherence score: {query_adherence_score}")
        except Exception as e:
            print(f"Error calculating query adherence score: {str(e)}")
            query_adherence_score = 0.0
    except Exception as e:
        print(f"Error processing query adherence metrics: {str(e)}")
        query_adherence_score = 0.0

    # Combine all indices
    print("\nCombining indices:")
    indices = pd.DataFrame({
        'language_quality_index': [language_quality_score],
        'reasoning_quality_index': [reasoning_quality_score],
        'query_adherence_index': [query_adherence_score]
    })

    indices['combined_index'] = indices.mean(axis=1)
    
    print("\nFinal indices:")
    print(indices)
    print("\nIndices info:")
    print(indices.info())
    
    return indices