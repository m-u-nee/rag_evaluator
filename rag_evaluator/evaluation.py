"""
Module for evaluating RAG model outputs through multiple metrics.
Focuses on citation quality, reference accuracy, and content analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
from .reference import ReferenceExtractor
import textdistance

class RAGEvaluator:
    """Class for evaluating RAG model outputs."""
    
    def __init__(self):
        self.reference_extractor = ReferenceExtractor()
        
    def evaluate_generation(self, 
                          generated_text: str, 
                          generation_id: str,
                          model_name: str) -> Dict[str, float]:
        """
        Evaluate a single generation from a RAG model.
        
        Args:
            generated_text (str): The generated text to evaluate
            generation_id (str): Unique identifier for this generation
            model_name (str): Name of the model being evaluated
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        # Extract references and sources
        processed = self.reference_extractor.process_rag_output(generated_text, generation_id)
        references_df = processed['references']
        sources = processed['sources']
        
        # Calculate all metrics
        metrics = {}
        
        # Basic statistics
        metrics.update(self._calculate_basic_stats(references_df))
        
        # Reference quality metrics
        metrics.update(self._evaluate_reference_quality(references_df))
        
        # Citation quality metrics
        metrics.update(self._evaluate_citation_quality(references_df, sources))
        
        # Add metadata
        metrics['generation_id'] = generation_id
        metrics['model_name'] = model_name
        
        return metrics
    
    def evaluate_dataset(self, 
                        generations: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Evaluate a dataset of RAG model generations.
        
        Args:
            generations (List[Dict]): List of dictionaries containing generations
                Each dict should have: 
                - 'text': generated text
                - 'generation_id': unique identifier
                - 'model': model name
                
        Returns:
            pd.DataFrame: DataFrame with evaluation results
        """
        results = []
        
        for gen in generations:
            metrics = self.evaluate_generation(
                gen['text'],
                gen['generation_id'],
                gen['model']
            )
            results.append(metrics)
            
        return pd.DataFrame(results)

    def _calculate_basic_stats(self, references_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic reference statistics."""
        if len(references_df) == 0:
            return {
                'num_references': 0,
                'avg_citation_length': 0,
                'avg_context_length': 0
            }
            
        return {
            'num_references': len(references_df),
            'avg_citation_length': references_df['citation_size'].mean(),
            'avg_context_length': references_df['statement_size'].mean()
        }

    def _evaluate_reference_quality(self, references_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate quality of references."""
        if len(references_df) == 0:
            return {
                'valid_reference_ratio': 0,
                'duplicate_reference_ratio': 0
            }
            
        metrics = {}
        
        # Valid references (more than 2 words)
        metrics['valid_reference_ratio'] = (
            references_df['citation_size'] > 2).mean()
        
        # Duplicate references
        duplicates = references_df.duplicated(
            subset=['generation_id', 'citation'], 
            keep=False
        ).mean()
        metrics['duplicate_reference_ratio'] = duplicates
        
        return metrics

    def _evaluate_citation_quality(self, 
                                 references_df: pd.DataFrame, 
                                 sources: Dict[str, str]) -> Dict[str, float]:
        """
        Evaluate quality of citations by comparing them to source texts.
        
        Uses fuzzy matching to detect hallucinated content.
        """
        if len(references_df) == 0 or not sources:
            return {
                'citation_accuracy': 0,
                'hallucination_ratio': 1.0  # Worst case
            }
            
        def calculate_similarity(citation: str, source: str) -> float:
            """Calculate text similarity using Levenshtein distance."""
            return textdistance.levenshtein.normalized_similarity(citation, source)
        
        # For each citation, find best matching source
        citation_scores = []
        
        for _, ref in references_df.iterrows():
            citation = ref['citation']
            best_score = 0
            
            # Compare with each source
            for source_text in sources.values():
                # Split source into chunks roughly the size of the citation
                words = source_text.split()
                chunk_size = len(citation.split()) + 5  # Allow some flexibility
                
                # Slide through source text
                for i in range(len(words) - chunk_size + 1):
                    chunk = ' '.join(words[i:i + chunk_size])
                    score = calculate_similarity(citation, chunk)
                    best_score = max(best_score, score)
                    
            citation_scores.append(best_score)
        
        # Calculate metrics
        citation_scores = np.array(citation_scores)
        metrics = {
            'citation_accuracy': np.mean(citation_scores),
            'hallucination_ratio': np.mean(citation_scores < 0.7)  # Threshold for hallucination
        }
        
        return metrics
    
    def get_evaluation_report(self, metrics_df: pd.DataFrame) -> str:
        """
        Generate a human-readable report from evaluation metrics.
        
        Args:
            metrics_df (pd.DataFrame): DataFrame with evaluation results
            
        Returns:
            str: Formatted evaluation report
        """
        report = []
        
        # Group by model if multiple models
        grouped = metrics_df.groupby('model_name')
        
        for model_name, group in grouped:
            report.append(f"\nModel: {model_name}")
            report.append("-" * 40)
            
            # Basic stats
            report.append(f"Total generations: {len(group)}")
            report.append(f"Average references per generation: {group['num_references'].mean():.2f}")
            
            # Reference quality
            report.append("\nReference Quality:")
            report.append(f"- Valid reference ratio: {group['valid_reference_ratio'].mean():.2%}")
            report.append(f"- Duplicate reference ratio: {group['duplicate_reference_ratio'].mean():.2%}")
            
            # Citation quality
            report.append("\nCitation Quality:")
            report.append(f"- Citation accuracy: {group['citation_accuracy'].mean():.2%}")
            report.append(f"- Hallucination ratio: {group['hallucination_ratio'].mean():.2%}")
            
        return "\n".join(report)