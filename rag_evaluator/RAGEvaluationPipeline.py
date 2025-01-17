import argparse
from dataclasses import asdict
import pandas as pd
import os
from typing import Dict, Any, Union

from .running_eval_answer import RAGLLMEvaluator
from .hallucination import RAGHallucinationEvaluator

class RAGEvaluationPipeline:
    """
    Combined pipeline for evaluating RAG systems using both 
    LLM-based metrics and hallucination detection.
    """
    
    def __init__(self, model_path: str, max_model_len: int = 8192):
        """
        Initialize the evaluation pipeline.
        
        Args:
            model_path: Path to the LLM evaluation model
            max_model_len: Maximum model length for processing
        """
        self.llm_evaluator = RAGLLMEvaluator(model_path, max_model_len)
        self.hallucination_evaluator = RAGHallucinationEvaluator()
        
    def evaluate(self, 
                data: Union[str, pd.DataFrame],
                output_file: str = None,
                response_col: str = 'generated_response',
                text_col: str = 'text') -> Dict[str, Any]:
        """
        Run comprehensive evaluation using both evaluators.
        
        Args:
            data: Either a DataFrame or path to input parquet file
            output_file: Optional path to save results
            response_col: Column name for generated responses
            text_col: Column name for input text
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Load data if necessary
        if isinstance(data, str):
            df = pd.read_parquet(data)
            # Run LLM-based evaluation with file
            llm_metrics = self.llm_evaluator.evaluate_from_file(data)
        else:
            df = data
            # Run LLM-based evaluation with DataFrame
            llm_metrics = self.llm_evaluator.evaluate_from_dataframe(df)
        
        # Run hallucination evaluation
        hallucination_metrics = self.hallucination_evaluator.evaluate(
            data=df,
            response_col=response_col,
            text_col=text_col
        )
        
        # Combine metrics
        combined_metrics = {
            "llm_metrics": llm_metrics.to_dict('records')[0],
            "hallucination_metrics": asdict(hallucination_metrics)
        }
        
        # Calculate overall RAG score as average of all metrics
        all_metrics = []
        # Add LLM metrics
        all_metrics.extend([
            combined_metrics['llm_metrics']['language_quality_index'],
            combined_metrics['llm_metrics']['reasoning_quality_index'],
            combined_metrics['llm_metrics']['query_adherence_index']
        ])
        # Add hallucination metrics
        all_metrics.extend([
            combined_metrics['hallucination_metrics']['non_hallucinated_citation'],
            combined_metrics['hallucination_metrics']['valid_quote'],
            combined_metrics['hallucination_metrics']['valid_identifier'],
            combined_metrics['hallucination_metrics']['unduplicated_quote']
        ])
        
        # Calculate overall score as simple average
        combined_metrics['overall_rag_score'] = sum(all_metrics) / len(all_metrics)
        
        # Save results if output file specified
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            pd.DataFrame([combined_metrics]).to_parquet(output_file)
            
        return combined_metrics

def main():
    parser = argparse.ArgumentParser(description='RAG System Evaluation Pipeline')
    parser.add_argument('--input', required=True, help='Input parquet file path')
    parser.add_argument('--output', help='Output file path for results')
    parser.add_argument('--model-path', required=True, help='Path to LLM evaluation model')
    parser.add_argument('--max-model-len', type=int, default=8192, help='Maximum model length')
    parser.add_argument('--response-col', default='generated_response', help='Response column name')
    parser.add_argument('--text-col', default='text', help='Text column name')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = RAGEvaluationPipeline(
        model_path=args.model_path,
        max_model_len=args.max_model_len
    )
    
    results = pipeline.evaluate(
        data=args.input,
        output_file=args.output,
        response_col=args.response_col,
        text_col=args.text_col
    )
    
    # Print results summary
    print("\nEvaluation Results Summary:")
    print("-" * 50)
    print("\nLLM-based Metrics:")
    for key, value in results['llm_metrics'].items():
        print(f"{key}: {value:.3f}")
    
    print("\nHallucination Metrics:")
    for key, value in results['hallucination_metrics'].items():
        print(f"{key}: {value:.3f}")
    
    print("\nOverall RAG Score:", f"{results['overall_rag_score']:.3f}")

if __name__ == "__main__":
    main()