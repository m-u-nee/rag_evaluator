from typing import Optional
import pandas as pd
import os
from .generators.base_generator import BaseGenerator
from .generators.vllm_generator import VLLMGenerator
from .generators.special_tokens_generator import SpecialTokensGenerator

class RAGGenerator:
    def __init__(
        self,
        generator_type: str,  # Must be either 'vllm' or 'special_tokens'
        input_path: str,
        model_path: str,
        num_rows: Optional[int] = None,
    ):
        """
        Initialize RAG Generator.

        Args:
            generator_type: Type of generator ('vllm' or 'special_tokens')
            input_path: Path to input parquet file
            model_path: Path to model weights
            num_rows: Number of rows to process (None for all)
        """
        if generator_type not in ['vllm', 'special_tokens']:
            raise ValueError("generator_type must be either 'vllm' or 'special_tokens'")
            
        self.generator_type = generator_type
        self.input_path = input_path
        self.model_path = model_path
        self.num_rows = num_rows if num_rows is not None else -1
        self.generator = self._initialize_generator()

    def _initialize_generator(self):
        """Initialize the appropriate generator based on type."""
        generators = {
            'vllm': VLLMGenerator,
            'special_tokens': SpecialTokensGenerator,
        }

        return generators[self.generator_type](
            model_path=self.model_path,
            input_path=self.input_path,
            num_rows=self.num_rows
        )

    def generate(self) -> pd.DataFrame:
        """
        Generate RAG outputs.

        Returns:
            DataFrame containing generated responses
        """
        return self.generator.generate()

    def save_outputs(self, df: pd.DataFrame, output_dir: str):
        """
        Save the generation outputs to disk.

        Args:
            df: DataFrame containing the generations
            output_dir: Directory where outputs should be saved
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save parquet
        output_parquet_path = os.path.join(output_dir, 'generations.parquet')
        df.to_parquet(output_parquet_path)
        
        # Save readable samples
        output_txt_path = os.path.join(output_dir, 'sample_generations.txt')
        self._save_readable_format(df, output_txt_path)

    def _save_readable_format(self, df: pd.DataFrame, output_path: str):
        """Save first 10 rows in readable format."""
        sample_df = df.head(10)
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, row in sample_df.iterrows():
                f.write(f"=== Sample {idx + 1} ===\n\n")
                f.write("Input Text:\n")
                f.write(f"{row['text']}\n\n")
                f.write("Generated Response:\n")
                f.write(f"{row['generated_response']}\n\n")
                f.write("-" * 80 + "\n\n")