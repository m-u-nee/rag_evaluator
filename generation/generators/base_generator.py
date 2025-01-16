from abc import ABC, abstractmethod
import pandas as pd
from vllm import SamplingParams

class BaseGenerator(ABC):
    def __init__(self, model_path: str, input_path: str, num_rows: int = -1):
        """
        Initialize base generator.

        Args:
            model_path: Path to model weights
            input_path: Path to input parquet file
            num_rows: Number of rows to process (-1 for all)
        """
        self.model_path = model_path
        self.input_path = input_path
        self.num_rows = num_rows
    
    @abstractmethod
    def get_sampling_params(self) -> SamplingParams:
        """Get sampling parameters for the model."""
        pass
    
    @abstractmethod
    def prepare_prompts(self, df: pd.DataFrame) -> list:
        """Prepare prompts for generation."""
        pass
    
    @abstractmethod
    def generate(self) -> pd.DataFrame:
        """Generate responses and return DataFrame with results."""
        pass