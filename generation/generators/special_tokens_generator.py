import pandas as pd
from vllm import LLM, SamplingParams
from .base_generator import BaseGenerator

class SpecialTokensGenerator(BaseGenerator):
   def __init__(self, model_path: str, input_path: str, num_rows: int = -1):
       super().__init__(model_path, input_path, num_rows)
   
   def get_sampling_params(self) -> SamplingParams:
       """Get sampling parameters for special tokens models."""
       return SamplingParams(
           temperature=0.0,
           top_p=0.95,
           max_tokens=2500,
           repetition_penalty=1,
           stop=["#END#"],
           skip_special_tokens=False,
       )

   def prepare_prompts(self, df: pd.DataFrame) -> list:
       """Prepare prompts for special tokens models."""
       return df['text'].tolist()

   def generate(self) -> pd.DataFrame:
       """Generate responses using special tokens approach."""
       # Load input data
       df = pd.read_parquet(self.input_path)
       df = df.head(self.num_rows) if self.num_rows > 0 else df

       # Initialize LLM
       llm = LLM(
           model=self.model_path,
           trust_remote_code=True,
       )

       # Prepare prompts and sampling parameters
       prompts = self.prepare_prompts(df)
       sampling_params = self.get_sampling_params()

       # Generate responses
       outputs = llm.generate(prompts, sampling_params)
       generated_texts = [output.outputs[0].text for output in outputs]

       # Create output dataframe
       output_df = df.copy()
       output_df['generated_response'] = generated_texts

       return output_df