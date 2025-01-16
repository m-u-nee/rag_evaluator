import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from .base_generator import BaseGenerator

class VLLMGenerator(BaseGenerator):
   def __init__(self, model_path: str, input_path: str, num_rows: int = -1):
       super().__init__(model_path, input_path, num_rows)
       self.tokenizer = AutoTokenizer.from_pretrained(model_path)
       
   def get_sampling_params(self) -> SamplingParams:
       """Get sampling parameters for VLLM models."""
       return SamplingParams(
           temperature=0.6,
           top_p=0.9,
           max_tokens=2500,
           stop=[self.tokenizer.eos_token, "<|eot_id|>"]
       )
   
   def prepare_prompts(self, df: pd.DataFrame) -> list:
       """Prepare prompts using chat template."""
       prompts = []
       for text in df['text']:
           messages = [
               {"role": "user", "content": text}
           ]
           prompt = self.tokenizer.apply_chat_template(
               messages,
               add_generation_prompt=True,
               tokenize=False
           )
           prompts.append(prompt)
       return prompts
   
   def generate(self) -> pd.DataFrame:
       """Generate responses using VLLM."""
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