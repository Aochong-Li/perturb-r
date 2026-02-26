import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from core.llm_engine import *
from core.openai_engine import *

from reward_score.math500 import math_if_boxed
from utils.chunk_r import equal_chunk
from utils.corrupt_num import *
from more_itertools import chunked

BUFFER_TOKENS = 8192

class InjectDistractorAblation(OpenLMEngine):
    def __init__(
        self,
        model_name: str,
        nick_name: str,
        tokenizer_name: str,
        results_dir: str,
        sample_size: int,
        num_distract_candidates: int,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        dtype: str = "bfloat16",
        max_tokens: int = 32768,
        temperature: float = 0.6,
        top_p: float = 1.0,
        top_k: int = -1,
        max_num_batched_tokens: int = 8192,
        mini_batch_size: int = None,
        granularity: int = 30,
        unit: float = 0.2,
        overwrite: bool = False,
        client_name: str = "",
        **kwargs,
    ):
        # Initialize attributes first
        self.model_name = model_name
        self.nick_name = nick_name
        self.tokenizer_name = tokenizer_name
        self.results_dir = results_dir
        self.sample_size = sample_size
        self.num_distract_candidates = num_distract_candidates
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_num_batched_tokens = max_num_batched_tokens
        self.mini_batch_size = mini_batch_size
        self.granularity = granularity
        self.unit = unit
        self.overwrite = overwrite
        self.client_name = client_name

        # Create output directory if it doesn't exist
        self.output_dir = os.path.join(self.results_dir, "inject_distractor_head_ablation")
        os.makedirs(self.output_dir, exist_ok=True)
        
        out_pickle = os.path.join(self.output_dir, f"{self.nick_name}.pickle")
        if os.path.exists(out_pickle) and not self.overwrite:
            print(f"Stress test (Inject Distractor) already exists: {self.nick_name}")
            exit()
        
        cfg = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        self.max_position_embeddings = cfg.max_position_embeddings

        if self.client_name == '':    
            # Initialize model config
            config = ModelConfig(
                model_name=self.model_name,
                tokenizer_name=self.tokenizer_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype=self.dtype,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_num_batched_tokens=self.max_num_batched_tokens,
                max_model_len=self.max_position_embeddings
            )
            # Initialize parent class
            super().__init__(config=config)
            
        self.load_dataset()
    
        print(f"Start stress testing: {self.nick_name} on distractor injection with original head")

    def load_dataset(self) -> None:
        self.dataset_path = os.path.join(self.results_dir, "inject_distractor", f"{self.nick_name}.pickle")
        self.df = pd.read_pickle(self.dataset_path)
        columns = ['problem', 'solution', 'source', 'original_response', 'solve_n', 'original_ratio', 'distractor_ratio',
                   'distractor_problem', 'distractor_solution', 'distractor_reasoning_chunks','distractor_source', 'distractor_solve_n']
        self.df = self.df[columns]
        self.df = self.df[self.df['distractor_ratio'] == 0.2][columns]
        self.df['distractor_response'] = self.df['distractor_reasoning_chunks'].apply(lambda x: "".join(x))
        self.df = self.df.drop(columns=['distractor_reasoning_chunks'])
        
        self.select_distractor()

    def select_distractor(self) -> None:
        self.df["reasoning_chunks"] = self.df["original_response"].apply(lambda x: x.replace("<think>", "").rstrip("\n"))
        self.df["reasoning_chunks"] = self.df["reasoning_chunks"].apply(lambda x: equal_chunk(x, self.granularity, keep_first_paragraph=True))
        
        self.df['distractor_reasoning_chunks'] = self.df['distractor_response'].apply(lambda x: x.replace("<think>", "").rstrip("\n"))
        self.df['distractor_reasoning_chunks'] = self.df['distractor_reasoning_chunks'].apply(lambda x: equal_chunk(x, self.granularity, keep_first_paragraph=True))
        self.df["distractor_reasoning_chunks"] = self.df["distractor_reasoning_chunks"].apply(lambda x: x[1:]) # drop the first paragraph
        
    def generate_distract_reasoning(self, row):
        original_ratio = row["original_ratio"]
        distractor_ratio = row["distractor_ratio"]
        
        original_chunks = row["reasoning_chunks"]
        distractor_chunks = row["distractor_reasoning_chunks"]

        n_orig = max(int(len(original_chunks) * original_ratio), 1)
        n_dist = int(len(distractor_chunks) * distractor_ratio)

        original_reasoning = "".join(original_chunks[:n_orig])
        distractor_reasoning = "".join(distractor_chunks[:n_dist])

        return original_reasoning + "Let me think." + distractor_reasoning
        
    def distract(self):
        self.df["reasoning_w_distractor"] = self.df.apply(self.generate_distract_reasoning, axis=1)

    def local_eval(self) -> None:
        self.responses = []
        self.mini_batch_size = self.df.shape[0] if self.mini_batch_size is None else self.mini_batch_size
        for batch in chunked(list(self.df["prompt"]), self.mini_batch_size):
            new_sampling_params = [
                {
                    "max_tokens": min(self.max_tokens, self.max_position_embeddings - len(self.tokenizer.encode(prompt)) - 1)
                }
                for prompt in batch
            ]
            out = self.generate(prompts=batch, new_sampling_params=new_sampling_params)
            self.responses.append(out)
        self.response = pd.concat(self.responses, ignore_index=True).rename(columns={'response': 'post_distraction_response'})
        self.response.index = self.df.index
        self.df = pd.concat([self.df, self.response], axis=1)

    def api_eval(self) -> None:
        os.makedirs(self.output_dir + f"/api", exist_ok=True)

        engine = OpenAI_Engine(
            input_df=self.df,
            nick_name=f"distract_{self.nick_name}",
            batch_io_root=str(Path.home()) + "/research/openai_batch_io/reasoning",
            cache_filepath=os.path.join(self.output_dir, f"api/{self.nick_name}_api_responses.pickle"),
            model=self.model_name,
            client_name=self.client_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            mode="completions"
        )
        engine.run_model(overwrite=self.overwrite)
        self.response = engine.retrieve_outputs(overwrite=self.overwrite)
        self.response = self.response.explode(['response']).set_index('idx').rename(columns={'response': 'post_distraction_response'})
        self.df = self.df.merge(self.response, left_index=True, right_index=True)

    def eval(self) -> None:
        template_prefix, template_suffix = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "HANDLE"}],
            tokenize=False,
            add_generation_prompt=True
        ).split("HANDLE")
        
        self.df["prompt"] = template_prefix + self.df['problem'] + template_suffix + self.df['reasoning_w_distractor']
        self.df = self.df.drop(columns = ['reasoning_w_distractor'])
        
        if self.client_name == '':
            self.local_eval()
        else:
            self.api_eval()
        
        output_path = os.path.join(self.output_dir, f"{self.nick_name}.pickle")
        self.df.to_pickle(output_path)

        self.result_df = self.df.copy()
        self.result_df['pred'] = self.result_df['post_distraction_response'].apply(lambda x: x.split('</think>')[-1].strip() if '</think>' in x else x)
        self.result_df['ground_truth'] = self.result_df['solution']
        self.result_df['if_boxed'] = self.result_df['pred'].apply(math_if_boxed)
        
        self.result_df.to_pickle(output_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parse Arguments for Reasoner QA evaluation")

    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--nick_name", type=str, required=True, help="Nickname for the model")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name of the tokenizer to use")
    parser.add_argument("--results_dir", type=str, default='/share/goyal/lio/reasoning/eval/', 
                       help="Directory to save evaluation results")
    parser.add_argument("--sample_size", type=int, default=100,
                        help="Number of problems to sample for the stress test")
    parser.add_argument("--num_distract_candidates", type=int, default=50,
                        help="Number of problems to use as distractors")
    
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                        help="Fraction of GPU memory to allocate")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Data type for model weights (e.g., bfloat16, float16)")
    parser.add_argument("--max_tokens", type=int, default=32768,
                        help="Maximum number of output tokens")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus sampling parameter")
    parser.add_argument("--top_k", type=int, default=-1,
                        help="Top-k sampling parameter")
    parser.add_argument("--max_num_batched_tokens", type=int, default=8192,
                        help="Maximum number of tokens in a batch")
    
    parser.add_argument("--mini_batch_size", type=int, default=None,
                        help="Mini batch size for generation")
    parser.add_argument("--granularity", type=int, default=30,
                        help="Granularity of the thinking chunks")
    parser.add_argument("--unit", type=float, default=0.2,
                        help="Unit of the thinking chunks")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results")
    parser.add_argument("--client_name", type=str, default="",
                        help="Name of the client (for OpenAI or other APIs)")
    args = parser.parse_args()
    
    engine = InjectDistractorAblation(
        **vars(args),
    )

    engine.distract()
    engine.eval()