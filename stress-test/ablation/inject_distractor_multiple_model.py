import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from core.llm_engine import *
from core.openai_engine import *

from reward_score.math_eval import math_if_boxed
from more_itertools import chunked

BUFFER_TOKENS = 8192
MIN_THINKING_TOKENS = 4096

class InjectDistractorCrossModel(OpenLMEngine):
    def __init__(
        self,
        model_name: str,
        nick_name: str,
        tokenizer_name: str,
        results_dir: str,
        num_distractor_models: int = 5,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        dtype: str = "bfloat16",
        max_tokens: int = 32768,
        temperature: float = 0.6,
        top_p: float = 1.0,
        top_k: int = -1,
        max_num_batched_tokens: int = 8192,
        mini_batch_size: int = None,
        overwrite: bool = False,
        client_name: str = "",
        **kwargs,
    ):
        # Initialize attributes first
        self.model_name = model_name
        self.nick_name = nick_name
        self.tokenizer_name = tokenizer_name
        self.results_dir = results_dir
        self.num_distractor_models = num_distractor_models
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_num_batched_tokens = max_num_batched_tokens
        self.mini_batch_size = mini_batch_size
        self.overwrite = overwrite
        self.client_name = client_name

        self.output_dir = os.path.join(self.results_dir, "inject_distractor_multiple_model")
        os.makedirs(self.output_dir, exist_ok=True)

        out_pickle = os.path.join(self.output_dir, f"{self.nick_name}.pickle")
        if os.path.exists(out_pickle) and not self.overwrite:
            print(f"Stress test (Inject Distractor Cross Model) already exists: {self.nick_name}")
            exit()

        cfg = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        self.max_position_embeddings = cfg.max_position_embeddings

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
        super().__init__(config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        self.load_dataset()

        print(f"Start stress testing: {self.nick_name} on cross-model distractor injection")

    def load_dataset(self) -> None:
        """Load the test model's inject_distractor resuclts"""
        self.dataset_path = os.path.join(self.results_dir, "inject_distractor", f"{self.nick_name}.pickle")

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        self.df = pd.read_pickle(self.dataset_path)
        print(f"Loaded {len(self.df)} rows from {self.nick_name}")

        # Drop columns that we'll replace with cross-model distractors
        columns_to_drop = [
            'distractor_ratio', 'distractor_problem', 'distractor_solution',
            'distractor_source', 'distractor_reasoning_chunks', 'distractor_solve_n',
            'prompt', 'post_distraction_response', 'pred', 'ground_truth',
            'if_boxed', 'model_is_correct', 'model_judge'
        ]

        columns_to_drop = [col for col in columns_to_drop if col in self.df.columns]
        self.df = self.df.drop(columns=columns_to_drop)

        self.load_distractor_models()

    def load_distractor_models(self) -> None:
        """Load pickle files from other models - keep their distractor columns directly"""
        inject_distractor_dir = os.path.join(self.results_dir, "inject_distractor")

        all_pickles = [f for f in os.listdir(inject_distractor_dir) if f.endswith('.pickle')]
        other_pickles = [f for f in all_pickles if f != f"{self.nick_name}.pickle"]

        if len(other_pickles) < self.num_distractor_models:
            print(f"Warning: Only {len(other_pickles)} other models available, using all of them")
            self.num_distractor_models = len(other_pickles)

        np.random.seed(hash(self.nick_name) % (2**32))
        selected_pickles = np.random.choice(other_pickles, size=self.num_distractor_models, replace=False)

        print(f"Selected {self.num_distractor_models} distractor models:")
        for p in selected_pickles:
            print(f"  - {p}")

        distractor_cols = [
            'distractor_ratio', 'distractor_problem', 'distractor_solution',
            'distractor_source', 'distractor_reasoning_chunks', 'distractor_solve_n'
        ]

        all_distractors = []
        for pickle_file in selected_pickles:
            model_name = pickle_file.replace('.pickle', '')
            df = pd.read_pickle(os.path.join(inject_distractor_dir, pickle_file))

            cols_to_keep = [col for col in distractor_cols if col in df.columns]
            if len(cols_to_keep) > 0:
                df_distractors = df[cols_to_keep].copy()
                df_distractors['distractor_model'] = model_name
                all_distractors.append(df_distractors)
                print(f"  - Loaded {len(df)} distractor rows from {model_name}")

        self.distractor_pool = pd.concat(all_distractors, ignore_index=True)
        self.distractor_pool = self.distractor_pool.drop_duplicates(subset=['distractor_problem', 'distractor_model']).reset_index(drop=True)
        print(f"\nTotal distractor pool size: {len(self.distractor_pool)} rows")

    def assign_distractors(self):
        """Randomly pair each row in test df with a distractor from the pool"""
        np.random.seed(42)
        sampled_distractors = self.distractor_pool.sample(n=len(self.df), replace=True, random_state=42).reset_index(drop=True)
        self.df = pd.concat([self.df.reset_index(drop=True), sampled_distractors], axis=1)

    def generate_distract_reasoning(self, row):
        """Generate distracted reasoning following the same logic as original"""
        original_ratio = row["original_ratio"]
        distractor_ratio = row["distractor_ratio"]

        original_chunks = row["reasoning_chunks"]
        distractor_chunks = row["distractor_reasoning_chunks"]

        n_orig = int(len(original_chunks) * original_ratio)
        n_dist = int(len(distractor_chunks) * distractor_ratio)

        original_reasoning = "".join(original_chunks[:n_orig])
        distractor_reasoning = "".join(distractor_chunks[:n_dist])

        if original_ratio > 0.0:
            original_reasoning = original_reasoning + "\nLet me think."
            distractor_reasoning = distractor_reasoning.replace("<think>", "").rstrip("\n")

        return original_reasoning + distractor_reasoning

    def assemble_prompt(self):
        """Assemble prompts with cross-model distractors"""
        template_prefix, template_suffix = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "HANDLE"}],
            tokenize=False,
            add_generation_prompt=True
        ).split("HANDLE")

        self.df["reasoning_w_distractor"] = self.df.apply(self.generate_distract_reasoning, axis=1)

        self.df["prompt"] = template_prefix + self.df['problem'] + template_suffix + self.df['reasoning_w_distractor']
        self.df = self.df.drop(columns=['reasoning_w_distractor'])

        print(f"Assembled {len(self.df)} prompts with cross-model distractors")

    def local_eval(self) -> None:
        self.responses = []
        self.mini_batch_size = self.df.shape[0] if self.mini_batch_size is None else self.mini_batch_size
        for batch in chunked(list(self.df["prompt"]), self.mini_batch_size):
            sampling_overrides = [
                {
                    "max_tokens": min(self.max_tokens, self.max_position_embeddings - len(self.tokenizer.encode(prompt)) - 1)
                }
                for prompt in batch
            ]
            out = self.generate(prompts=batch, sampling_overrides=sampling_overrides)
            self.responses.append(out)
        self.response = pd.concat(self.responses, ignore_index=True).rename(columns={'response': 'post_distraction_response'})
        self.response.index = self.df.index
        self.df = pd.concat([self.df, self.response], axis=1)

    def eval(self) -> None:
        self.assemble_prompt()

        self.local_eval()

        output_path = os.path.join(self.output_dir, f"{self.nick_name}.pickle")
        self.df.to_pickle(output_path)

        self.result_df = self.df.copy()
        self.result_df['pred'] = self.result_df['post_distraction_response'].apply(
            lambda x: x.split('</think>')[-1].strip() if '</think>' in x else x
        )
        self.result_df['ground_truth'] = self.result_df['solution']

        self.result_df.to_pickle(output_path)
        print(f"Results saved to {output_path}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Cross-model distractor injection ablation study")

    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--nick_name", type=str, required=True, help="Nickname for the model")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name of the tokenizer to use")
    parser.add_argument("--results_dir", type=str, default='/share/goyal/lio/reasoning/eval/',
                       help="Directory containing inject_distractor results")
    parser.add_argument("--num_distractor_models", type=int, default=5,
                        help="Number of different models to use as distractor sources")

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
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results")
    parser.add_argument("--client_name", type=str, default="",
                        help="Name of the client (for OpenAI or other APIs)")

    args = parser.parse_args()
    
    engine = InjectDistractorCrossModel(
        **vars(args),
    )
    engine.assign_distractors()
    engine.eval()
