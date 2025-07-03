import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import logging
import json
import random
import math
import argparse
from multiprocessing import Pool

from core.llm_engine import *

class RandomThoughtGenerator(OpenLMEngine):
    def __init__(self,
                 model_name: str,
                 nick_name: str,
                 tokenizer_name: str,
                 dataset_size: int = 1000,
                 results_dir: str = None,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.85,
                 dtype: str = "bfloat16",
                 max_tokens: int = 16384,
                 temperature: float = 0.6,
                 top_p: float = 0.9,
                 top_k: int = 32,
                 max_num_batched_tokens: int = 16384,
                 **kwargs
                 ):
        self.nick_name = nick_name
        self.results_dir = results_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.dataset_size = dataset_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_num_batched_tokens = max_num_batched_tokens

        self.output_dir = os.path.join(self.results_dir, "random_thoughts")
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize model config
        config = ModelConfig(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            dtype=self.dtype,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_num_batched_tokens=self.max_num_batched_tokens
        )

        super().__init__(config=config)

        print(f"Start generating random thoughts: {self.nick_name}")
    
    def generate_random_thoughts(self, min_tokens: int = 2048) -> None:
        try:
            template = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "Let's think about math problem"}], tokenize=False, add_generation_prompt=True
            )
            prompts = self.dataset_size * ["Let's think about math problem<think>", template + "<think>\n"]
            self.response = self.generate(prompts=prompts).rename(columns = {'response': 'random_thought'})
            self.response["prompt"] = prompts

            self.response["long_cot"] = self.response["random_thought"]. \
            apply(lambda x: "</think>" in x and len(self.tokenizer.encode(x.split("</think>")[0])) > min_tokens)

            # Save results
            output_path = os.path.join(self.output_dir, f"{self.nick_name}.pickle")
            self.response.to_pickle(output_path)
            
        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            raise

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parse Arguments for Reasoner QA evaluation")

    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--nick_name", type=str, required=True, help="Nickname for the model")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name of the tokenizer to use")
    parser.add_argument("--results_dir", type=str, required=True, help="Name of the dataset to evaluate on")
    parser.add_argument("--dataset_size", type=int, default=1000, help="Number of random thoughts to generate")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.4,
                        help="Fraction of GPU memory to allocate")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Data type for model weights (e.g., bfloat16, float16)")
    parser.add_argument("--max_tokens", type=int, default=16384,
                        help="Maximum number of output tokens")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus sampling parameter")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-k sampling parameter")
    parser.add_argument("--max_num_batched_tokens", type=int, default=16384,
                        help="Maximum number of tokens to batch")
    
    args = parser.parse_args()
    
    engine = RandomThoughtGenerator(**vars(args))
    engine.generate_random_thoughts()