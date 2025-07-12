import os
import pandas as pd

from core.llm_engine import *
import argparse
from datasets import load_dataset, load_from_disk
from reward_score.math import math_compute_score, math_if_answer
from reward_score.countdown import compute_score as countdown_compute_score, extract_solution as countdown_extract_solution

import numpy as np

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == "true":
        return True
    elif v.lower() == "false":
        return False

class BenchmarkEval(OpenLMEngine):
    def __init__(self,
                 model_name: str,
                 nick_name: str,
                 tokenizer_name: str,
                 dataset_name_or_path: str = None,
                 subset_name: str = None,
                 split_name: str = 'test',
                 sample_size: int = None,
                 output_dir: str = './results/hendrycks_math/sample200/benchmark_eval',
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.85,
                 dtype: str = "bfloat16",
                 max_tokens: int = 16384,
                 temperature: float = 0.7,
                 top_p: float = 1.0,
                 top_k: int = 0,
                 sample_k: int = 1,
                 enable_thinking: bool = True,
                 max_num_batched_tokens: int = 32768,
                 overwrite: bool = False
                 ):

        # Initialize attributes first
        self.nick_name = nick_name
        self.output_dir = output_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.sample_k = sample_k
        self.enable_thinking = enable_thinking
        self.overwrite = overwrite
        self.max_num_batched_tokens = max_num_batched_tokens

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        if not self.overwrite:
            if os.path.exists(os.path.join(self.output_dir, f"{self.nick_name}{'_nothinking' if not self.enable_thinking else ''}.pickle")):
                print(f"Results already exist for {self.nick_name} with enable_thinking={self.enable_thinking}")
                exit()
        elif "qwen3" not in self.nick_name.lower() and not self.enable_thinking:
            print(f"No-Thinking mode is not supported for {self.nick_name}")
            exit()
        
        # Load dataset
        self.load_dataset(
            dataset_name_or_path,
            subset_name,
            split_name,
            sample_size
        )
    
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
            n = self.sample_k,
            max_num_batched_tokens=self.max_num_batched_tokens
        )

        # Initialize parent class
        super().__init__(config=config)

        print(f"Start evaluating {self.nick_name} on dataset: {dataset_name_or_path} | subset: {subset_name} | split: {split_name} | avg@{self.sample_k}")

    def load_dataset(self, dataset_name: str, subset_name: str, split_name: str, sample_size: int) -> None:
        """Load dataset from HuggingFace or local disk.
        
        Args:
            dataset_name: Name of the dataset or path to local dataset
            subset_name: Subset name if applicable
            split_name: Split name (e.g., 'train', 'test')
            sample_size: Number of samples to use, if None use all
        """
        try:
            dataset = load_from_disk(dataset_name)[split_name]
        except Exception as e:
            print(f"Error loading from disk: {e}")
            try:
                dataset = load_dataset(dataset_name, subset_name)[split_name]
            except Exception as e:
                raise RuntimeError(f"Failed to load dataset from Hugging Face: {e}")
        if sample_size:
            self.df = pd.DataFrame(dataset).sample(n=sample_size, random_state=45).reset_index(drop = True)
        else:
            self.df = pd.DataFrame(dataset)

    def apply_chat_template (self, question: str):
        chat_history = [
            {'role': 'user', 'content': question}
        ]
        if not self.enable_thinking:
            tokenized_prompt = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize = False,
                add_generation_prompt = True,
                enable_thinking=self.enable_thinking
            )
        else:
            tokenized_prompt = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize = False,
                add_generation_prompt = True,
            )
        
        return tokenized_prompt
    
    def eval(self) -> None:
        """Evaluate the model on the dataset and save results.
        
        This method:
        1. Applies chat template to problems
        2. Generates model responses
        3. Computes correctness scores
        4. Saves results to disk
        """
        try:
            # Apply chat template to problems
            prompts = self.df['problem'].apply(self.apply_chat_template)

            # Generate model responses
            self.response = self.generate(prompts=prompts)
            self.df = self.df.loc[np.repeat(self.df.index, self.sample_k)].reset_index(drop=True)

            self.response.index = self.df.index
            self.df = pd.concat([self.df, self.response], axis=1)
            
            # Compute correctness scores
            correctness, check_if_answer = [], []
            for idx, row in self.df.iterrows():
                solution = row['response']
                if "</think>" in solution:
                    solution = solution.split("</think>")[1]
                task = row.get('task', 'math')

                if task == 'math':
                    ground_truth = self.df.loc[idx, 'solution']
                    score = math_compute_score(solution, ground_truth)
                    if_answer = math_if_answer(solution)

                elif task == 'countdown':
                    numbers = self.df.loc[idx, 'nums']
                    target = self.df.loc[idx, 'target']
                    score = countdown_compute_score(solution, numbers, target)
                    if_answer = countdown_extract_solution(solution) is not None

                correctness.append(score)
                check_if_answer.append(if_answer)
            
            # Combine results
            self.df['correct'] = correctness
            self.df['if_answer'] = check_if_answer
            
            # Save results
            output_path = os.path.join(self.output_dir, f"{self.nick_name}{'_nothinking' if not self.enable_thinking else ''}.pickle")
            self.df.to_pickle(output_path)
         
            # Log summary statistics
            accuracy = sum(correctness) / len(correctness)
            print(f"Evaluation complete. Accuracy: {accuracy:.2%}")
            
        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            raise

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parse Arguments for Reasoner QA evaluation")

    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--nick_name", type=str, required=True, help="Nickname for the model")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name of the tokenizer to use")
    parser.add_argument("--dataset_name_or_path", type=str, required=True, help="Name of the dataset to evaluate on")
    parser.add_argument("--subset_name", type=str, default=None, help="Name of the dataset subset (default: None)")
    parser.add_argument("--split_name", type=str, default='test', help="Dataset split to use (default: test)")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of samples to use (default: None)")
    parser.add_argument("--output_dir", type=str, default='/share/goyal/lio/reasoning/eval/', 
                       help="Directory to save evaluation results")
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.75,
                        help="Fraction of GPU memory to allocate")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Data type for model weights (e.g., bfloat16, float16)")
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="Maximum number of output tokens")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus sampling parameter")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-k sampling parameter")
    parser.add_argument("--sample_k", type=int, default=1,
                        help="Sample@k parameter")
    parser.add_argument("--overwrite", type=str2bool, default=False,
                        help="Overwrite existing results")
    parser.add_argument("--enable_thinking", type=str2bool, default=True,
                        help="Enable thinking")
    parser.add_argument("--max_num_batched_tokens", type=int, default=32768,
                        help="Maximum number of tokens to batch")

    args = parser.parse_args()

    engine = BenchmarkEval(
        **vars(args),
    )
    engine.eval()
