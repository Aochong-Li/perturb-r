import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import logging
import json
import random
import argparse

from core.llm_engine import *

from reward_score.math import compute_score as math_compute_score
from reward_score.countdown import compute_score as countdown_compute_score, extract_solution

from utils.chunk_r import equal_chunk
from utils.corrupt_num import *

class CorruptNumbers(OpenLMEngine):
    def __init__(self,
                 task: str,
                 model_name: str,
                 nick_name: str,
                 tokenizer_name: str,
                 results_dir: str = None,
                 question_ids_fname: str = None,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.85,
                 dtype: str = "bfloat16",
                 max_tokens: int = 16384,
                 temperature: float = 0.6,
                 top_p: float = 0.9,
                 top_k: int = 32,
                 granularity: int = 20,
                 overwrite: bool = False,
                 generate_data_only: bool = False,
                 **kwargs
                 ):
        # Initialize attributes first
        self.task = task
        self.nick_name = nick_name
        self.results_dir = results_dir
        self.question_ids_fname = question_ids_fname
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.granularity = granularity

        # Create output directory if it doesn't exist
        self.output_dir = os.path.join(self.results_dir, "corrupt_numbers")
        os.makedirs(self.output_dir, exist_ok=True)

        if os.path.exists(os.path.join(self.output_dir, f"{self.nick_name}.pickle")) and not overwrite:
            print(f"Stress test (Corrupt Numbers) already exists: {self.nick_name}")
            exit()
        
        # Load dataset from pickle file
        self.load_dataset()
    
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
            top_k=self.top_k
        )

        if generate_data_only:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            super().__init__(config=config)

        print(f"Start stress testing: {self.nick_name} on corrupted digit")

    def load_dataset(self) -> None:
        """Load dataset from pickle file.
        
        Args:
            dataset_path: Path to the pickle file containing the dataset
        """
        self.dataset_path = os.path.join(self.results_dir, "benchmark", f"{self.nick_name}.pickle")
        question_ids = json.load(open(os.path.join(self.results_dir, self.question_ids_fname)))

        try:
            if self.task == "math":
                self.compute_score = math_compute_score
                self.df = pd.read_pickle(self.dataset_path)
                self.df = self.df[
                    self.df['unique_id'].isin(question_ids) & 
                    (self.df['correct'] == 1) & 
                    (self.df["response"].str.contains("</think>"))
                ]
                self.df = self.df.drop_duplicates(subset = ["unique_id"]).reset_index(drop = True)
                self.df = self.df.drop(columns = ['correct']).rename(columns = {'response': 'original_response'})
            elif self.task == "countdown":
                self.compute_score = countdown_compute_score
                self.df = pd.read_pickle(self.dataset_path)
                self.df = self.df[self.df['problem'].isin(question_ids)].reset_index(drop = True)
                self.df = self.df.drop(columns = ['correct']).rename(columns = {'response': 'original_response'})
            
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from pickle file: {e}")

    def corrupt_thinking(self, unit = 0.2, seed: int = None) -> None:
        rng = random.Random(seed)
        
        self.df["reasoning_chunks"] = self.df["original_response"].apply(self.process_response)
        self.df = self.df.dropna(how='any').reset_index(drop = True)
        
        # generate all possible start and end positions
        pos = np.arange(0, 1, unit)
        windows = [(round(p, 2), round(p + unit, 2)) for p in pos]
        
        def sample_window (row, windows=windows):
            chunks = row["reasoning_chunks"]
            n = len(chunks)
            
            res = {k: [] for k in ["start_pos",
                                   "end_pos",
                                   "prefix_reasoning",
                                   "corrupted_reasoning",
                                   "num_prefix_tokens",
                                   "num_corrupted_tokens"]}
            
            for start, end in windows:
                start_chunk_idx = max(0, int(start * n))
                end_chunk_idx = max(start_chunk_idx + 1, int(end * n))
                end_chunk_idx = min(end_chunk_idx, n)

                prefix_txt = '\n\n'.join(chunks[:start_chunk_idx])
                corrupted_txt = '\n\n'.join(chunks[start_chunk_idx:end_chunk_idx])
                num_prefix_tokens = len(self.tokenizer.encode(prefix_txt, add_special_tokens=False))
                num_corrupted_tokens = len(self.tokenizer.encode(corrupted_txt, add_special_tokens=False))
                
                res["start_pos"].append(start)
                res["end_pos"].append(end)
                res["prefix_reasoning"].append(prefix_txt)
                res["corrupted_reasoning"].append(corrupted_txt)
                res["num_prefix_tokens"].append(num_prefix_tokens)
                res["num_corrupted_tokens"].append(num_corrupted_tokens)

            return pd.Series(res, dtype = object)

        self.df[["start_pos", "end_pos",
                 "prefix_reasoning", "corrupted_reasoning",
                 "num_prefix_tokens", "num_corrupted_tokens"]] = self.df.apply(sample_window, axis=1)
        
        self.df = self.df.explode(
            column = ["start_pos", "end_pos",
                      "prefix_reasoning", "corrupted_reasoning",
                      "num_prefix_tokens", "num_corrupted_tokens"],
                      ignore_index = True
                      )
        
        self.corrupt_number(rng)
    
    def process_response(self, response: str) -> None:
        reasoning, _ = response.split('</think>')
        reasoning += "</think>\n"

        return equal_chunk(reasoning, self.granularity)
    
    def corrupt_number(self, rng: random.Random) -> None:
        def process_corrupted_reasoning(row):
            corrupted_reasoning = row["corrupted_reasoning"]
            numbers = extract_number(corrupted_reasoning)
            replacement = {number: perturb_number(number, rng) for number in list(numbers)}
            
            corrupted_reasoning = replace_number(corrupted_reasoning, replacement)
            
            return corrupted_reasoning, replacement
        
        def countdown_corrupt_number (row):
            corrupted_reasoning = row["corrupted_reasoning"]
            target = row["target"]
            numbers = extract_number(corrupted_reasoning)

            replacement = {number: perturb_number(number, rng) for number in list(numbers) if number != target}
            
            corrupted_reasoning = replace_number(corrupted_reasoning, replacement)
            
            return corrupted_reasoning, replacement
        
        func = countdown_corrupt_number if self.task == "countdown" else process_corrupted_reasoning
        self.df["corrupted_reasoning"], self.df["replacement"] = zip(*self.df.apply(func, axis=1))

    def eval(self) -> None:
        """Evaluate the model on the dataset and save results.
        
        This method:
        1. Generates model responses
        2. Computes correctness scores
        3. Saves results to disk
        """
        try:
            # The suffix after corrupt reasoning is to force the model to answer with </think>
            template_prefix, template_suffix = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "HANDLE"}], tokenize=False, add_generation_prompt=True).\
                split("HANDLE")
            
            # Case 1: corrupt the reasoning trace after </think>
            self.post_think_df = self.df.copy()
            self.post_think_df["prompt"] = template_prefix + self.post_think_df["problem"] + template_suffix + \
                self.post_think_df['prefix_reasoning'] + self.post_think_df['corrupted_reasoning']
            self.post_think_df["type"] = "post_<think>"

            # Case 2: corrupt the reasoning trace before </think>
            self.pre_think_df = self.df.copy()
            self.pre_think_df["prompt"] = template_prefix +  self.pre_think_df["problem"] + \
                "\n\nThis is user's thought process for this problem: " + \
                self.pre_think_df['prefix_reasoning'].replace("<think>", "") + \
                self.pre_think_df['corrupted_reasoning'].replace("<think>", "") + template_suffix
            
            self.pre_think_df["type"] = "pre_<think>"

            self.df = pd.concat([self.post_think_df, self.pre_think_df], axis=0, ignore_index=True)
            
            self.response = self.generate(prompts=self.df["prompt"]).rename(columns = {'response': 'post_corruption_response'})
            self.response.index = self.df.index
            self.df = pd.concat([self.df, self.response], axis=1)

            # Compute correctness scores and check if model answers
            correctness = []
            if_answer = []
            for _, row in self.df.iterrows():
                solution = row['post_corruption_response']
                
                # Clean solution if it contains thinking steps
                if "</think>" in solution:
                    solution = solution.split("</think>")[-1]
                
                if self.task == "math":
                    ground_truth = row.get("solution")
                    score = self.compute_score(solution, ground_truth)
                elif self.task == "countdown":
                    numbers, target = row['nums'], row['target']
                    score = self.compute_score(solution, numbers, target)
                    if_answer.append(extract_solution(solution) != None)

                correctness.append(score)
            
            # Combine results
            self.df['still_correct'] = correctness
            
            if if_answer:
                self.df['if_answer'] = if_answer
            
            # Save results
            output_path = os.path.join(self.output_dir, f"{self.nick_name}.pickle")
            self.df.to_pickle(output_path)
            
            # Log summary statistics
            print(f"Evaluation complete. Pre-Think Accuracy: {self.df[self.df['type'] == 'pre_</think>']['still_correct'].mean():.2%}")
            print(f"Evaluation complete. Post-Think Accuracy: {self.df[self.df['type'] == 'post_</think>']['still_correct'].mean():.2%}")
            
        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            raise

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parse Arguments for Reasoner QA evaluation")

    parser.add_argument("--task", type=str, required=True, help="Task to evaluate on")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--nick_name", type=str, required=True, help="Nickname for the model")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name of the tokenizer to use")
    parser.add_argument("--results_dir", type=str, required=True, help="Name of the dataset to evaluate on")
    parser.add_argument("--question_ids_fname", type=str, required=True, help="stress test question ids file name")
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
    parser.add_argument("--granularity", type=int, default=20,
                        help="Granularity of the reasoning chunks")
    parser.add_argument("--unit", type=float, default=0.25,
                        help="Unit of the reasoning chunks")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results")
    parser.add_argument("--generate_data_only", action="store_true",
                        help="Generate data only")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    engine = CorruptNumbers(
        **vars(args),
    )
        
    engine.corrupt_thinking(unit = args.unit, seed = args.seed)
    if not args.generate_data_only:
        engine.eval()