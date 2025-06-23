import os
import pandas as pd
import numpy as np
import sys
sys.path.append("/home/al2644/research")
from codebase.reasoning.llm_engine import *
import argparse
from reward_score.math import compute_score
import logging
import json
import random
import re
from typing import Dict, Set

class CorruptNumbers(OpenLMEngine):
    def __init__(self,
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
                 overwrite: bool = False,
                 generate_data_only: bool = False,
                 **kwargs
                 ):
        # Initialize attributes first
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
    
        # Create output directory if it doesn't exist
        self.output_dir = os.path.join(self.results_dir, "corrupt_numbers")
        os.makedirs(self.output_dir, exist_ok=True)

        if os.path.exists(os.path.join(self.output_dir, f"{self.nick_name}.pickle")) and not overwrite:
            print(f"Dataset already exists: {self.nick_name}")
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
            self.df = pd.read_pickle(self.dataset_path)
            self.df = self.df[
                self.df['unique_id'].isin(question_ids) & 
                (self.df['correct'] == 1) & 
                (self.df["response"].str.contains("</think>"))
            ]
            self.df = self.df.drop_duplicates(subset = ["unique_id"]).reset_index(drop = True)
            self.df = self.df.drop(columns = ['correct']).rename(columns = {'response': 'original_response'})
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from pickle file: {e}")

    def corrupt_thinking(self, unit = 0.2, seed: int = None) -> None:
        rng = random.Random(seed)
        
        self.df["reasoning_chunks"] = self.df["original_response"].apply(self.process_response)
        self.df = self.df.dropna(how='any').reset_index(drop = True)
        
        # generate all possible start and end positions
        pos = np.arange(0, 1, unit)
        windows = [(round(p, 1), round(p + unit, 1)) for p in pos]
        
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
    
    def process_response(self, response: str, granularity: int = 20) -> None:
        reasoning, _ = response.split('</think>')
        reasoning += "</think>\n"

        def chunk(reasoning: str):
            chunks = reasoning.split('\n\n')
            masks = [len(chunk.split()) > granularity for chunk in chunks]
            
            # Step 1: chunk the sequence into small chunks
            merged, buffer = [], []
            for c, m in zip(chunks, masks):
                if not m:
                    buffer.append(c)
                else:
                    if buffer:
                        merged.append('\n\n'.join(buffer))
                        buffer.clear()
                    merged.append(c)
            if buffer:
                merged.append('\n\n'.join(buffer))
            
            # Step 2: merge small chunks to big chunks
            super_chunks, current = [], None
            for c in merged:
                if len(c.split()) > granularity:
                    if current is not None:
                        super_chunks.append(current)
                    current = c
                else:
                    if current is None:
                        current = c
                    else:
                        current += '\n\n' + c
            
            if current is not None:
                super_chunks.append(current)
            
            return super_chunks

        return chunk(reasoning)
    
    def corrupt_number(self, rng: random.Random) -> None:
        def extract_number(text: str) -> Set[str]:
            """
            Return the set of all (unique) numeric literals in `text`.
            Handles integers, decimals with either leading or trailing digits,
            and optional scientific notation.
            """
            _NUMBER_RE = re.compile(
                r'(?:\d+\.\d+|\d+|\.\d+)'        # 123.456   | 123   | .456
            )
            return {m.group(0) for m in _NUMBER_RE.finditer(text)}

        def perturb_number(num: str, rng: random.Random) -> str:
            """
            Make a realistic one-step typo in `num` under the rules:
                • single-digit integer  → substitute with a different digit
                • multi-digit integer  → transpose OR dup/del (50-50)
                • float (contains '.') → decimal-point error (random mode)
            Always returns a *different* string and never introduces a leading zero.
            """
            sign = ''
            if num[0] in '+-':            # keep explicit sign, if any
                sign, num = num[0], num[1:]

            def _strip_leading_zero(s: str) -> str:
                while len(s) > 1 and s[0] == '0' and s[1].isdigit():
                    s = s[1:]
                return s

            # ---------------------- floats ------------------------------------------
            if '.' in num:
                digits = num.replace('.', '')
                if rng.random() < 0.5 or len(digits) < 2:
                    new_core = digits

                else:
                    new_pos = rng.randint(0, len(digits))
                    new_core = digits[:new_pos] + '.' + digits[new_pos:]
                new_core = _strip_leading_zero(new_core)

            # ---------------------- integers ----------------------------------------
            elif len(num) == 1:
                # single digit → substitute
                new_digit = rng.choice([d for d in '0123456789' if d != num])
                new_core = new_digit
            else:
                # multi-digit int
                if rng.random() < .5:          # --- randomize the order of digits
                    lst = list(num)
                    rng.shuffle(lst)
                    new_core = ''.join(lst)
                else:                           # ---- duplicate OR delete
                    i = rng.randint(0, len(num) - 1)
                    if rng.random() < 0.5 and len(num) > 1:   # delete
                        new_core = num[:i] + num[i+1:]
                    else:                                     # duplicate
                        new_core = num[:i+1] + num[i] + num[i+1:]
                new_core = _strip_leading_zero(new_core)

            # Guarantee change; if not, recurse once (extremely rare)
            if new_core == num:
                return perturb_number(num, rng)
            return sign + new_core
        
        def replace_number(text: str, replacement: Dict[str, str]) -> str:
            """
            Replace every numeric literal in `text` according to `replacement`.
            Any number not in the dict is left unchanged.
            """
            _NUMBER_RE = re.compile(
                r'(?<!\w)'          # not preceded by a letter/number/underscore
                r'(?:\d+\.\d+|'     # 123.456
                r'\d+|'             # 123
                r'\.\d+)'           # .456
                r'(?!\w)'           # not followed by a letter/number/underscore
            )
            return _NUMBER_RE.sub(
                lambda m: replacement.get(m.group(0), m.group(0)),
                text
            )

        def process_corrupted_reasoning(row):
            corrupted_reasoning = row["corrupted_reasoning"]
            numbers = extract_number(corrupted_reasoning)
            replacement = {number: perturb_number(number, rng) for number in list(numbers)}
            
            corrupted_reasoning = replace_number(corrupted_reasoning, replacement)
            
            return corrupted_reasoning, replacement
        
        self.df["corrupted_reasoning"], self.df["replacement"] = zip(*self.df.apply(process_corrupted_reasoning, axis=1))

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
                self.post_think_df['prefix_reasoning'] + '\n\n' + self.post_think_df['corrupted_reasoning']
            self.post_think_df["type"] = "post_</think>"

            # Case 2: corrupt the reasoning trace before </think>
            self.pre_think_df = self.df.copy()
            self.pre_think_df["prompt"] = template_prefix +  self.pre_think_df["problem"] + \
                "\n\n This is my thought process for this problem: " + \
                self.pre_think_df['corrupted_reasoning'] + '\n\n' + self.pre_think_df['prefix_reasoning'] + template_suffix
            self.pre_think_df["type"] = "pre_</think>"

            self.df = pd.concat([self.post_think_df, self.pre_think_df], axis=0, ignore_index=True)
            
            self.response = self.generate(prompts=self.df["prompt"]).rename(columns = {'response': 'post_corruption_response'})
            self.response.index = self.df.index
            self.df = pd.concat([self.df, self.response], axis=1)

            # Compute correctness scores
            correctness = []
            for _, row in self.df.iterrows():
                solution = row['post_corruption_response']
                ground_truth = row['solution']
                
                # Clean solution if it contains thinking steps
                if "</think>" in solution:
                    solution = solution.split("</think>")[-1]
                
                score = compute_score(solution, ground_truth)
                correctness.append(score)
            
            # Combine results
            self.df['still_correct'] = correctness
            
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
    parser.add_argument("--overwrite", type=bool, default=False,
                        help="Overwrite existing results")
    parser.add_argument("--generate_data_only", type=bool, default=False,
                        help="Generate data only")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    engine = CorruptNumbers(
        **vars(args),
    )
        
    engine.corrupt_thinking(unit = 0.2, seed = args.seed)
    if not args.generate_data_only:
        engine.eval()