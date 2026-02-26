import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
from multiprocessing.pool import ThreadPool

from core.llm_engine import *
from core.openai_engine import *

from reward_score.math_eval import math_if_boxed
from utils.chunk_r import equal_chunk
from more_itertools import chunked

TEACHERS = [
    'DeepSeek-R1-0528',
    'DeepSeek-R1',
    'Qwen3-235B-A22B-2507',
    'Qwen3-235B-A22B',
    'QwQ-32B'
]

class TeacherGuideCorrect(OpenLMEngine):
    def __init__(
        self,
        model_name: str,
        nick_name: str,
        tokenizer_name: str,
        results_dir: str,
        sample_size: int,
        min_reasoning_tokens: int = 4096,
        min_solve_n: int = 1,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        dtype: str = "bfloat16",
        max_tokens: int = 32768,
        temperature: float = 0.6,
        top_p: float = 1.0,
        top_k: int = -1,
        num_responses_per_problem: int = 1,
        max_num_batched_tokens: int = 8192,
        mini_batch_size: int = None,
        granularity: int = 30,
        overwrite: bool = False,
        client_name: str = "",
        **kwargs,
    ):
        self.model_name = model_name
        self.nick_name = nick_name
        self.tokenizer_name = tokenizer_name
        self.results_dir = results_dir
        self.sample_size = sample_size
        self.min_reasoning_tokens = min_reasoning_tokens
        self.min_solve_n = min_solve_n
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_responses_per_problem = num_responses_per_problem
        self.max_num_batched_tokens = max_num_batched_tokens
        self.mini_batch_size = mini_batch_size
        self.sample_size = sample_size
        self.granularity = granularity
        self.overwrite = overwrite
        self.client_name = client_name

        self.output_dir = os.path.join(self.results_dir, "teacher_guide_correct")
        os.makedirs(self.output_dir, exist_ok=True)

        out_pickle = os.path.join(self.output_dir, f"{self.nick_name}.pickle")
        if os.path.exists(out_pickle) and not self.overwrite:
            print(f"Stress test (Teacher Guide Correct) already exists: {self.nick_name}")
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
                n=self.num_responses_per_problem,
                max_num_batched_tokens=self.max_num_batched_tokens,
                max_model_len=self.max_position_embeddings
            )

        super().__init__(config=config)
        self.load_dataset()

        print(f"Start stress testing: {self.nick_name} on teacher guided reasoning (correct only)")

    @staticmethod
    def convert_model_is_correct(label) -> float:
        if label == True:
            return 1.0
        elif label == False:
            return 0.0
        elif type(label) == float:
            return label
        else:
            raise ValueError(f"Invalid label: {label}")

    def read_benchmark_df(self, model_name):
        df = pd.read_pickle(os.path.join(self.results_dir, "benchmark", f"{model_name}.pickle"))
        df["model_is_correct"] = df["model_is_correct"].apply(self.convert_model_is_correct)
        return df

    def extract_reasoning_trace(self, response):
        """Extract reasoning trace from response"""
        if "</think>" in response:
            reasoning = response.split("</think>")[0]
        else:
            reasoning = response
        reasoning = reasoning.replace("<think>", "").strip('\n')
        return reasoning

    def build_teacher_pool(self):
        """Build a pool of teacher solutions, only keeping correctly solved problems with sufficient reasoning"""
        teacher_df = []
        for teacher_name in TEACHERS:
            if teacher_name == self.nick_name:
                continue
            df = self.read_benchmark_df(teacher_name)
            df = df[(df['model_is_correct'] == 1.0) & (df['pred'] != '')]
            df["teacher"] = teacher_name

            df["teacher_reasoning_trace"] = df["response"].apply(self.extract_reasoning_trace)
            with ThreadPool(32) as pool:
                df["teacher_reasoning_token_count"] = pool.map(
                    lambda x: len(self.tokenizer.encode(x)),
                    df["teacher_reasoning_trace"].tolist()
                )

            df = df[df["teacher_reasoning_token_count"] >= self.min_reasoning_tokens]

            for col in ["error", "retries", "is_correct", "if_answer", "model_judge"]:
                if col in df.columns:
                    df = df.drop(columns=[col])

            teacher_df.append(df)

        teacher_df = pd.concat(teacher_df, ignore_index=True)
        teacher_df = teacher_df[["problem", "teacher", "teacher_reasoning_trace", "teacher_reasoning_token_count"]]
        teacher_df = teacher_df.drop_duplicates(subset=["problem", "teacher"]).reset_index(drop=True)

        return teacher_df

    def load_dataset(self) -> None:
        """Load dataset filtering for problems where student solved >= min_solve_n times correctly"""
        self.df = self.read_benchmark_df(self.nick_name)

        problem_stats = self.df.groupby(["problem", "source"]).agg({
            "model_is_correct": ["sum", "count"]
        }).reset_index()

        problem_stats.columns = ["problem", "source", "correct_count", "total_count"]

        problem_stats = problem_stats[
            (problem_stats["correct_count"] >= self.min_solve_n)
        ].reset_index(drop=True)

        print(f"Found {len(problem_stats)} problems where student solved >= {self.min_solve_n} times correctly")

        self.teacher_df = self.build_teacher_pool()

        self.df = self.df[["problem", "solution", "source"]].drop_duplicates(subset=["problem"])

        self.df = self.df.merge(
            problem_stats[["problem", "source", "correct_count", "total_count"]],
            on=["problem", "source"]
        ).reset_index(drop=True).rename(columns={"correct_count": "solve_n"}).drop(columns=["total_count"])

        self.df = self.df.merge(self.teacher_df, on=["problem"]).reset_index(drop=True)

        if self.sample_size is not None:
            if self.df['problem'].nunique() < self.sample_size:
                raise ValueError(f"Not enough unique problems: {self.df['problem'].nunique()} < {self.sample_size}")
            if self.df[self.df['teacher'] != 'QwQ-32B']['problem'].nunique() >= self.sample_size:
                self.df = self.df[self.df['teacher'] != 'QwQ-32B']. \
                            sample(frac=1.0, random_state=42). \
                            drop_duplicates(subset='problem'). \
                            sample(n=self.sample_size, random_state=42). \
                            reset_index(drop=True)
            else:
                self.priority_df = self.df[self.df['teacher'] != 'QwQ-32B']. \
                                    sample(frac=1.0, random_state=42). \
                                    drop_duplicates(subset='problem')
                self.rest_df = self.df[(self.df['teacher'] == 'QwQ-32B') & (~self.df['problem'].isin(self.priority_df['problem']))]. \
                                    drop_duplicates(subset='problem'). \
                                    sample(n=self.sample_size - len(self.priority_df), random_state=42)
                self.df = pd.concat([self.priority_df, self.rest_df], ignore_index=True)

        print(f"Loaded {len(self.df)} student-teacher pairs for guidance")
        print(f"Unique problems: {self.df['problem'].nunique()}")
        print(f"Teachers distribution:\n{self.df['teacher'].value_counts()}")

    def guide(self, row):
        """Extract a portion of teacher reasoning based on ratio"""
        ratio = row["ratio"]
        teacher_reasoning = row["teacher_reasoning_trace"]

        teacher_chunks = equal_chunk(teacher_reasoning, self.granularity)
        num_chunks = int(len(teacher_chunks) * ratio)
        teacher_chunks = teacher_chunks[:num_chunks]

        return "".join(teacher_chunks)

    def guide_reasoning(self):
        """Apply different guidance ratios (0.2, 0.4, 0.6, 0.8)"""
        windows = [0.2, 0.4, 0.6, 0.8]
        self.df["ratio"] = len(self.df) * [windows]
        self.df = self.df.explode("ratio", ignore_index=True)
        self.df["guide_reasoning"] = self.df.apply(self.guide, axis=1)

    def local_eval(self) -> None:
        """Local evaluation using vLLM"""
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
        self.response = pd.concat(self.responses, ignore_index=True).rename(columns={'response': 'student_response'})
        self.df = self.df.loc[np.repeat(self.df.index, self.num_responses_per_problem)].reset_index(drop=True)
        self.response.index = self.df.index
        self.df = pd.concat([self.df, self.response], axis=1)

    def apply_chat_template(self, row) -> str:
        """Apply chat template with teacher guidance prefix"""
        template_prefix, template_suffix = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "HANDLE"}],
            tokenize=False,
            add_generation_prompt=True
        ).split("HANDLE")

        if '<think>' not in template_suffix and 'limo' not in self.model_name.lower():
            if "openthinker" in self.model_name.lower():
                template_suffix = template_suffix + "<think> "
            else:
                template_suffix = template_suffix + "<think>\n"

        problem = row["problem"]
        guide_reasoning = row["guide_reasoning"]
        prompt = template_prefix + problem + template_suffix + guide_reasoning

        return prompt

    def eval(self) -> None:
        """Run evaluation"""
        self.df["prompt"] = self.df.apply(self.apply_chat_template, axis=1)
        self.df["guide_reasoning_token_counts"] = self.df["guide_reasoning"].apply(lambda x: len(self.tokenizer.encode(x)))

        output_path = os.path.join(self.output_dir, f"{self.nick_name}.pickle")
        self.local_eval()

        self.result_df = self.df.copy()
        self.result_df['pred'] = self.result_df['student_response'].apply(lambda x: x.split('</think>')[-1].strip() if '</think>' in x else x)
        self.result_df['ground_truth'] = self.result_df['solution']
        self.result_df['if_boxed'] = self.result_df['pred'].apply(math_if_boxed)

        self.result_df.to_pickle(output_path)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Arguments for Teacher Guide (Correct Only) evaluation")

    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--nick_name", type=str, required=True, help="Nickname for the model")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name of the tokenizer to use")
    parser.add_argument("--results_dir", type=str, default='/share/goyal/lio/reasoning/eval/',
                       help="Directory to save evaluation results")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of problems to sample for the stress test")
    parser.add_argument("--min_reasoning_tokens", type=int, default=2048,
                        help="Minimum reasoning tokens required for teacher solutions to be included")
    parser.add_argument("--min_solve_n", type=int, default=8,
                        help="Minimum number of correct solves required for student to be included")
    parser.add_argument("--num_responses_per_problem", type=int, default=1,
                        help="Number of responses to generate per problem")

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
    parser.add_argument("--max_num_batched_tokens", type=int, default=32768,
                        help="Maximum number of tokens in a batch")

    parser.add_argument("--mini_batch_size", type=int, default=None,
                        help="Mini batch size for generation")
    parser.add_argument("--granularity", type=int, default=30,
                        help="Granularity of the thinking chunks")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results")
    parser.add_argument("--client_name", type=str, default="",
                        help="Name of the client (for OpenAI or other APIs)")

    args = parser.parse_args()
    
    engine = TeacherGuideCorrect(
        **vars(args),
    )
    engine.guide_reasoning()
    engine.eval()
