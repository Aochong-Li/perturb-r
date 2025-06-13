import os
import sys
import time
import logging
from typing import Union, List, Optional, Dict
from dataclasses import dataclass

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
# Environment variable to allow longer max_model_len in vLLM
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

@dataclass
class ModelConfig:
    model_name: str
    tokenizer_name: Optional[str] = None
    max_tokens: int = 512
    max_model_len: int = 32768
    temperature: float = 0.6
    n: int = 1
    best_of: int = 1
    top_p: float = 0.95
    top_k: int = 32
    stop_tokens: Optional[List[str]] = None
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    gpu_memory_utilization: float = 0.4
    dtype: str = 'bfloat16'
    max_num_batched_tokens: Optional[int] = None
    
    # Distributed inference settings
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    distributed_executor_backend: str = 'mp'  # 'mp' or 'ray'
    trust_remote_code: bool = False

class OpenLMEngine:
    """
    Wrapper for vLLM inference with support for multi-GPU, and detailed logging.
    """
    def __init__(self, config: ModelConfig):
        self.config = config
        self._init_model_tokenizer()
        self._load_model_tokenizer()

    def _init_model_tokenizer(self) -> None:
        """Set up model and tokenizer names."""
        self.tokenizer_name = self.config.tokenizer_name or self.config.model_name
        self.model_name = self.config.model_name

    def _load_model_tokenizer(self) -> None:
        """Instantiate the vLLM LLM with distributed settings and load tokenizer."""
        try:
            _ = self.model  # Check if already loaded
            logging.info('Model already loaded, skipping reload')
        except AttributeError:
            logging.info(f'Loading model: {self.model_name}')
            self.model = LLM(
                model=self.model_name,
                tokenizer=self.tokenizer_name,
                dtype=self.config.dtype,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
                tensor_parallel_size=self.config.tensor_parallel_size,
                pipeline_parallel_size=self.config.pipeline_parallel_size,
                distributed_executor_backend=self.config.distributed_executor_backend,
                trust_remote_code=self.config.trust_remote_code,
                enforce_eager=True
            )
        
        self.sampling_params = SamplingParams(
            n=self.config.n,
            best_of=max(self.config.best_of, self.config.n),
            logprobs=self.config.logprobs,
            prompt_logprobs=self.config.prompt_logprobs,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stop=self.config.stop_tokens,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.tokenizer.model_max_length = self.config.max_model_len

    def generate(self,
                 prompts: Union[str, List[str]],
                 ) -> pd.DataFrame:
        """
        Generate responses. Supports optional latency logging.

        Args:
            prompts: Single prompt or list of prompts.

        Returns:
            DataFrame with one row per generated response.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        start = time.monotonic()
        try:
            outputs = self.model.generate(
                prompts=prompts,
                sampling_params=self.sampling_params,
            )
        except Exception as e:
            logging.error(f"Generation error: {e}, retrying once...")
            outputs = self.model.generate(prompts=prompts, sampling_params=self.sampling_params)
        duration = time.monotonic() - start
        logging.info(f"Generated {len(prompts)} prompt(s) in {duration:.2f}s")

        rows = []
        for req in outputs:
            for out in req.outputs:
                rows.append(out.text)
        return pd.DataFrame(rows, columns=['response'])

    def chat(self, prompts: Union[str, List[str]]) -> pd.DataFrame:
        """
        Chat interface: formats prompts with role tags and generates responses.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        formatted = []
        for prompt in prompts:
            conv = [{"role": "user", "content": prompt}]
            formatted.append(
                self.tokenizer.apply_chat_template(
                    conversation=conv,
                    tokenize=False,
                    add_generation_prompt=True
                )
            )
        outputs = self.model.generate(formatted, sampling_params=self.sampling_params)
        return pd.DataFrame(
            [out.text for req in outputs for out in req.outputs],
            columns=['response']
        )

    def console_generate(self) -> None:
        """Interactive generation mode: line-by-line prompts without history."""
        print("Interactive generation mode. Type 'exit' to quit.\n")
        while True:
            try:
                print("User: ", end="", flush=True)
                user_input = ""
                while True:
                    try:
                        line = input()
                        user_input += line + "\n"
                    except EOFError:
                        break
                user_input = user_input.rstrip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting interactive session.")
                break

            if user_input.lower() == 'exit':
                print("Exiting interactive session.")
                break
            if not user_input:
                continue

            try:
                df = self.generate(user_input)
                for idx, resp in enumerate(df['response'], 1):
                    print(f"Response {idx}: {resp}\n")
            except Exception as e:
                logging.error(f"Generation error: {e}")

    def console_chat(self, keep_history: bool = True) -> None:
        """Interactive chat mode: maintains conversation history and supports clearing."""
        print("Interactive chat mode. Type 'exit' to quit, 'clear' to reset history.\n")
        history: List[Dict[str, str]] = [] if keep_history else []

        while True:
            try:
                user_input = input("User: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting interactive session.")
                break

            cmd = user_input.lower()
            if cmd == 'exit':
                print("Exiting interactive session.")
                break
            if keep_history and cmd == 'clear':
                history.clear()
                print("Chat history cleared.\n")
                continue
            if not user_input:
                continue

            # Append user message
            if keep_history:
                history.append({"role": "user", "content": user_input})
            else:
                history = [{"role": "user", "content": user_input}]

            # Format and generate
            formatted = self.tokenizer.apply_chat_template(
                conversation=history,
                tokenize=False,
                add_generation_prompt=True
            )
            try:
                outputs = self.model.generate(formatted, sampling_params=self.sampling_params)
                response = outputs[0].outputs[0].text
                print(f"Assistant: {response}\n")
            except Exception as e:
                logging.error(f"Chat error: {e}")
                continue

            # Append assistant response
            if keep_history:
                history.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    # Fix: Use a local path with repo_type="local" for local model loading
    config = ModelConfig(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        tokenizer_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.6,
        dtype="bfloat16",
        max_tokens=16384,
        temperature=0.6,
        top_p=1.0,
        top_k=-1
    )
    engine_qwen = OpenLMEngine(config)
    # prompts = ['<|im_start|>system\nPlease reason step by step, and  put your final answer within \\boxed{}<|im_end|>\n<|im_start|>user\nThe decimal expansion of $8/11$ is a repeating decimal. What is the least number of digits in a repeating block of 8/11?<|im_end|>\n<|im_start|>assistant\n<think>']
    # df = engine_qwen.generate(prompts)
    # print(df['response'].loc[0])
    engine_qwen.console_generate()
