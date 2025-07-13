import os
import time
import logging
from typing import Union, List, Optional
from dataclasses import dataclass

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Allow longer max_model_len in vLLM
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

@dataclass
class ModelConfig:
    model_name: str
    tokenizer_name: Optional[str] = None
    max_tokens: int = 512
    max_model_len: int = 32768
    temperature: float = 0.6
    n: int = 1
    # best_of: int = 1
    top_p: float = 0.95
    top_k: int = 32
    stop_tokens: Optional[List[str]] = None
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    gpu_memory_utilization: float = 0.75
    dtype: str = 'bfloat16'
    max_num_batched_tokens: Optional[int] = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    distributed_executor_backend: str = 'mp'
    trust_remote_code: bool = True
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    # Speed optimization parameters
    enforce_eager: bool = False  # Keep CUDA graphs for speed
    # speculative_config: Optional[Union[dict, str]] = "auto"

class OpenLMEngine:
    """
    Production-ready vLLM inference engine for batch prompt generation.
    """
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_name = config.model_name
        self.tokenizer_name = config.tokenizer_name or config.model_name

        # if self.config.speculative_config == "auto":
        #     if "R1-Distill" in self.config.model_name and self.config.model_name != "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":
        #         self.config.speculative_config = {
        #             "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        #             "num_speculative_tokens": 6,
        #             "draft_tensor_parallel_size": 2
        #         }
        #     elif "Qwen3" in self.config.model_name and self.config.model_name != "Qwen/Qwen3-1.7B":
        #         self.config.speculative_config = {
        #             "model": "Qwen/Qwen3-1.7B",
        #             "num_speculative_tokens": 6,
        #             "draft_tensor_parallel_size": 2
        #         }
        #     else:
        #         self.config.speculative_config = None

        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self) -> None:
        """Instantiate vLLM LLM and tokenizer with config."""
        if hasattr(self, "model"):
            logging.info("Model already loaded, skipping reload.")
            return

        logging.info(f"Loading model: {self.model_name}")
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
            enable_chunked_prefill=self.config.enable_chunked_prefill,
            enable_prefix_caching=self.config.enable_prefix_caching,
            # speculative_config=self.config.speculative_config,
            enforce_eager=self.config.enforce_eager
        )

        self.sampling_params = SamplingParams(
            n=self.config.n,
            # best_of=max(self.config.best_of, self.config.n),
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

    def generate(self, prompts: Union[str, List[str]]) -> pd.DataFrame:
        """
        Generate responses for a single prompt or list of prompts.

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

        responses = [out.text for req in outputs for out in req.outputs]
        return pd.DataFrame(responses, columns=['response'])

    def console_generate(self) -> None:
        """Interactive mode: prompt for user input and generate responses."""
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
                for _, resp in enumerate(df['response'], 1):
                    print(f"Assistant: {resp}\n")
            except Exception as e:
                logging.error(f"Generation error: {e}")

if __name__ == '__main__':
    config = ModelConfig(
        model_name="aochongoliverli/Qwen2.5-3B-countdown-level4-5-grpo-20k-1epoch",
        tensor_parallel_size=2,

        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        max_tokens=16384,
        temperature=0.6,
        top_p=1.0,
        top_k=-1
    )
    engine = OpenLMEngine(config)
    engine.console_generate()
