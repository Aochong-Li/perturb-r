import os
import time
import logging
from typing import Union, List, Optional, Dict
from dataclasses import dataclass

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# === ADDED ===
import math
import ray

# Allow longer max_model_len in vLLM
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["RAY_CGRAPH_get_timeout"] = "1200"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

@dataclass
class ModelConfig:
    model_name: str
    tokenizer_name: Optional[str] = None
    lora_path: Optional[str] = None
    lora_name: str = "adapter"  
    max_tokens: int = 512
    max_model_len: int = 32768
    temperature: float = 0.6
    n: int = 1
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
    enforce_eager: bool = True
    # === ADDED: data-parallel controls ===
    data_parallel_replicas: int = 1           
    ray_address: Optional[str] = None

def _maybe_build_lora_modules(cfg: Dict):
    lp = cfg.get("lora_path")
    if not lp:
        return None
    return [{"lora_name": cfg.get("lora_name", "adapter"), "lora_path": lp}]

@ray.remote(num_gpus=1, num_cpus=2)
class _VLLMWorker:
    def __init__(self, cfg: Dict, sampling_params: Dict):
        cfg = dict(cfg)
        cfg.setdefault("tensor_parallel_size", cfg.get("tensor_parallel_size", 1))
        cfg.setdefault("pipeline_parallel_size", 1)
        # os.environ.setdefault("HF_HUB_OFFLINE", 1)

        lora_modules = _maybe_build_lora_modules(cfg)

        self.model = LLM(
            model=cfg["model_name"],
            tokenizer=cfg.get("tokenizer_name") or cfg["model_name"],
            dtype=cfg["dtype"],
            gpu_memory_utilization=cfg["gpu_memory_utilization"],
            max_model_len=cfg["max_model_len"],
            max_num_batched_tokens=cfg.get("max_num_batched_tokens"),
            tensor_parallel_size=cfg["tensor_parallel_size"],
            pipeline_parallel_size=cfg["pipeline_parallel_size"],
            distributed_executor_backend=cfg["distributed_executor_backend"],
            trust_remote_code=cfg["trust_remote_code"],
            enable_chunked_prefill=cfg["enable_chunked_prefill"],
            enable_prefix_caching=cfg["enable_prefix_caching"],
            enforce_eager=cfg["enforce_eager"],
        )
        self._sampling_params_base = sampling_params
        self.tokenizer_name = cfg.get("tokenizer_name") or cfg["model_name"]
        self.trust_remote_code = cfg["trust_remote_code"]

    def generate(self, prompts: List[str], new_sampling_params: Optional[List[Dict]] = None) -> List[str]:
        if not prompts:
            return []
        if new_sampling_params is not None:
            sp = [SamplingParams(**{**self._sampling_params_base, **over}) for over in new_sampling_params]
        else:
            sp = SamplingParams(**self._sampling_params_base)
        outs = self.model.generate(prompts=prompts, sampling_params=sp)
        return [o.text for req in outs for o in req.outputs]

class OpenLMEngine:
    """
    Production-ready vLLM inference engine for batch prompt generation.
    """
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_name = config.model_name
        self.tokenizer_name = config.tokenizer_name or config.model_name
        self._dp_enabled = (self.config.data_parallel_replicas or 1) > 1
        self._workers = None  # filled if DP enabled
        self._load_model_and_tokenizer()
        if self._dp_enabled:
            self._init_data_parallel()
        
        self._lora_req = None
        if self.config.lora_path:
            self._lora_req = LoRARequest(
                self.config.lora_name, 1, self.config.lora_path
                )

    def _load_model_and_tokenizer(self) -> None:
        """Instantiate vLLM LLM and tokenizer with config."""
        if hasattr(self, "model"):
            logging.info("Model already loaded, skipping reload.")
            return
        
        if not self._dp_enabled:
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
                enforce_eager=self.config.enforce_eager,
                enable_lora=self.config.lora_path is not None
            )

        else:
            _ = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=self.config.trust_remote_code) # download model weights from HF

        self.sampling_params = {
            "n": self.config.n,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "stop": self.config.stop_tokens,
            "logprobs": self.config.logprobs,
            "prompt_logprobs": self.config.prompt_logprobs,
        }
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, trust_remote_code=self.config.trust_remote_code)
        self.tokenizer.model_max_length = self.config.max_model_len

    def _init_data_parallel(self) -> None:
        if ray.is_initialized():
            pass
        else:
            if self.config.ray_address:
                ray.init(address=self.config.ray_address, ignore_reinit_error=True)
            else:
                ray.init(ignore_reinit_error=True)

        replicas = max(1, int(self.config.data_parallel_replicas))

        cfg_dict = {
            "model_name": self.config.model_name,
            "tokenizer_name": self.config.tokenizer_name or self.config.model_name,
            "dtype": self.config.dtype,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "max_model_len": self.config.max_model_len,
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "pipeline_parallel_size": 1,
            "distributed_executor_backend": "mp",
            "trust_remote_code": self.config.trust_remote_code,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "enforce_eager": self.config.enforce_eager
        }
        sp = dict(self.sampling_params)

        self._workers = [_VLLMWorker.remote(cfg_dict, sp) for _ in range(replicas)]
        logging.info(f"Initialized data-parallel with {len(self._workers)} replica(s).")

    def generate(self, prompts: Union[str, List[str]], new_sampling_params: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Generate responses for a single prompt or list of prompts.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        # Single-replica (default) path: unchanged
        if not self._dp_enabled:
            if new_sampling_params is not None:
                sampling_params = [
                    SamplingParams(**{**self.sampling_params, **over})
                    for over in new_sampling_params
                ]
            else:
                sampling_params = SamplingParams(**self.sampling_params)

            start = time.monotonic()
            try:
                outputs = self.model.generate(prompts=prompts, sampling_params=sampling_params, lora_request=self._lora_req)
            except Exception as e:
                logging.error(f"Generation error: {e}")
                raise e
            duration = time.monotonic() - start
            logging.info(f"Generated {len(prompts)} prompt(s) in {duration:.2f}s")

            responses = [out.text for req in outputs for out in req.outputs]
            return pd.DataFrame(responses, columns=['response'])

        # === ADDED: data-parallel path ===
        if self._workers is None:
            self._init_data_parallel()

        k = len(self._workers)
        shards: List[List[str]] = [[] for _ in range(k)]
        params_shards: Optional[List[Optional[List[Dict]]]] = None
        if new_sampling_params is not None:
            if len(new_sampling_params) != len(prompts):
                raise ValueError("len(new_sampling_params) must equal len(prompts) in DP mode.")
            params_shards = [[] for _ in range(k)]

        for i, p in enumerate(prompts):
            r = i % k
            shards[r].append(p)
            if params_shards is not None:
                params_shards[r].append(new_sampling_params[i])

        if new_sampling_params is not None:
            sampling_params = [
                SamplingParams(**{**self.sampling_params, **over})
                for over in new_sampling_params
            ]
        else:
            sampling_params = SamplingParams(**self.sampling_params)

        start = time.monotonic()
        try:
            pending = []
            for idx, (worker, shard) in enumerate(zip(self._workers, shards)):
                if shard:
                    pshard = None if params_shards is None else params_shards[idx]
                    pending.append(worker.generate.remote(shard, pshard))
            results: List[List[str]] = ray.get(pending) if pending else []
        except Exception as e:
            logging.error(f"DP generation error: {e}")
            raise e
        duration = time.monotonic() - start
        logging.info(f"[DP] Generated {len(prompts)} prompt(s) across {k} replica(s) in {duration:.2f}s")

        flat = [r for group in results for r in group]
        return pd.DataFrame(flat, columns=['response'])

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
    
    def console_chat_completions(self) -> None:
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
                conversation = [{'role': 'user', 'content': user_input}]
                conversation = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                df = self.generate(conversation)
                for _, resp in enumerate(df['response'], 1):
                    print(f"Assistant: {resp}\n")
            except Exception as e:
                logging.error(f"Generation error: {e}")

if __name__ == '__main__':
    config = ModelConfig(
        model_name="Qwen/Qwen2.5-7B",
        lora_path="/mnt/home/al2644/research/projects/rlvr/sft/LLaMA-Factory/outputs/math8k/Qwen2.5-7B-math8k-distill-QwQ-32B-16k-10epochs-5e-5lr/checkpoint-100",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        max_tokens=16384,
        temperature=0.6,
        top_p=1.0,
        top_k=-1
    )
    engine = OpenLMEngine(config)
    engine.console_chat_completions()
