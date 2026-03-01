"""
HumanEval/MBPP Code Leakage Detection via Forced Generation

Checks whether teacher-guide steers for coding tasks already contain enough
information to solve the problem without any additional reasoning.

Approach:
  1. Load teacher_guide pickles (which already contain teacher_reasoning at
     various ratios).
  2. Filter to coding benchmarks (humaneval, mbpp, mbppplus).
  3. Construct prompts identical to teacher_guide.py but FORCE </think>
     immediately after the steer so the model cannot reason further.
  4. Generate code via vLLM.
  5. Execute generated code against test cases.
  6. If it passes, the steer alone was sufficient --> leakage.

Usage:
    python stress-test/ablation/teachability_force_code.py \
        --model_name Qwen/Qwen3-1.7B \
        --nick_name Qwen3-1.7B \
        --input_dir results/allcode/teacher_guide \
        --output_dir results/allcode/teacher_guide/teachability_force_code \
        --tensor_parallel_size 4 \
        --temperature 0.6 \
        --max_tokens 2048
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import re
import argparse
import logging

import numpy as np
import pandas as pd
from transformers import AutoConfig, AutoTokenizer
from datasets import load_from_disk
from more_itertools import chunked
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from core.llm_engine import ModelConfig, OpenLMEngine
from reward_score.codeeval import (
    extract_code,
    code_verify_score,
    evaluate_row,
)

CODING_SOURCES = ['humaneval', 'mbpp', 'mbppplus']


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_teacher_guide_pickles(input_dir: str, nick_name: str) -> pd.DataFrame:
    """Load the teacher_guide pickle for a specific student model.

    The pickle already contains rows exploded across teachers and ratios,
    with the teacher_reasoning column pre-computed.
    """
    pickle_path = os.path.join(input_dir, f"{nick_name}.pickle")
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(
            f"Teacher-guide pickle not found: {pickle_path}\n"
            f"Available pickles: {[f for f in os.listdir(input_dir) if f.endswith('.pickle')]}"
        )

    df = pd.read_pickle(pickle_path)
    print(f"Loaded {len(df)} rows from {pickle_path}")
    return df


def merge_entry_points(df: pd.DataFrame, dataset_path: str) -> pd.DataFrame:
    """Merge entry_point from the original allcode dataset.

    The teacher_guide pickle does not contain entry_point, which is needed
    for HumanEval evaluation (check(entry_point)).
    """
    ds = load_from_disk(dataset_path)['test']
    ds_df = pd.DataFrame(ds)[['problem', 'entry_point']].drop_duplicates(subset=['problem'])

    n_before = len(df)
    df = df.merge(ds_df, on='problem', how='left')
    assert len(df) == n_before, (
        f"Merge changed row count: {n_before} -> {len(df)}. "
        "Check for duplicate problems in the dataset."
    )
    n_missing = df['entry_point'].isna().sum()
    if n_missing > 0:
        print(f"WARNING: {n_missing} rows have no entry_point after merge")

    return df


def prepare_data(
    input_dir: str,
    nick_name: str,
    dataset_path: str,
    sample_size: int = None,
) -> pd.DataFrame:
    """Load, filter to coding sources, merge entry_point, optionally sample."""
    df = load_teacher_guide_pickles(input_dir, nick_name)

    # Filter to coding benchmarks only
    df = df[df['source'].isin(CODING_SOURCES)].reset_index(drop=True)
    print(f"After filtering to coding sources: {len(df)} rows")
    print(f"  Sources: {dict(df['source'].value_counts())}")
    print(f"  Ratios:  {sorted(df['ratio'].unique())}")
    print(f"  Teachers: {sorted(df['teacher'].unique())}")

    # Merge entry_point from original dataset
    df = merge_entry_points(df, dataset_path)

    # Keep the columns we need
    keep_cols = [
        'problem', 'solution', 'source', 'teacher', 'ratio',
        'teacher_reasoning', 'teacher_reasoning_token_counts',
        'ground_truth', 'entry_point',
    ]
    # Only keep columns that actually exist (ground_truth == solution in some pickles)
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()

    # Ensure ground_truth exists
    if 'ground_truth' not in df.columns:
        df['ground_truth'] = df['solution']

    if sample_size is not None:
        df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
        print(f"Sampled down to {len(df)} rows")

    return df


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_force_code_prompt(
    row: pd.Series,
    template_prefix: str,
    template_suffix: str,
) -> str:
    """Build a prompt that forces the model to generate code immediately.

    Identical to teacher_guide.py:apply_chat_template but appends
    '\\n</think>\\n\\n' after the teacher_reasoning so the model cannot
    do any additional reasoning and must output code directly.
    """
    problem = row['problem']
    teacher_reasoning = row['teacher_reasoning']

    prompt = (
        template_prefix
        + problem
        + template_suffix
        + teacher_reasoning
        + "\n</think>\n\n"
    )
    return prompt


def compute_template_parts(tokenizer, model_name: str):
    """Compute template_prefix and template_suffix once.

    Follows the exact same logic as teacher_guide.py:228-239.
    """
    template_prefix, template_suffix = tokenizer.apply_chat_template(
        [{"role": "user", "content": "HANDLE"}],
        tokenize=False,
        add_generation_prompt=True,
    ).split("HANDLE")

    if '<think>' not in template_suffix and 'limo' not in model_name.lower():
        if "openthinker" in model_name.lower():
            template_suffix = template_suffix + "<think> "
        else:
            template_suffix = template_suffix + "<think>\n"

    return template_prefix, template_suffix


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def run_generation(
    engine: OpenLMEngine,
    df: pd.DataFrame,
    max_tokens: int,
    max_position_embeddings: int,
    mini_batch_size: int = None,
) -> pd.DataFrame:
    """Run vLLM generation in mini-batches, returning the updated dataframe.

    Follows the same pattern as teacher_guide.py:local_eval.
    """
    mini_batch_size = len(df) if mini_batch_size is None else mini_batch_size
    responses = []

    for batch_prompts in chunked(list(df['prompt']), mini_batch_size):
        # Per-prompt max_tokens capped by model context window
        sampling_overrides = [
            {
                "max_tokens": min(
                    max_tokens,
                    max_position_embeddings - len(engine.tokenizer.encode(prompt)) - 1,
                )
            }
            for prompt in batch_prompts
        ]
        out = engine.generate(prompts=batch_prompts, sampling_overrides=sampling_overrides)
        responses.append(out)

    response_df = pd.concat(responses, ignore_index=True).rename(
        columns={'response': 'force_code_response'}
    )
    response_df.index = df.index
    df = pd.concat([df, response_df], axis=1)
    return df


# ---------------------------------------------------------------------------
# Code evaluation
# ---------------------------------------------------------------------------

def _eval_worker(args):
    """Worker for parallel code evaluation.

    Wraps code_verify_score with the force-code prediction column.
    """
    idx, row = args
    pred = row.get('force_code_pred', '')
    solution = row.get('solution', '')
    source = row.get('source', '')
    entry_point = row.get('entry_point', None)
    problem = row.get('problem', '')

    result = code_verify_score(problem, pred, solution, source, entry_point)
    return idx, result


def evaluate_predictions(df: pd.DataFrame, n_workers: int = 32) -> pd.DataFrame:
    """Run generated code against test cases in parallel."""
    rows_to_eval = [(idx, row.to_dict()) for idx, row in df.iterrows()]

    results = {}
    print(f"Evaluating {len(rows_to_eval)} predictions with {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_eval_worker, item) for item in rows_to_eval]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Code eval"):
            idx, result = future.result()
            results[idx] = result

    df['force_code_correct'] = [results[idx] for idx in df.index]
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment B: Code leakage detection via forced generation"
    )

    # Model / tokenizer
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model name or local path")
    parser.add_argument("--nick_name", type=str, required=True,
                        help="Short name for this student model (used as pickle filename stem)")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Tokenizer name (defaults to model_name)")

    # I/O paths
    parser.add_argument("--input_dir", type=str,
                        default="results/allcode/teacher_guide",
                        help="Directory containing teacher_guide pickles")
    parser.add_argument("--output_dir", type=str,
                        default="results/allcode/teacher_guide/teachability_force_code",
                        help="Directory to save output pickles")
    parser.add_argument("--dataset_path", type=str,
                        default="./data/allcode",
                        help="Path to allcode dataset (for entry_point)")

    # Sampling / subset
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Subsample for quick testing")

    # Generation
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Max new tokens (code is short, 2048 is plenty)")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=-1)

    # vLLM engine
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_num_batched_tokens", type=int, default=32768)
    parser.add_argument("--mini_batch_size", type=int, default=None,
                        help="Mini-batch size for generation (None = all at once)")

    # Evaluation
    parser.add_argument("--n_eval_workers", type=int, default=32,
                        help="Parallel workers for code execution")

    # Misc
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output pickle")

    args = parser.parse_args()
    if args.tokenizer_name is None:
        args.tokenizer_name = args.model_name

    # -----------------------------------------------------------------------
    # Output guard
    # -----------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.nick_name}.pickle")
    if os.path.exists(output_path) and not args.overwrite:
        print(f"Output already exists: {output_path}  (use --overwrite to replace)")
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Step 1: Data preparation
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Loading and preparing data")
    print("=" * 60)
    df = prepare_data(
        input_dir=args.input_dir,
        nick_name=args.nick_name,
        dataset_path=args.dataset_path,
        sample_size=args.sample_size,
    )

    # -----------------------------------------------------------------------
    # Step 2: Build prompts
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Building forced-code prompts")
    print("=" * 60)

    # Get model max context length
    cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    max_position_embeddings = cfg.max_position_embeddings

    # We need the tokenizer for template construction. OpenLMEngine will also
    # load it, but we need it before engine init to build prompts.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, trust_remote_code=True
    )
    tokenizer.model_max_length = max_position_embeddings

    template_prefix, template_suffix = compute_template_parts(tokenizer, args.model_name)
    print(f"Template prefix (last 60 chars): ...{template_prefix[-60:]}")
    print(f"Template suffix: {repr(template_suffix)}")

    df['prompt'] = df.apply(
        lambda row: build_force_code_prompt(row, template_prefix, template_suffix),
        axis=1,
    )
    print(f"Built {len(df)} prompts")
    print(f"Sample prompt (first 300 chars):\n{df['prompt'].iloc[0][:300]}...")
    print(f"Sample prompt (last 100 chars):\n...{df['prompt'].iloc[0][-100:]}")

    # -----------------------------------------------------------------------
    # Step 3: vLLM generation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Running vLLM generation")
    print("=" * 60)

    config = ModelConfig(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        n=1,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=max_position_embeddings,
    )
    engine = OpenLMEngine(config=config)

    df = run_generation(
        engine=engine,
        df=df,
        max_tokens=args.max_tokens,
        max_position_embeddings=max_position_embeddings,
        mini_batch_size=args.mini_batch_size,
    )

    # -----------------------------------------------------------------------
    # Step 4: Extract predictions
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Extracting predictions from generated text")
    print("=" * 60)

    # The model output should be code directly (we forced </think>).
    # But in case the model emits another </think>, strip everything before
    # the last one, same as teacher_guide.py:258.
    df['force_code_pred'] = df['force_code_response'].apply(
        lambda x: x.split('</think>')[-1].strip() if '</think>' in str(x) else str(x).strip()
    )

    print(f"Sample prediction (first 200 chars):\n{df['force_code_pred'].iloc[0][:200]}")

    # -----------------------------------------------------------------------
    # Step 5: Code evaluation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Evaluating generated code against test cases")
    print("=" * 60)

    df = evaluate_predictions(df, n_workers=args.n_eval_workers)

    # -----------------------------------------------------------------------
    # Step 6: Save and summarize
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 6: Saving results")
    print("=" * 60)

    df.to_pickle(output_path)
    print(f"Saved {len(df)} rows to {output_path}")

    # Summary stats
    print("\nResults summary:")
    for source in sorted(df['source'].unique()):
        src_df = df[df['source'] == source]
        for ratio in sorted(src_df['ratio'].unique()):
            subset = src_df[src_df['ratio'] == ratio]
            n_correct = subset['force_code_correct'].sum()
            n_total = len(subset)
            n_nan = subset['force_code_correct'].isna().sum()
            pct = n_correct / n_total * 100 if n_total > 0 else 0
            print(f"  {source} | ratio={ratio:.1f} | {n_correct:.0f}/{n_total} ({pct:.1f}%) | NaN={n_nan}")

    overall = df['force_code_correct'].mean()
    print(f"\n  Overall force-code pass rate: {overall:.2%}")


if __name__ == "__main__":
    main()
