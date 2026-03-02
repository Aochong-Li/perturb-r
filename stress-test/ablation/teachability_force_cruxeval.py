"""
CruxEval Leakage Detection via Forced Generation

Same approach as teachability_force_code.py but for CruxEval:
  1. Load teacher_guide pickles, filter to cruxeval source.
  2. Construct prompts with forced </think> after the steer.
  3. Generate output predictions via vLLM (short assert statements).
  4. Evaluate against ground truth using code execution + string matching.

Usage:
    python stress-test/ablation/teachability_force_cruxeval.py \
        --model_name Qwen/Qwen3-1.7B \
        --nick_name Qwen3-1.7B \
        --input_dir results/allcode/teacher_guide \
        --output_dir results/allcode/teacher_guide/teachability_force_cruxeval \
        --tensor_parallel_size 1 \
        --temperature 0.6 \
        --max_tokens 512
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import pandas as pd
from transformers import AutoConfig, AutoTokenizer
from more_itertools import chunked
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from core.llm_engine import ModelConfig, OpenLMEngine
from reward_score.codeeval import code_verify_score


# ---------------------------------------------------------------------------
# Data loading (reuses logic from teachability_force_code.py)
# ---------------------------------------------------------------------------

def load_and_filter(input_dir: str, nick_name: str, sample_size: int = None) -> pd.DataFrame:
    pickle_path = os.path.join(input_dir, f"{nick_name}.pickle")
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(
            f"Teacher-guide pickle not found: {pickle_path}\n"
            f"Available: {[f for f in os.listdir(input_dir) if f.endswith('.pickle')]}"
        )

    df = pd.read_pickle(pickle_path)
    df = df[df['source'] == 'cruxeval'].reset_index(drop=True)
    print(f"Loaded {len(df)} cruxeval rows from {pickle_path}")
    print(f"  Ratios:  {sorted(df['ratio'].unique())}")
    print(f"  Teachers: {sorted(df['teacher'].unique())}")

    if 'ground_truth' not in df.columns:
        df['ground_truth'] = df['solution']

    if sample_size is not None:
        df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
        print(f"Sampled down to {len(df)} rows")

    return df


# ---------------------------------------------------------------------------
# Prompt construction (identical to force_code)
# ---------------------------------------------------------------------------

def compute_template_parts(tokenizer, model_name: str):
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


def build_prompt(row: pd.Series, template_prefix: str, template_suffix: str) -> str:
    return (
        template_prefix
        + row['problem']
        + template_suffix
        + row['teacher_reasoning']
        + "\n</think>\n\n"
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def run_generation(engine, df, max_tokens, max_position_embeddings, mini_batch_size=None):
    mini_batch_size = len(df) if mini_batch_size is None else mini_batch_size
    responses = []

    for batch_prompts in chunked(list(df['prompt']), mini_batch_size):
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
        columns={'response': 'force_response'}
    )
    response_df.index = df.index
    df = pd.concat([df, response_df], axis=1)
    return df


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _eval_worker(args):
    idx, row = args
    pred = row.get('force_pred', '')
    solution = row.get('solution', '')
    problem = row.get('problem', '')
    result = code_verify_score(problem, pred, solution, 'cruxeval', None)
    return idx, result


def evaluate_predictions(df, n_workers=32):
    rows_to_eval = [(idx, row.to_dict()) for idx, row in df.iterrows()]
    results = {}
    print(f"Evaluating {len(rows_to_eval)} predictions with {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_eval_worker, item) for item in rows_to_eval]
        for future in tqdm(as_completed(futures), total=len(futures), desc="CruxEval eval"):
            idx, result = future.result()
            results[idx] = result

    df['force_correct'] = [results[idx] for idx in df.index]
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CruxEval leakage detection via forced generation")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--nick_name", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default=None)

    parser.add_argument("--input_dir", type=str, default="results/allcode/teacher_guide")
    parser.add_argument("--output_dir", type=str,
                        default="results/allcode/teacher_guide/teachability_force_cruxeval")

    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="CruxEval answers are short assertions")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=-1)

    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_num_batched_tokens", type=int, default=32768)
    parser.add_argument("--mini_batch_size", type=int, default=None)

    parser.add_argument("--n_eval_workers", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    if args.tokenizer_name is None:
        args.tokenizer_name = args.model_name

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.nick_name}.pickle")
    if os.path.exists(output_path) and not args.overwrite:
        print(f"Output already exists: {output_path}  (use --overwrite to replace)")
        sys.exit(0)

    # Step 1: Data
    df = load_and_filter(args.input_dir, args.nick_name, args.sample_size)

    # Step 2: Prompts
    cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    max_position_embeddings = cfg.max_position_embeddings

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    tokenizer.model_max_length = max_position_embeddings

    template_prefix, template_suffix = compute_template_parts(tokenizer, args.model_name)
    df['prompt'] = df.apply(lambda row: build_prompt(row, template_prefix, template_suffix), axis=1)
    print(f"Built {len(df)} prompts")

    # Step 3: Generate
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

    df = run_generation(engine, df, args.max_tokens, max_position_embeddings, args.mini_batch_size)

    # Step 4: Extract prediction (strip any extra </think>)
    df['force_pred'] = df['force_response'].apply(
        lambda x: x.split('</think>')[-1].strip() if '</think>' in str(x) else str(x).strip()
    )

    # Step 5: Evaluate
    df = evaluate_predictions(df, n_workers=args.n_eval_workers)

    # Step 6: Save
    df.to_pickle(output_path)
    print(f"\nSaved {len(df)} rows to {output_path}")

    # Summary
    print("\nResults summary:")
    for ratio in sorted(df['ratio'].unique()):
        subset = df[df['ratio'] == ratio]
        n_correct = subset['force_correct'].sum()
        n_total = len(subset)
        pct = n_correct / n_total * 100 if n_total > 0 else 0
        print(f"  ratio={ratio:.1f} | {n_correct:.0f}/{n_total} ({pct:.1f}%)")

    print(f"\n  Overall force-cruxeval pass rate: {df['force_correct'].mean():.2%}")


if __name__ == "__main__":
    main()
