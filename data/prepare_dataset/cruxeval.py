#!/usr/bin/env python3
"""
Process CRUXEval dataset into a unified format for code reasoning evaluation.

CRUXEval consists of two tasks:
- CRUXEval-O (Output Prediction): Given code + input, predict output
- CRUXEval-I (Input Prediction): Given code + output, predict input

This script uses the EXACT prompt templates from the CRUXEval repository:
https://github.com/facebookresearch/cruxeval
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict


# Exact copy from cruxeval/prompts.py
def make_direct_output_prompt(s):
    code, input = s
    return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(n):
    return n
assert f(17) == ??
[/PYTHON]
[ANSWER]
assert f(17) == 17
[/ANSWER]

[PYTHON]
def f(s):
    return s + "a"
assert f("x9j") == ??
[/PYTHON]
[ANSWER]
assert f("x9j") == "x9ja"
[/ANSWER]

[PYTHON]
{code}
assert f({input}) == ??
[/PYTHON]
[ANSWER]"""


# Exact copy from cruxeval/prompts.py
def make_direct_input_prompt(s):
    """Create direct input prediction prompt (CRUXEval-I).

    Copied exactly from: https://github.com/facebookresearch/cruxeval/blob/main/prompts.py
    """
    code, output = s
    return f"""You will be given a function f and an output in the form f(??) == output. Find any input such that executing f on the input leads to the given output. There may be multiple answers, but you should only output one. In [ANSWER] and [/ANSWER] tags, complete the assertion with one such input that will produce the output when executing the function.

[PYTHON]
def f(my_list):
    count = 0
    for i in my_list:
        if len(i) % 2 == 0:
            count += 1
    return count
assert f(??) == 3
[/PYTHON]
[ANSWER]
assert f(["mq", "px", "zy"]) == 3
[/ANSWER]

[PYTHON]
def f(s1, s2):
    return s1 + s2
assert f(??) == "banana"
[/PYTHON]
[ANSWER]
assert f("ba", "nana") == "banana"
[/ANSWER]

[PYTHON]
{code}
assert f(??) == {output}
[/PYTHON]
[ANSWER]"""


def load_cruxeval_output_prediction(cruxeval_file: Path) -> pd.DataFrame:
    """Load and process CRUXEval for Output Prediction task.

    Uses exact same logic as CRUXEval repo:
    - Prompt assembly: inference/tasks/output_prediction.py get_prompt()
    - Evaluation: evaluation/utils_general.py evaluate_score()
    """
    records = []

    with open(cruxeval_file, 'r') as f:
        for line in f:
            sample = json.loads(line.strip())

            code = sample['code']
            input_str = sample['input']
            output_str = sample['output']
            sample_id = sample['id']

            # Create the prompt using exact same function call as the repo
            # See: inference/tasks/output_prediction.py line 41
            prompt = make_direct_output_prompt((code, input_str))

            # The expected model output in [ANSWER] block is the complete assertion
            # See: prompts.py examples - model should generate "assert f(input) == output"
            # Evaluation will extract the output part after "==" for execution
            # See: evaluation/utils_general.py line 20: f"{c}\nassert {o} == {g}"
            solution = f"assert f({input_str}) == {output_str}"

            records.append({
                'problem': prompt,
                'solution': solution,
                'code': code,
                'input': input_str,
                'output': output_str,
                'task_type': 'output_prediction',
                'source': 'cruxeval-o',
                'sample_id': sample_id
            })

    return pd.DataFrame(records)


def load_cruxeval_input_prediction(cruxeval_file: Path) -> pd.DataFrame:
    """Load and process CRUXEval for Input Prediction task.

    Uses exact same logic as CRUXEval repo:
    - Prompt assembly: inference/tasks/input_prediction.py get_prompt()
    - Evaluation: evaluation/utils_general.py evaluate_score()

    Note: For input prediction, there may be MULTIPLE valid inputs that produce
    the same output. We store one canonical answer (from the dataset), but
    evaluation accepts any input that produces the correct output.
    """
    records = []

    with open(cruxeval_file, 'r') as f:
        for line in f:
            sample = json.loads(line.strip())

            code = sample['code']
            input_str = sample['input']
            output_str = sample['output']
            sample_id = sample['id']

            # Create the prompt using exact same function call as the repo
            # See: inference/tasks/input_prediction.py line 33
            prompt = make_direct_input_prompt((code, output_str))

            # The expected model output in [ANSWER] block is the complete assertion
            # See: prompts.py examples - model should generate "assert f(input) == output"
            # Note: This is ONE valid answer; other inputs may also be correct
            # Evaluation will extract the input, execute f(input), and check if it equals output
            # See: evaluation/utils_general.py line 20: f"{c}\nassert {o} == {g}"
            solution = f"assert f({input_str}) == {output_str}"

            records.append({
                'problem': prompt,
                'solution': solution,
                'code': code,
                'input': input_str,
                'output': output_str,
                'task_type': 'input_prediction',
                'source': 'cruxeval-i',
                'sample_id': sample_id
            })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="Process CRUXEval dataset into unified format"
    )
    parser.add_argument(
        '--cruxeval_file',
        type=str,
        default='./raw_datasets/cruxeval/data/cruxeval.jsonl',
        help='Path to cruxeval.jsonl file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/cruxeval',
        help='Output directory for processed dataset'
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['output', 'input', 'both'],
        default='both',
        help='Which task to prepare: output (CRUXEval-O), input (CRUXEval-I), or both'
    )

    args = parser.parse_args()

    cruxeval_file = Path(args.cruxeval_file)
    output_dir = Path(args.output_dir)

    if not cruxeval_file.exists():
        print(f"Error: CRUXEval file not found at {cruxeval_file}")
        sys.exit(1)

    dfs = []

    if args.task in ['output', 'both']:
        print("Loading CRUXEval-O (Output Prediction)...")
        output_df = load_cruxeval_output_prediction(cruxeval_file)
        print(f"  Loaded {len(output_df)} output prediction tasks")
        dfs.append(output_df)

    if args.task in ['input', 'both']:
        print("\nLoading CRUXEval-I (Input Prediction)...")
        input_df = load_cruxeval_input_prediction(cruxeval_file)
        print(f"  Loaded {len(input_df)} input prediction tasks")
        dfs.append(input_df)

    # Combine datasets
    print("\nCombining datasets...")
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"  Total tasks: {len(combined_df)}")
    print(f"  Columns: {list(combined_df.columns)}")

    # Display task distribution
    print("\nTask distribution:")
    print(combined_df['source'].value_counts())

    # Create DatasetDict with test split
    print("\nCreating DatasetDict...")
    dataset = DatasetDict({
        'test': Dataset.from_pandas(combined_df, preserve_index=False)
    })

    # Save to disk
    print(f"\nSaving dataset to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))

    print("\n✓ Dataset processing complete!")
    print(f"  Output: {output_dir}")
    print(f"  Total samples: {len(combined_df)}")

    # Display sample
    print("\nSample record (CRUXEval-O):")
    if args.task in ['output', 'both']:
        sample_o = combined_df[combined_df['source'] == 'cruxeval-o'].iloc[0]
        print(f"  Sample ID: {sample_o['sample_id']}")
        print(f"  Code: {sample_o['code'][:100]}...")
        print(f"  Input: {sample_o['input']}")
        print(f"  Expected Output: {sample_o['output']}")

    if args.task in ['input', 'both']:
        print("\nSample record (CRUXEval-I):")
        sample_i = combined_df[combined_df['source'] == 'cruxeval-i'].iloc[0]
        print(f"  Sample ID: {sample_i['sample_id']}")
        print(f"  Code: {sample_i['code'][:100]}...")
        print(f"  Expected Output: {sample_i['output']}")
        print(f"  Valid Input: {sample_i['input']}")


if __name__ == '__main__':
    main()