#!/usr/bin/env python3
"""
Process HumanEval, MBPP, and CRUXEval datasets into a unified format for code generation evaluation.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import Dataset, DatasetDict


def load_humaneval(raw_datasets_dir: Path) -> pd.DataFrame:
    """Load and process HumanEval dataset."""
    # Import the human_eval package
    import sys
    sys.path.insert(0, str(raw_datasets_dir / "human-eval"))
    from human_eval.data import read_problems

    problems = read_problems()

    records = []
    for task_id, data in problems.items():
        entry_point = data['entry_point']
        prompt = data['prompt']
        canonical_solution = data['canonical_solution']
        test = data['test']

        # Assemble the instruction template
        instruction = f'''You will complete a function whose docstring describes the required behavior. Return only the full function implementation, preserving the original function name {entry_point} as entry point, wrapped enclosed in ```python ```. Think step by step and implement and return the function.'''
        problem = instruction + f'\n\n```python\n{prompt}\n```'

        records.append({
            'problem': problem,
            'entry_point': entry_point,
            'canonical_answer': canonical_solution,
            'solution': test,
            'source': 'humaneval'
        })

    return pd.DataFrame(records)


def parse_entry_point_from_test(test_list: List[str]) -> str:
    """Parse the function name from the test assertions."""
    for test in test_list:
        # Look for function calls in assertions
        # Pattern: assert function_name(...) or function_name(...)
        match = re.search(r'(?:assert\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', test)
        if match:
            return match.group(1)

    # Fallback: return empty string if no function found
    return ""


def load_humaneval_plus() -> pd.DataFrame:
    """Load and process HumanEval+ dataset from HuggingFace."""
    from datasets import load_dataset

    dataset = load_dataset("evalplus/humanevalplus", split="test")

    records = []
    for sample in dataset:
        entry_point = sample['entry_point']
        prompt = sample['prompt']
        canonical_solution = sample['canonical_solution']
        test = sample['test']  # Enhanced test with 80x more inputs

        # Use same instruction template as original HumanEval
        instruction = f'''You will complete a function whose docstring describes the required behavior. Return only the full function implementation, preserving the original function name {entry_point} as entry point, wrapped enclosed in ```python ```. Think step by step and implement and return the function.'''
        problem = instruction + f'\n\n```python\n{prompt}\n```'

        records.append({
            'problem': problem,
            'entry_point': entry_point,
            'canonical_answer': canonical_solution,
            'solution': test,
            'source': 'humanevalplus'
        })

    return pd.DataFrame(records)


def load_mbpp_plus() -> pd.DataFrame:
    """Load and process MBPP+ dataset from HuggingFace."""
    from datasets import load_dataset

    dataset = load_dataset("evalplus/mbppplus", split="test")

    records = []
    for sample in dataset:
        problem_text = sample['prompt']  # MBPP+ uses 'prompt' instead of 'text'
        code = sample['code']
        test_list = sample['test_list']  # Original test assertions
        test = sample['test']  # Enhanced test function with 35x more inputs
        tests = '\n'.join(test_list)

        # Parse entry point from tests
        entry_point = parse_entry_point_from_test(test_list)

        # Use same prompt template as original MBPP
        prompt = f'''You will write a function according to the task for the required behavior. Return the full function implementation, with the same function name as in the tests as entry point, wrapped enclosed in ```python ```. Think step by step and implement and return the function.

Task: {problem_text}
Tests:
{tests}'''

        records.append({
            'problem': prompt,
            'entry_point': entry_point,
            'canonical_answer': code,
            'solution': test,  # Use enhanced test function
            'source': 'mbppplus'
        })

    return pd.DataFrame(records)


def make_direct_output_prompt(s):
    code, input = s
    return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. DO NOT output any extra information. Think step by step and predict the full assertion with the correct output wrapped in ```python ```, following the examples.

[PYTHON]
```python
def f(n):
    return n
assert f(17) == ??
```
[/PYTHON]
[ANSWER]
```python
def f(n):
    return n
assert f(17) == 17
```
[/ANSWER]

[PYTHON]
```python
def f(s):
    return s + "a"
assert f("x9j") == ??
```
[/PYTHON]
[ANSWER]
```python
def f(s):
    return s + "a"
assert f("x9j") == "x9ja"
```
[/ANSWER]

[PYTHON]
```python
{code}
assert f({input}) == ??
```
[/PYTHON]
"""


def load_cruxeval(raw_datasets_dir: Path) -> pd.DataFrame:
    """Load and process CRUXEval-O (Output Prediction) dataset."""
    cruxeval_file = raw_datasets_dir / "cruxeval" / "data" / "cruxeval.jsonl"

    records = []
    with open(cruxeval_file, 'r') as f:
        for line in f:
            sample = json.loads(line.strip())

            code = sample['code']
            input_str = sample['input']
            output_str = sample['output']

            # Create prompt using exact same logic as CRUXEval repo
            prompt = make_direct_output_prompt((code, input_str))

            # Expected solution: complete assertion
            solution = f"assert f({input_str}) == {output_str}"

            records.append({
                'problem': prompt,
                'entry_point': 'f',
                'canonical_answer': code,
                'solution': solution,
                'source': 'cruxeval'
            })

    return pd.DataFrame(records)


def load_mbpp(raw_datasets_dir: Path) -> pd.DataFrame:
    """Load and process MBPP dataset (Task IDs 11-510)."""
    mbpp_file = raw_datasets_dir / "mbpp" / "mbpp.jsonl"

    records = []
    with open(mbpp_file, 'r') as f:
        for line in f:
            example = json.loads(line.strip())
            task_id = example['task_id']

            # Only use Task IDs 11-510
            if task_id < 11 or task_id > 510:
                continue

            problem_text = example['text']
            code = example['code']
            test_list = example['test_list']
            tests = '\n'.join(test_list)

            # Parse entry point from tests
            entry_point = parse_entry_point_from_test(test_list)

            # Assemble the prompt template
            prompt = f'''You will write a function according to the task for the required behavior. Return the full function implementation, with the same function name as in the tests as entry point, wrapped enclosed in ```python ```. Think step by step and implement and return the function.

Task: {problem_text}
Tests:
{tests}'''

            # (DEPRECATED) Wrap test assertions in a check function (similar to MBPP+)
#             wrapped_assertions = '\n'.join(
#                 '    ' + assertion.replace(f'{entry_point}(', 'candidate(')
#                 for assertion in test_list
#             )
#             test_function = f"""def check(candidate):
# {wrapped_assertions}
# """
            assertions = '\n'.join(test_list)
            records.append({
                'problem': prompt,
                'entry_point': entry_point,
                'canonical_answer': code,
                'solution': assertions,
                'source': 'mbpp'
            })

    return pd.DataFrame(records)


def main():
    """
    Example usage:
    python data/prepare_dataset/allcode.py --raw_datasets_dir ./raw_datasets --output_dir ./data/allcode
    python data/prepare_dataset/allcode.py --raw_datasets_dir ./raw_datasets --output_dir ./data/cruxeval
    """
    parser = argparse.ArgumentParser(
        description="Process HumanEval, MBPP, CRUXEval, and EvalPlus datasets into unified format"
    )
    parser.add_argument(
        '--raw_datasets_dir',
        type=str,
        default='./raw_datasets',
        help='Path to raw_datasets directory containing human-eval, mbpp, and cruxeval'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/allcode',
        help='Output directory for processed dataset'
    )

    args = parser.parse_args()

    raw_datasets_dir = Path(args.raw_datasets_dir)
    output_dir = Path(args.output_dir)

    print("Loading HumanEval dataset...")
    humaneval_df = load_humaneval(raw_datasets_dir)

    print("Loading MBPP dataset (Task IDs 11-510)...")
    mbpp_df = load_mbpp(raw_datasets_dir)

    print("Loading CRUXEval dataset (Output Prediction)...")
    cruxeval_df = load_cruxeval(raw_datasets_dir)

    print("Loading HumanEval+ dataset...")
    humaneval_plus_df = load_humaneval_plus()

    print("Loading MBPP+ dataset...")
    mbpp_plus_df = load_mbpp_plus()

    combined_df = pd.concat([
        humaneval_df,
        mbpp_df,
        cruxeval_df,
        humaneval_plus_df,
        mbpp_plus_df
    ], ignore_index=True)
    print(f"  Total problems: {len(combined_df)}")

    dataset = DatasetDict({
        'test': Dataset.from_pandas(combined_df, preserve_index=False)
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))

    print("\n✓ Dataset processing complete!")
    print(f"  Output: {output_dir}")
    print(f"  Total samples: {len(combined_df)}")

if __name__ == '__main__':
    main()
