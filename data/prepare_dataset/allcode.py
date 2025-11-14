#!/usr/bin/env python3
"""
Process HumanEval and MBPP datasets into a unified format for code generation evaluation.
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
        instruction = f'''You will complete a function whose docstring describes the required behavior. Return only the full function implementation, preserving the original function name {entry_point} as entry point, wrapped between ```python and ```. Provide no explanations or extra text—only the code.'''
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
            prompt = f'''You will write a function according to the task for the required behavior. Return only the full function implementation, with the same function name as in the tests as entry point, wrapped between ```python and ```. Provide no explanations or extra text. Just a single code snippet.

Task: {problem_text}
Tests:
{tests}'''

            records.append({
                'problem': prompt,
                'entry_point': entry_point,
                'canonical_answer': code,
                'solution': tests,
                'source': 'mbpp'
            })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="Process HumanEval and MBPP datasets into unified format"
    )
    parser.add_argument(
        '--raw_datasets_dir',
        type=str,
        default='./raw_datasets',
        help='Path to raw_datasets directory containing human-eval and mbpp'
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
    print(f"  Loaded {len(humaneval_df)} HumanEval problems")

    print("\nLoading MBPP dataset (Task IDs 11-510)...")
    mbpp_df = load_mbpp(raw_datasets_dir)
    print(f"  Loaded {len(mbpp_df)} MBPP problems")

    # Concatenate both datasets
    print("\nCombining datasets...")
    combined_df = pd.concat([humaneval_df, mbpp_df], ignore_index=True)
    print(f"  Total problems: {len(combined_df)}")
    print(f"  Columns: {list(combined_df.columns)}")

    # Display source distribution
    print("\nSource distribution:")
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
    print("\nSample record (HumanEval):")
    sample_he = combined_df[combined_df['source'] == 'humaneval'].iloc[0]
    print(f"  Entry point: {sample_he['entry_point']}")
    print(f"  Problem preview: {sample_he['problem'][:200]}...")

    print("\nSample record (MBPP):")
    sample_mbpp = combined_df[combined_df['source'] == 'mbpp'].iloc[0]
    print(f"  Entry point: {sample_mbpp['entry_point']}")
    print(f"  Problem preview: {sample_mbpp['problem'][:200]}...")


if __name__ == '__main__':
    main()
