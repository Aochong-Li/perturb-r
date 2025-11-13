# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from gpqa.py parsing logic for GPQA-style science questions
import argparse
import pandas as pd
import os
import re
from typing import Optional
from math_eval import math_if_boxed, last_boxed_only_string, remove_boxed

def parse_answer(response: str) -> Optional[str]:
    """Extract answer choice (A-J) from model response.

    Uses the same parsing logic as GPQA's parse_sampled_answer.

    Args:
        response: The model's response string

    Returns:
        The parsed answer letter (A-J) or None if parsing failed
    """
    if not response:
        return None
    
    # Pre-processing response
    if math_if_boxed(response):
        response = remove_boxed(last_boxed_only_string(response))
        response = f"ANSWER IS ({response})"
    
    response = response.upper().replace("**", "")
    # Valid answer choices for GPQA
    VALID_CHOICES = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}

    # Try multiple patterns (from GPQA baselines/run_baseline.py)
    patterns = [
        r'ANSWER IS \(?([A-J])\)?',
        r'ANSWER: \(?([A-J])\)?',
        r'ANSWER \(?([A-J])\)?',
        r'\(([A-J])\)'
    ]

    for pattern in patterns:
        match = re.search(pattern, response)
        if match and match.group(1) in VALID_CHOICES:
            return match.group(1)

    if 'ANSWER IS' in response:
        return response.split('ANSWER IS')[-1].strip()
    elif 'ANSWER:' in response:
        return response.split('ANSWER:')[-1].strip()

    return None

def science_verify_score(solution_str: str, response_str: str = None, ground_truth: str = None, ground_truth_text: str = None) -> float:
    """Verify if the parsed solution matches the ground truth for GPQA-style questions.

    Args:
        solution_str: The parsed solution (should be A/B/C/D)
        ground_truth: The correct answer (should be A/B/C/D)
        response_str: The full response string (used if solution_str is None)

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    if solution_str:
        answer = parse_answer(solution_str)
    elif response_str:
        answer = parse_answer(response_str)
    else:
        return 0.0
    
    if not answer:
        return None
    elif answer.strip().upper() == ground_truth.strip().upper():
        return 1.0
    elif answer.strip().upper() == ground_truth_text.strip().upper():
        return 1.0
    else:
        return 0.0

if __name__ == "__main__":
    """
    Usage:
    conda activate zero
    python reward_score/science_eval.py --input_dir ./results/gpqa_diamond/benchmark --overwrite
    python reward_score/science_eval.py --input_dir ./results/mmlu_redux/benchmark --overwrite

    python reward_score/science_eval.py --input_dir ./results/gpqa_diamond/inject_distractor
    python reward_score/science_eval.py --input_dir ./results/mmlu_redux/inject_distractor
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=False)
    parser.add_argument("--input_dir", type=str, required=False)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source = args.input_dir if args.input_dir else args.file_path

    pred_col = "pred"
    gt_col = "ground_truth"
    if "benchmark" in source:
        response_col = "response"
    elif "distract" in source:
        response_col = "post_distraction_response"
    elif "teacher" in source:
        response_col = "student_response"

    if_strict_answer = True
    ground_truth_text_col = "correct_answer_text"

    if args.file_path:
        df = pd.read_pickle(args.file_path)
        if "gt" in df.columns:
            df = df.rename(columns={"gt": gt_col})
            df.to_pickle(args.file_path)

        df["model_is_correct"] = df.apply(
            lambda x: science_verify_score(x[pred_col], x[response_col], x[gt_col]),
            axis=1
        )
        df.to_pickle(args.file_path)

        # Print summary statistics
        accuracy = df["model_is_correct"].mean()
        total = len(df)
        correct = df["model_is_correct"].sum()
        print(f"Accuracy: {accuracy:.2%} ({int(correct)}/{total})")

    else:
        for fname in os.listdir(args.input_dir):
            if fname.endswith(".pickle"):
                print("Evaluating {}".format(fname))
                df = pd.read_pickle(os.path.join(args.input_dir, fname))
                if "gt" in df.columns:
                    df = df.rename(columns={"gt": gt_col})
                    df.to_pickle(os.path.join(args.input_dir, fname))

                if ("model_is_correct" in df.columns
                    or "original_correct" in df.columns
                    or "distractor_correct" in df.columns) and not args.overwrite:
                    print("Skipping {} because it already has model_is_correct column".format(fname))
                    continue

                if "distract" in source:
                    df["original_correct"] = df.apply(
                        lambda x: science_verify_score(x["pred"], x["post_distraction_response"], x["solution"], x[ground_truth_text_col]),
                        axis=1
                    )
                    df["distractor_correct"] = df.apply(
                        lambda x: science_verify_score(x["pred"], x["post_distraction_response"], x["distractor_solution"], x[ground_truth_text_col]),
                        axis=1
                    )
                else:
                    df["model_is_correct"] = df.apply(
                        lambda x: science_verify_score(x[pred_col], x[response_col], x[gt_col], x[ground_truth_text_col]),
                        axis=1
                    )

                is_correct_col = "model_is_correct" if "model_is_correct" in df.columns else "original_correct"
                if if_strict_answer:
                    # If response equals pred (no reasoning), mark as incorrect
                    df.loc[df[response_col] == df[pred_col], is_correct_col] = 0.0

                df.to_pickle(os.path.join(args.input_dir, fname))

                # Print summary statistics
                accuracy = df[is_correct_col].mean()
                total = len(df)
                correct = df[is_correct_col].sum()
                print(f"{fname} - Accuracy: {accuracy:.2%} ({int(correct)}/{total})")
