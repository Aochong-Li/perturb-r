import os
import json
import pandas as pd
import argparse
import sys
import random
sys.path.append("/home/al2644/research")
from reward_score.math import *

def has_answer_digit(response: str) -> bool:
    try:
        _, answer = response.split('</think>')
        answer = remove_boxed(last_boxed_only_string(answer))
        return any(c.isdigit() for c in answer)
    except:
        return False

def filter_questions_stress_test (model_list: list,
                                  eval_dir: str,
                                  question_id: str="unique_id",
                                  min_correct: int=1,
                                  test_size: int=200,
                                  output_path="./results/questions_stress_test.json"
                                  ):
    random.seed(42)
    common_question_ids = set()
    benchmark_dir = os.path.join(eval_dir, "benchmark")
    for fname in os.listdir(benchmark_dir):
        if any(model in fname for model in model_list):
            df = pd.read_pickle(os.path.join(benchmark_dir, fname))
            df = df[df["response"].apply(has_answer_digit)]
            question_ids = df.groupby(question_id)[["correct"]].sum()
            question_ids = question_ids[question_ids["correct"] >= min_correct].index.tolist()

            if len(common_question_ids) == 0:
                common_question_ids = set(question_ids)
            else:
                common_question_ids = common_question_ids.intersection(set(question_ids))

    stress_test_sample = random.sample(list(common_question_ids), test_size)
    with open(output_path, "w") as f:
        json.dump(stress_test_sample, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_list", type=str, required=True)
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--question_id", type=str, required=True)
    parser.add_argument("--min_correct", type=int, required=True)
    parser.add_argument("--test_size", type=int, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    filter_questions_stress_test(args.model_list,
                                args.eval_dir,
                                args.question_id,
                                args.min_correct,
                                args.test_size,
                                args.output_path)