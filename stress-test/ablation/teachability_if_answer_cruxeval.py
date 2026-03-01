import sys
import os
import pandas as pd
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.openai_engine import *
import re

EXTRACT_PROMPT_TEMPLATE = '''<instruction>
You are a professional, unbiased, and detailed exam grader who extracts answers explicitly written by the student within their derivation. You never solve or continue the derivation yourself.

Focus strictly on:
    1. Understand what value/quantity/expression the problem asks for.
    2. Locate and report every candidate answer the student explicitly states (they need not commit or be confident).
    3. Do not derive, infer, simplify, or complete missing steps. Extract only what is written.
    4. Even if the student is close to a conclusion, do not add new information—extract only stated answers.
</instruction>
<format>
Put the answer (or all candidates) written by the student between <answer> </answer> tags.

Rules:
    • The content between the tags must be less than 20 words total.
    • Separate multiple candidates with commas
    • Keep numbers/units/expressions exactly as written.
    • If no explicit candidate is stated, output <answer>No Answer Found</answer>.
    • Do not include explanations, analysis, or any other text outside the tags.
</format>

### Problem: {problem}

### Student Derivation: {reasoning}'''

JUDGE_PROMPT_TEMPLATE = '''### System Prompt
You are an unbiased examiner who evaluates whether a student's guesses to a given question are correct.
Your task is to determine if the student's guesses match the standard answer provided, based solely on correctness and the question's specific requirements.
Do not perform any additional calculations or reinterpret the question. Simply compare the student's guesses to the standard answer to determine if it satisfies the question's requirements.

Focus strictly on:
1. Understanding the exact requirement of the question.
2. Comparing the student's guesses directly and rigorously to the provided standard answer.
3. Your task is not to solve the problem but to determine whether the student's guesses is correct based on the question's requirements. Avoid any unnecessary analysis, assumptions, or re-solving the problem.

Note:
- The student might make multiple guesses, but the standard answer MUST be one of the guesses;
- For intervals/ranges: The question may ask for intervals, ranges, or multiple values.The student guesses must cover the EXACT range as the standard answer, NOT just any single value or subset within that range of standard answer;
- If the standard answer contains multiple solutions connected by "or"/"and", all of them must be listed in the student's guesses;
- If student's response does not mention any answer, it is considered WRONG;
- You must be deterministic and rigorous - always declare the guesses as either CORRECT or WRONG

Your response must include:
### Short Analysis
Provide a short and evidence-backed analysis between <analysis> </analysis> tags, in which you should extract the final solution value from the standard answer and the student's answer and judge whether they are the same.

### Correctness
Based on the analysis, you should report a label CORRECT or WRONG between <judge> </judge> tags (e.g., <judge>CORRECT</judge> or <judge>WRONG</judge>).

### User Prompt
Problem: {problem}

Standard Answer: {standard_answer}

Student Guesses: {student_answer}'''


def extract_label(row):
    raw_response = row['model_judge']

    response = raw_response.lower().replace(' ', '').replace('\n', '').strip()
    pattern = r"<judge>(.*?)</judge>"

    match = re.search(pattern, response)
    if match and match.group(1) == "correct":
        return 1.0
    elif match and match.group(1) == "wrong":
        return 0.0
    elif "CORRECT" in raw_response and "WRONG" not in raw_response:
        return 1.0
    elif "WRONG" in raw_response and "CORRECT" not in raw_response:
        return 0.0
    else:
        return 0.0

def extract_answer(row):
    raw_response = row['model_judge']
    response = raw_response.lower().replace(' ', '').replace('\n', '').strip()
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, response)
    if match:
        return match.group(1)
    else:
        return "No Answer Found"

def prepare_data(
    input_dir: str,
    output_dir: str,
    ):
    """Load coding teacher_guide pickles, filter to cruxeval, and save as parquet."""
    os.makedirs(output_dir, exist_ok=True)
    fnames = [f for f in os.listdir(input_dir) if f.endswith(".pickle")]

    teacher_dfs = []
    for fname in fnames:
        df = pd.read_pickle(os.path.join(input_dir, fname))
        df = df[df['source'] == 'cruxeval']
        if len(df) > 0:
            teacher_dfs.append(df)

    teacher_df = pd.concat(teacher_dfs).reset_index(drop=True)
    teacher_df['cross_tier'] = ''
    teacher_df = teacher_df.drop_duplicates(
        subset=['problem', 'teacher', 'ratio', 'teacher_reasoning']
    ).reset_index(drop=True)
    teacher_df = teacher_df[['problem', 'source', 'solution', 'teacher', 'cross_tier', 'ratio', 'teacher_reasoning']]
    teacher_df.to_parquet(os.path.join(output_dir, "teacher_reasoning_pool.parquet"))
    print(f"Prepared {len(teacher_df)} cruxeval rows -> {os.path.join(output_dir, 'teacher_reasoning_pool.parquet')}")

def extract_teacher_answer(
    output_dir: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    overwrite: bool = False
    ):
    """Extract candidate answers from teacher reasoning traces using LLM."""
    teacher_df = pd.read_parquet(os.path.join(output_dir, "teacher_reasoning_pool.parquet"))

    engine = OpenAI_Engine(
        input_df=teacher_df,
        prompt_template=EXTRACT_PROMPT_TEMPLATE,
        template_map={
            "problem": "problem",
            "solution": "solution",
            "reasoning": "teacher_reasoning"
        },
        nick_name="teachability_if_answer_cruxeval",
        batch_io_root="/home/al2644/research/openai_batch_io/reasoning",
        cache_filepath=os.path.join(output_dir, "extract_teacher_answer_response.pickle"),
        model="deepseek-chat",
        client_name="deepseek",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    engine.run_model(overwrite=overwrite)
    response = engine.retrieve_outputs()

    if 'response' in response.columns:
        response = response.set_index('idx').rename(columns={'response': 'model_judge'})
        response = response.explode(['model_judge'])

    response['answer'] = response.apply(extract_answer, axis=1)
    teacher_df = teacher_df.merge(response[["model_judge", "answer"]], left_index=True, right_index=True)
    teacher_df.to_parquet(os.path.join(output_dir, "extract_teacher_answer.parquet"))
    print(f"Extracted answers -> {os.path.join(output_dir, 'extract_teacher_answer.parquet')}")

def judge_teacher_answer(
    output_dir: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    overwrite: bool = False
    ):
    """Judge whether extracted answers match ground truth."""
    answer_df = pd.read_parquet(os.path.join(output_dir, "extract_teacher_answer.parquet"))
    engine = OpenAI_Engine(
        input_df=answer_df,
        prompt_template=JUDGE_PROMPT_TEMPLATE,
        template_map={
            "problem": "problem",
            "standard_answer": "solution",
            "student_answer": "answer"
        },
        nick_name="teachability_judge_answer_cruxeval",
        batch_io_root="/home/al2644/research/openai_batch_io/reasoning",
        cache_filepath=os.path.join(output_dir, "teachability_judge_answer_response.pickle"),
        model="deepseek-chat",
        client_name="deepseek",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    engine.run_model(overwrite=overwrite)
    response = engine.retrieve_outputs()

    if 'response' in response.columns:
        response = response.set_index('idx').rename(columns={'response': 'model_judge'})
        response = response.explode(['model_judge'])

    response['contains_answer'] = response.apply(extract_label, axis=1)
    answer_df = answer_df.rename(columns={'model_judge': 'model_extract'})
    answer_df = answer_df.merge(response[["model_judge", "contains_answer"]], left_index=True, right_index=True)

    answer_df.to_parquet(os.path.join(output_dir, "teachability_judge_answer.parquet"))
    print(f"Judged answers -> {os.path.join(output_dir, 'teachability_judge_answer.parquet')}")

if __name__ == "__main__":
    """
    python stress-test/ablation/teachability_if_answer_cruxeval.py \
        --input_dir results/allcode/teacher_guide \
        --output_dir results/allcode/teacher_guide/teachability_contains_answer \
        --stage all \
        --temperature 0.7 \
        --max_tokens 512
    """
    parser = argparse.ArgumentParser(
        description="CruxEval teachability answer-leakage detection pipeline"
    )
    parser.add_argument("--input_dir", type=str,
                        default="results/allcode/teacher_guide",
                        help="Directory containing coding teacher_guide pickle files")
    parser.add_argument("--output_dir", type=str,
                        default="results/allcode/teacher_guide/teachability_contains_answer",
                        help="Directory to save output parquet files")
    parser.add_argument("--stage", type=str, choices=["prepare", "extract", "judge", "all"],
                        default="all",
                        help="Pipeline stage to run")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.stage in ("prepare", "all"):
        prepare_data(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )

    if args.stage in ("extract", "all"):
        extract_teacher_answer(
            output_dir=args.output_dir,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            overwrite=args.overwrite,
        )

    if args.stage in ("judge", "all"):
        judge_teacher_answer(
            output_dir=args.output_dir,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            overwrite=args.overwrite,
        )
