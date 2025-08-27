import sys
import os
import pandas as pd
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.openai_engine import *
import re

PROMPT_TEMPLATE = """<instruction>
You are a professional, unbiased, and detailed examiner who evaluates whether a student's reasoning process explicitly contains the correct answer to a given question.

Focus strictly on:
    1. Understanding the exact requirement of the question: what value/quantity the question asks for, and why that value answers the question.
    2. Summarizing the student's reasoning procedures and the intermediate or final conclusions the student has derived.
    3. Your task is not to solve the problem, but to determine whether the student arrives at the solution at a certain point in the reasoning.

Note:
    1. If the student explicitly states the correct final value anywhere in the reasoning, mark it as CORRECT. The student does not need to express confidence or commit to the final answer. If multiple candidate answers are listed and one equals the ground-truth answer, it is CORRECT.
    2. If the student's reasoning only implies the solution value or on the correct trajectory but has not explicitly stated it, mark it as WRONG.
    3. Small differences in rounding, equivalent units, or algebraic expressions are permitted as long as they are clearly equivalent to the ground-truth answer.
</instruction>
<format>
Your response must include ONLY the following sections:
1. Analysis: From start to end of the reasoning, provide a summary of each major stage and what conclusions/results the student has derived. State whether the correct answer is explicitly mentioned. Do not omit steps directly related to the solution. Put the summary between <analysis> and </analysis> tags.
2. Correctness: Based on the analysis, output CORRECT or WRONG between <judge> and </judge> tags (e.g., <judge>CORRECT</judge> or <judge>WRONG</judge>).
</format>

###Problem: {problem}

###Ground Truth Answer: {solution}

###Student Reasoning: {reasoning}"""

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
You are an unbiased examiner who evaluates whether a student's answer to a given question is correct. 
Your task is to determine if the student's final answer matches the standard answer provided, based solely on correctness and the question's specific requirements. 
Do not perform any additional calculations or reinterpret the question. Simply compare the student's answer to the standard answer to determine if it satisfies the question's requirements.

Focus strictly on:
1. Understanding the exact requirement of the question.
2. Comparing the student's final answer directly and rigorously to the provided standard answer.
3. Your task is not to solve the problem but to determine whether the student's answer is correct based on the question's requirements. Avoid any unnecessary analysis, assumptions, or re-solving the problem.

Note:
- The student may propose multiple independent answer candidates separated by comma. As long as one of them matches the standard answer, declare as CORRECT.
- Do not infer the standard answer from the student answers. If the standard answer is not in student's answers, declare as WRONG.
- If student's response does not mention any answer, it is considered WRONG;
- You must be deterministic and rigorous - always declare the answer as either CORRECT or WRONG;
- Small rounding differences are permitted given the value asked by the problem.

Your response must include:
### Short Analysis
Provide a short and evidence-backed analysis between <analysis> </analysis> tags, in which you should judge whether the standard answer is one of the student's answers.

### Correctness
Based on the analysis, you should report a label CORRECT or WRONG between <judge> </judge> tags (e.g., <judge>CORRECT</judge> or <judge>WRONG</judge>).

### User Prompt
Problem: {problem}

Standard Answer: {standard_answer}

Student Answer: {student_answer}'''


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

def extract_teacher_answer(
    input_dir: str,
    output_dir: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    overwrite: bool = False
    ):
    import pdb; pdb.set_trace()
    os.makedirs(output_dir, exist_ok=True)
    fnames = [f for f in os.listdir(input_dir) if f.endswith(".parquet")]

    teacher_df  = []
    for fname in fnames:
        df = pd.read_parquet(os.path.join(input_dir, fname))
        teacher_df.append(df)
    teacher_df = pd.concat(teacher_df).drop_duplicates(subset=['problem', 'teacher', 'ratio', 'teacher_reasoning']).reset_index(drop=True)
    teacher_df = teacher_df[['problem', 'source', 'solution', 'teacher', 'cross_tier', 'ratio', 'teacher_reasoning']]
    teacher_df.to_parquet(os.path.join(output_dir, "teacher_reasoning_pool.parquet"))
    
    engine = OpenAI_Engine(
        input_df=teacher_df,
        prompt_template=EXTRACT_PROMPT_TEMPLATE,
        template_map={
            "problem": "problem",
            "solution": "solution",
            "reasoning": "teacher_reasoning"
        },
        nick_name="teachability_if_answer",
        batch_io_root="/home/al2644/research/openai_batch_io/reasoning",
        cache_filepath=os.path.join(output_dir, "extract_teacher_answer_response.pickle"),
        model="deepseek-chat",
        client_name="deepseek",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # engine.run_model(overwrite=overwrite)
    response = engine.retrieve_outputs()
    
    if 'response' in response.columns:
        response = response.set_index('idx').rename(columns={'response': 'model_judge'})
        response = response.explode(['model_judge'])

    response['answer'] = response.apply(extract_answer, axis=1)
    teacher_df = teacher_df.merge(response[["model_judge", "answer"]], left_index=True, right_index=True)
    teacher_df.to_parquet(os.path.join(output_dir, "extract_teacher_answer.parquet"))

def judge_teacher_answer(
    input_dir: str,
    output_dir: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    overwrite: bool = False
    ):
    answer_df = pd.read_parquet(os.path.join(input_dir, "extract_teacher_answer.parquet"))
    engine = OpenAI_Engine(
        input_df=answer_df,
        prompt_template=JUDGE_PROMPT_TEMPLATE,
        template_map={
            "problem": "problem",
            "standard_answer": "solution",
            "student_answer": "answer"
        },
        nick_name="teachability_judge_answer",
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
    answer_df = answer_df.rename(columns = {'model_judge': 'model_extract'})
    answer_df = answer_df.merge(response[["model_judge","contains_answer"]], left_index=True, right_index=True)
    
    answer_df.to_parquet(os.path.join(output_dir, "teachability_judge_answer.parquet"))

if __name__ == "__main__":
    """
    python stress-test/ablation/teachability_if_answer.py \
        --input_dir /home/al2644/research/codebase/reasoning/perturb-r/results/allmath/teacher_guide \
        --output_dir /home/al2644/research/codebase/reasoning/perturb-r/results/allmath/teacher_guide/teachability_contains_answer \
        --temperature 0.7 \
        --max_tokens 512 \
        --overwrite
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    extract_teacher_answer(**vars(args))
    # judge_teacher_answer(**vars(args))
