from core.openai_engine import OpenAI_Engine
import pandas as pd
from pathlib import Path
import argparse
import re
import os
import time

PROMPT_TEMPLATE = """### System Prompt
You are an experienced examiner who evaluates whether a student's answer to a given question is correct. 
Your task is to determine if the student's final answer matches the standard answer provided, based solely on correctness and the question's specific requirements. 
Do not perform any additional calculations or reinterpret the question. Simply compare the student's answer to the standard answer to determine if it satisfies the question's requirements.

Focus strictly on:
1. Understanding the exact requirement of the question.
2. Comparing the student's final answer directly to the provided standard answer.
3. Your task is not to solve the problem but to determine whether the student's answer is correct based on the question's requirements. Avoid any unnecessary analysis, assumptions, or re-solving the problem.

Note:
- For intervals/ranges: The student's answer must cover the EXACT SAME range as the standard answer, NOT just any single value or subset within that range;
- If the standard answer contains multiple solutions connected by "or"/"and", all of them must be listed in the student's answer;
- If student's response does not mention any answer, it is considered WRONG;
- You must be deterministic - always declare the answer as either CORRECT or WRONG;

Your response must include:
### Short Analysis
Provide a brief (< 50 words) and direct analysis that compares the student's answer to the standard answer between <analysis> </analysis> tags.

### Correctness
At the end, You should report a label CORRECT or WRONG between <judge> </judge> tags (e.g., <judge>CORRECT</judge>).


### User Prompt
Question: {problem}

Standard Answer: {solution}

Student's Final Answer: {model_pred}
"""

class ModelJudge():
    def __init__(self,
                 input_df: pd.DataFrame,
                 problem_col: str,
                 solution_col: str,
                 response_col: str,
                 pred_col: str,
                 output_dir: str,
                 nick_name: str,
                 ):
        self.input_df = input_df
        self.problem_col = problem_col
        self.solution_col = solution_col
        self.response_col = response_col
        self.pred_col = pred_col
        self.output_dir = output_dir
        self.nick_name = nick_name
        
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_pred(self, row):
        pred = row[self.pred_col]
        if pred != "":
            return pred
        return row[self.response_col]    
    
    def extract_label(self, row):
        raw_response = row['model_judge']

        response = raw_response.lower().replace(' ', '').replace('\n', '').strip()
        pattern = r"<judge>(.*?)</judge>"

        match = re.search(pattern, response)
        if match and match.group(1) == "correct":
            return True
        elif match and match.group(1) == "wrong":
            return False
        elif "CORRECT" in raw_response and "WRONG" not in raw_response:
            return True
        elif "WRONG" in raw_response and "CORRECT" not in raw_response:
            return False
        else:
            return None

    def run(self, overwrite: bool = True) -> pd.DataFrame:
        self.eval_df = self.input_df.copy()
        self.eval_df['model_pred'] = self.eval_df.apply(self.extract_pred, axis=1)
        
        engine = OpenAI_Engine(
            input_df=self.eval_df,
            prompt_template=PROMPT_TEMPLATE,
            template_map={"problem": self.problem_col, "solution": self.solution_col, "model_pred": "model_pred"},
            nick_name=f"model_judge_{self.nick_name}",
            batch_io_root=str(Path.home()) + "/research/openai_batch_io/reasoning",
            cache_filepath=self.output_dir + f"/{self.nick_name}_model_judge.pickle",
            model = "deepseek-chat",
            client_name = "deepseek"
        )

        engine.run_model(overwrite=overwrite)
        self.response = engine.retrieve_outputs()
        if 'response' in self.response.columns:
            self.response = self.response.set_index('idx').rename(columns={'response': 'model_judge'})
            self.response = self.response.explode(['model_judge'])
        
        self.response['model_is_correct'] = self.response.apply(self.extract_label, axis=1)
        self.response.to_pickle(self.output_dir + f"/{self.nick_name}_model_judge.pickle")
        
        return self.response
    
    def merge(self) -> pd.DataFrame:
        self.eval_df = self.eval_df.merge(self.response[['model_judge', 'model_is_correct']], left_index=True, right_index=True)
    
        return self.eval_df
    
if __name__ == "__main__":
    """
    Example usage:
    python model-judge.py \
      --input_filepath ./results/allmath/benchmark/R1-Distill-Qwen-32B.pickle \
      --output_dir ./results/allmath/benchmark/model_judge \
      --nick_name R1-Distill-Qwen-32B

    python model-judge.py \
      --input_dir ./results/allmath/benchmark \
      --output_dir ./results/allmath/benchmark/model_judge
    """

    parser = argparse.ArgumentParser(
        description="Judge model predictions using an LLM."
    )
    parser.add_argument("--input_dir", type=str, required=False, help="Directory to read input")
    parser.add_argument("--input_filepath", type=str, required=False, help="Path to input pickle file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--nick_name", type=str, required=False, help="Nickname for this run.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs if set.")

    args = parser.parse_args()
    problem_col = "problem"
    solution_col = "solution"
    response_col = "post_corruption_response"
    pred_col = "pred"
    
    if args.input_filepath:
        input_df = pd.read_pickle(args.input_filepath)
        judge_engine = ModelJudge(
            input_df=input_df,
            problem_col=problem_col,
            solution_col=solution_col,
            response_col=response_col,
            pred_col=pred_col,
            output_dir=args.output_dir,
            nick_name=args.nick_name
        )
        judge_engine.run(overwrite=args.overwrite)
        result_df = judge_engine.merge()
        result_df.to_pickle(os.path.join(args.output_dir, f"{args.nick_name}_result.pickle"))
        
    elif args.input_dir:
        finished = []
        time_limit = 10 * 60 * 60 # 10 hours
        start_time = time.time()
        while True:
            if time.time() - start_time > time_limit:
                break

            for fname in os.listdir(args.input_dir):
                if fname.endswith(".pickle") and fname not in finished:
                    input_df = pd.read_pickle(os.path.join(args.input_dir, fname))
                    nick_name = fname.replace(".pickle", "")
                    
                    if not args.overwrite and "model_is_correct" in input_df.columns:
                        print(f"Skipping {nick_name} because it already exists")
                        finished.append(nick_name)
                        continue
                    
                    judge_engine = ModelJudge(
                        input_df=input_df,
                        problem_col=problem_col,
                        solution_col=solution_col,
                        response_col=response_col,
                        pred_col=pred_col,
                        output_dir=args.output_dir,
                        nick_name=nick_name
                    )
                    print(f"Starting to judge {nick_name}")
                    judge_engine.run(overwrite=args.overwrite)
                    result_df = judge_engine.merge()
                    result_df.to_pickle(os.path.join(args.input_dir, fname))
                    print(f"Finished judging {nick_name}")
            
            print(f"Waiting for 60 seconds before checking again")
            time.sleep(60)
                    