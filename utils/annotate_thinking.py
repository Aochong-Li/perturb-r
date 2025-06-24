import os
import pandas as pd
from openai_engine import OpenAI_Engine
from utils.process_thinking import chunk as chunk_tool
from reward_score.math import *
import re
import argparse

class ReasoningAnnotator:
    def __init__(self,
                 nick_name: str,
                 reasoning_col: str = "response",
                 results_dir: str = None,
                 granuality: int = 30
    ):
        self.nick_name = nick_name
        self.reasoning_col = reasoning_col
        self.results_dir = results_dir
        self.granuality = granuality

        self.load_dataset()
    
    def load_dataset(self):
        self.df = pd.read_pickle(os.path.join(self.results_dir, "benchmark", f"{self.nick_name}.pickle"))

    def locate_1st_final_answer(self):
        def process_response(row):
            response = row[self.reasoning_col]
            if "</think>" not in response:
                return None, None
            reasoning, _ = response.split("</think>")

            chunks = chunk_tool(reasoning, self.granuality)
            prompt = """You are given a sequence of reasoning steps, divided into chunks, representing how a model solves a problem. Your task is to identify the chunk number where the final answer is first derived. Even if later chunks verify or try alternative methods, only mark the first chunk where the correct final answer appears.

You MUST return the chunk number using the format:
<answer_chunk>CHUNK_NUMBER</answer_chunk>
This format is required for downstream processing."""

            for idx, chunk in enumerate(chunks):
                prompt += f"###START OF CHUNK {idx}\n"
                prompt += chunk
                prompt += f"\n###END OF CHUNK {idx}\n"
            
            return chunks, prompt
        
        def extract_chunk_number(response: list[str]):
            response = response[0]

            pattern = r"<answer_chunk>(.*?)</answer_chunk>"
            match = re.search(pattern, response)
            if match:
                chunk_number = match.group(1)
                try:
                    return int(chunk_number)
                except:
                    return chunk_number
            else:
                return None
        
        self.df["chunks"], self.df["prompt"] = zip(*self.df.apply(process_response, axis = 1))
        self.df = self.df.dropna(subset=["chunks", "prompt"])

        cache_output_dir = os.path.join(self.results_dir, f"annotated_reasoning")
        
        if not os.path.exists(cache_output_dir):
            os.makedirs(cache_output_dir)

        self.locate_answer_engine = OpenAI_Engine(
            model="gpt-4.1-mini",
            input_df=self.df,
            prompt_template="{prompt}",
            template_map={"prompt": "prompt"},
            nick_name=f"locate_1st_final_answer_{self.nick_name}",
            cache_filepath=os.path.join(cache_output_dir, f"{self.nick_name}.pickle"),
        )

        self.locate_answer_engine.run_model()
        self.outputs = self.locate_answer_engine.retrieve_outputs().rename(columns={"response": "1st_answer_annotation_raw"})
        self.outputs["1st_answer_chunk_index"] = self.outputs["1st_answer_annotation_raw"].apply(extract_chunk_number)
        self.outputs.index = self.df.index
        self.df = self.df.merge(self.outputs, left_index=True, right_index=True)
        
        self.df.to_pickle(os.path.join(cache_output_dir, f"{self.nick_name}.pickle"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nick_name", type=str, required=True)
    parser.add_argument("--reasoning_col", type=str, default="response")
    parser.add_argument("--results_dir", type=str, default="./results/aime2425")
    args = parser.parse_args()

    annotator = ReasoningAnnotator(
        **vars(args)
    )
    import ipdb; ipdb.set_trace()
    annotator.locate_1st_final_answer()


