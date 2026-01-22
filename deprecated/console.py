import random
import pandas as pd
from datasets import Dataset, DatasetDict

import pandas as pd 
import os 

def merge_mmlu_and_science():
    def proc_mmlu(df):
        columns = ['question', 'choice_A', 'choice_B', 'choice_C',
        'choice_D', 'correct_answer', 'correct_answer_text', 'subject', 
            'problem', 'solution', 'response', 'pred',
        'ground_truth', 'model_is_correct'
                ]
        df = df[columns]
        df['source'] = 'mmlu'
        
        return df

    output_dir = './results/allscience/benchmark'

    for fname in os.listdir('./results/allscience/benchmark'):
        if '.pickle' not in fname:
            continue
            
        print(fname)
        science_df = pd.read_pickle(os.path.join('./results/allscience/benchmark/', fname))
        science_df = science_df[science_df['source'] != 'mmlu'].reset_index(drop=True)
        mmlu_df = pd.read_pickle(os.path.join('./results/mmlu/benchmark/', fname))

        mmlu_df = proc_mmlu(mmlu_df)
        
        df = pd.concat([science_df, mmlu_df], ignore_index=True)
        df.to_pickle(os.path.join(output_dir, fname))

def print_solve_rate_distribution():
    output_dir = './results/allscience/benchmark'
    for fname in os.listdir(output_dir):
        if '.pickle' not in fname:
            continue
            
        print(fname)
        df = pd.read_pickle(os.path.join(output_dir, fname))
        stats = df.groupby(['problem', 'source']).agg({'model_is_correct': 'sum'}).reset_index().rename(columns = {'model_is_correct': 'solve_n'})
        print(stats['solve_n'].value_counts())

if __name__ == "__main__":
    print_solve_rate_distribution()