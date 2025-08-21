
import os
from datasets import DatasetDict, Dataset
import argparse
import shutil
import json
import pandas as pd

def create_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)
            
def rename_columns(row, source):
    if source == "aime24" or source == "aime25" or source == "amc23" or source == "math500":
        row['problem'] = str(row['problem'])
        row['solution'] = str(row['answer'])
    elif source == "minerva_math":
        row['problem'] = str(row['problem'])
        row['solution'] = str(row['solution'])
    elif source == "olympiadbench":
        row['problem'] = str(row['question'])
        solution = str(row['final_answer'][0])
        if solution[0] == '$' and solution[-1] == '$':
            solution = solution[1:-1]
        row['solution'] = solution

    row['source'] = source
    return row

def main():
    """
    python data/prepare_dataset/allmath.py \
        --output_dir ./data/allmath_plus \
        --dataset_dir ../rlvr/Qwen2.5-Eval/evaluation/data \
        --datasets olympiadbench math500
    """    
    parser = argparse.ArgumentParser(description='Prepare allmath dataset')
    parser.add_argument('--output_dir', default='./data/allmath', 
                       help='Output directory to save the dataset')
    parser.add_argument('--dataset_dir', default='../rlvr/Qwen2.5-Eval/evaluation/data', 
                       help='Dataset directory to save the dataset')
    parser.add_argument('--datasets', nargs='+', default=[], 
                       help='Datasets to prepare')
    args = parser.parse_args()
    
    output_dir = args.output_dir
    dataset_dir = args.dataset_dir
    dataset_names = args.datasets

    import pdb; pdb.set_trace()
    datasets = []
    for dataset_name in dataset_names:
        dataset_path = os.path.join(dataset_dir, f"{dataset_name}/test.jsonl")
        df = pd.DataFrame(list(read_jsonl(dataset_path)))
        if dataset_name == "math500":
            # we only keep level 5 question to ensure difficulty
            df = df[df['level'] != 5]
        df = df.apply(rename_columns, axis=1, args=(dataset_name,))
        datasets.append(df)

    final_dataset = pd.concat(datasets, ignore_index=True)[['problem', 'solution', 'source']]
    final_dataset = DatasetDict({
        'test': Dataset.from_pandas(final_dataset, preserve_index=False)
    })
    
    create_dir(output_dir)
    final_dataset.save_to_disk(output_dir)
    
    print(f"Dataset saved to: {output_dir}")

if __name__ == "__main__":
    main()
