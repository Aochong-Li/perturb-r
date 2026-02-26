
import os
from datasets import load_dataset, DatasetDict, concatenate_datasets
import argparse
import shutil

def create_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

def main():
    """
    python data/prepare_dataset/aime.py --output_dir ./data/aime2425
    """
    parser = argparse.ArgumentParser(description='Prepare AIME 2024-2025 dataset')
    parser.add_argument('--output_dir', default='./data/aime2425', 
                       help='Output directory to save the dataset')
    args = parser.parse_args()
    
    try:
        # Load the test split from the dataset
        aime24_dataset = load_dataset("Maxwell-Jia/AIME_2024")["train"].remove_columns(["ID", "Solution"])
        
        aime25_i_dataset = load_dataset("opencompass/AIME2025", "AIME2025-I")["test"]
        aime25_ii_dataset = load_dataset("opencompass/AIME2025", "AIME2025-II")["test"]
        aime25_dataset = concatenate_datasets([aime25_i_dataset, aime25_ii_dataset])

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    def rename_columns(example, source):
        if 'Problem' in example:
            example['problem'] = str(example.pop('Problem'))
        if 'Answer' in example:
            example['solution'] = str(example.pop('Answer'))

        if "question" in example:
            example['problem'] = str(example.pop('question'))
        if "answer" in example:
            example['solution'] = str(example.pop('answer'))
        example['source'] = source
        return example
    
    aime24_dataset = aime24_dataset.map(lambda x: rename_columns(x, "aime 24"))
    aime25_dataset = aime25_dataset.map(lambda x: rename_columns(x, "aime 25"))
    aime_dataset = concatenate_datasets([aime24_dataset, aime25_dataset])

    final_dataset = DatasetDict({
        'test': aime_dataset
    })
    # Create output directory if it doesn't exist
    create_dir(args.output_dir)
    
    # Save the dataset using save_to_disk
    final_dataset.save_to_disk(args.output_dir)
    
    print(f"Dataset saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
