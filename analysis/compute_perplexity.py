"""
Compute perplexity of teacher reasoning sequences from dataframes.

Usage:
    python compute_perplexity.py --nickname R1-Distill-Qwen-1.5B --batch_size 8
"""

import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import numpy as np
import torch.multiprocessing as mp


def find_teacher_reasoning_indices(prompt, tokenizer):
    """Find token indices for teacher reasoning within the prompt."""
    full_tokens = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)

    template_result = tokenizer.apply_chat_template(
        [{"role": "user", "content": "HANDLE"}],
        tokenize=False,
        add_generation_prompt=True
    )
    _, template_suffix = template_result.split("HANDLE")

    input_prompt_text, _ = prompt.split(template_suffix, 1)

    input_prompt_tokens = tokenizer(
        input_prompt_text + template_suffix,
        return_tensors='pt',
        add_special_tokens=False
    )
    input_length = len(input_prompt_tokens['input_ids'][0])
    prompt_length = len(full_tokens['input_ids'][0])

    return {
        'teacher_indices': list(range(input_length, prompt_length)),
        'full_tokens': full_tokens
    }


def preprocess_all_data(df, tokenizer, max_batch_tokens):
    """Preprocess all data and prepare batches upfront."""
    print("Preprocessing all data and creating batches...")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Step 1: Tokenize all prompts and find teacher indices
    all_boundaries = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
        boundary = find_teacher_reasoning_indices(row['prompt'], tokenizer)
        all_boundaries.append(boundary)

    # Step 2: Group into batches based on max_batch_tokens
    batches = []
    current_batch_indices = []
    current_max_len = 0

    for idx, boundary in enumerate(all_boundaries):
        seq_len = boundary['full_tokens']['input_ids'].shape[1]

        # Estimate memory: (batch_size + 1) × max(current_max, new_seq_len)
        new_max_len = max(current_max_len, seq_len)
        estimated_tokens = (len(current_batch_indices) + 1) * new_max_len

        # If adding this sample exceeds limit and batch is not empty, save batch
        if current_batch_indices and estimated_tokens > max_batch_tokens:
            batches.append(current_batch_indices)
            current_batch_indices = [idx]
            current_max_len = seq_len
        else:
            current_batch_indices.append(idx)
            current_max_len = new_max_len

    # Don't forget the last batch
    if current_batch_indices:
        batches.append(current_batch_indices)

    # Step 3: Prepare padded tensors for each batch
    prepared_batches = []
    print(f"Preparing {len(batches)} batches...")
    for batch_indices in tqdm(batches, desc="Padding batches"):
        batch_boundaries = [all_boundaries[idx] for idx in batch_indices]
        max_len = max(b['full_tokens']['input_ids'].shape[1] for b in batch_boundaries)

        padded_input_ids = []
        padded_attention_masks = []
        teacher_indices_batch = []

        for boundary in batch_boundaries:
            tokens = boundary['full_tokens']
            seq_len = tokens['input_ids'].shape[1]
            padding_len = max_len - seq_len

            padded_input_ids.append(torch.cat([
                tokens['input_ids'],
                torch.full((1, padding_len), pad_token_id, dtype=tokens['input_ids'].dtype)
            ], dim=1))

            padded_attention_masks.append(torch.cat([
                tokens['attention_mask'],
                torch.zeros(1, padding_len, dtype=tokens['attention_mask'].dtype)
            ], dim=1))

            teacher_indices_batch.append(boundary['teacher_indices'])

        tokens_batch = {
            'input_ids': torch.cat(padded_input_ids, dim=0),
            'attention_mask': torch.cat(padded_attention_masks, dim=0)
        }

        prepared_batches.append((tokens_batch, teacher_indices_batch))

    print(f"Preprocessing complete! Created {len(prepared_batches)} batches.")
    return prepared_batches


def process_data_on_gpu(gpu_id, model_name, df_subset, max_batch_tokens):
    """Process a subset of data on a specific GPU.

    Args:
        gpu_id: GPU device ID
        model_name: HuggingFace model name
        df_subset: Subset of dataframe to process
        max_batch_tokens: Maximum tokens per batch

    Returns:
        List of perplexities for the subset
    """
    # Set device for this process
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')

    # Load tokenizer and model on this GPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={'': gpu_id}  # Load model entirely on this GPU
    )
    model.eval()

    print(f"GPU {gpu_id}: Processing {len(df_subset)} samples")

    # Preprocess data for this GPU
    prepared_batches = preprocess_all_data(df_subset, tokenizer, max_batch_tokens)

    # Process batches
    perplexities = []
    for tokens_batch, teacher_indices_batch in tqdm(
        prepared_batches,
        desc=f"GPU {gpu_id}",
        position=gpu_id
    ):
        with torch.no_grad():
            input_ids_gpu = tokens_batch['input_ids'].to(device, non_blocking=True)
            attention_mask_gpu = tokens_batch['attention_mask'].to(device, non_blocking=True)

            outputs = model(input_ids=input_ids_gpu, attention_mask=attention_mask_gpu)
            logits = outputs.logits
            del outputs

            # Process batch on GPU
            batch_perplexities = process_batch_gpu(
                logits, input_ids_gpu, teacher_indices_batch, device
            )
            perplexities.extend(batch_perplexities)

            # Cleanup GPU memory
            del logits, input_ids_gpu, attention_mask_gpu
            torch.cuda.empty_cache()

    return perplexities


def compute_perplexities_for_dataframe(model, tokenizer, df, device='cuda', max_batch_tokens=32768, use_data_parallel=False, model_name=None):
    """Compute perplexity for teacher reasoning with preprocessed batches.

    Args:
        max_batch_tokens: Maximum tokens allowed in batch (batch_size × max_seq_len)
        use_data_parallel: If True and multiple GPUs available, use data parallelism
        model_name: Model name (required for data parallelism)
    """
    # Check if we should use data parallelism
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if use_data_parallel and num_gpus > 1 and model_name is not None:
        print(f"Using data parallelism across {num_gpus} GPUs")
        return compute_perplexities_data_parallel(model_name, df, num_gpus, max_batch_tokens)

    # Single GPU mode - preprocess all data upfront
    prepared_batches = preprocess_all_data(df, tokenizer, max_batch_tokens)

    # Single GPU or model parallelism mode
    # For multi-GPU setups with device_map, find the first device
    if hasattr(model, 'hf_device_map'):
        first_device = list(model.hf_device_map.values())[0]
    else:
        first_device = device

    # Process batches
    perplexities = []
    print("Running inference...")

    for tokens_batch, teacher_indices_batch in tqdm(prepared_batches, desc="GPU inference"):
        # GPU forward pass
        with torch.no_grad():
            input_ids_gpu = tokens_batch['input_ids'].to(first_device, non_blocking=True)
            attention_mask_gpu = tokens_batch['attention_mask'].to(first_device, non_blocking=True)

            outputs = model(input_ids=input_ids_gpu, attention_mask=attention_mask_gpu)
            logits = outputs.logits
            del outputs

            # All processing on GPU
            batch_perplexities = process_batch_gpu(
                logits, input_ids_gpu, teacher_indices_batch, first_device
            )
            perplexities.extend(batch_perplexities)

            # Cleanup GPU memory
            del logits, input_ids_gpu, attention_mask_gpu, tokens_batch
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()

    return perplexities


def compute_perplexities_data_parallel(model_name, df, num_gpus, max_batch_tokens):
    """Compute perplexities using data parallelism across multiple GPUs."""
    # Split dataframe into chunks for each GPU
    chunk_size = len(df) // num_gpus
    df_chunks = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_gpus - 1 else len(df)
        df_chunks.append(df.iloc[start_idx:end_idx].reset_index(drop=True))

    # Process each chunk on a separate GPU using multiprocessing
    with mp.Pool(processes=num_gpus) as pool:
        results = pool.starmap(
            process_data_on_gpu,
            [(i, model_name, df_chunks[i], max_batch_tokens) for i in range(num_gpus)]
        )

    # Concatenate results from all GPUs
    all_perplexities = []
    for result in results:
        all_perplexities.extend(result)

    return all_perplexities


def process_batch_gpu(logits, input_ids, teacher_indices_batch, device):
    """Process batch results on GPU (computing perplexity from logits)."""
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]

    # Create teacher mask on GPU
    full_teacher_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    for batch_idx, teacher_indices in enumerate(teacher_indices_batch):
        valid_indices = [idx for idx in teacher_indices if idx < seq_len and idx > 0]
        full_teacher_mask[batch_idx, valid_indices] = True

    # Shift for causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = full_teacher_mask[..., 1:].contiguous()

    # Compute losses on GPU
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)

    token_losses = loss_fct(flat_logits, flat_labels)
    token_losses = token_losses.view(batch_size, -1)

    # Compute perplexities on GPU
    masked_losses = token_losses * shift_mask.float()
    sum_losses = masked_losses.sum(dim=1)
    num_teacher_tokens = shift_mask.sum(dim=1).float().clamp(min=1.0)
    mean_losses = sum_losses / num_teacher_tokens
    perplexities = torch.exp(mean_losses)

    # Only move final results to CPU
    batch_perplexities = perplexities.cpu().tolist()

    # Cleanup intermediate GPU tensors
    del full_teacher_mask, shift_logits, shift_labels, shift_mask
    del flat_logits, flat_labels, token_losses, masked_losses
    del sum_losses, num_teacher_tokens, mean_losses, perplexities

    return batch_perplexities

def main():
    """
    python analysis/compute_perplexity.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --nickname R1-Distill-Qwen-1.5B --max_batch_tokens 32768
    python analysis/compute_perplexity.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --nickname R1-Distill-Qwen-1.5B --max_batch_tokens 32768 --data_parallel
    """
    parser = argparse.ArgumentParser(description='Compute perplexity of teacher reasoning')
    parser.add_argument('--model_name', type=str, required=True, help='HuggingFace model name')
    parser.add_argument('--nickname', type=str, required=True, help='Nickname for input/output files')
    parser.add_argument('--input_dir', type=str, default='./results/allmath/teacher_guide')
    parser.add_argument('--output_dir', type=str, default='./results/perplexity')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_rows', type=int, default=None)
    parser.add_argument('--max_batch_tokens', type=int, default=32768,
                        help='Dynamic batching: max batch_size × max_seq_len')
    parser.add_argument('--data_parallel', action='store_true',
                        help='Use data parallelism across multiple GPUs (loads separate model on each GPU)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing results')
    args = parser.parse_args()

    if args.overwrite and os.path.exists(os.path.join(args.output_dir, f'{args.nickname}_perplexity.pickle')):
        print(f"Results already exist for {args.nickname}")
        exit()

    # Check GPU availability
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_data_parallel = args.data_parallel and num_gpus > 1

    if use_data_parallel:
        print(f"Data parallelism enabled: will use {num_gpus} GPUs")
        # Only load tokenizer in main process for preprocessing
        print(f"Loading tokenizer: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = None  # Model will be loaded separately on each GPU
    else:
        print(f"Loading tokenizer and model: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto' if args.device == 'cuda' else None
        )
        model.eval()

        if args.device == 'cuda' and torch.cuda.is_available():
            print(f"Model loaded across {torch.cuda.device_count()} GPU(s)")

    df_path = os.path.join(args.input_dir, f'{args.nickname}.pickle')
    print(f"Loading dataframe from {df_path}...")
    df = pd.read_pickle(df_path).reset_index(drop=True)

    if args.max_rows:
        df = df.head(args.max_rows)
        print(f"Processing first {args.max_rows} rows")

    print(f"Computing perplexities for {len(df)} rows with max_batch_tokens={args.max_batch_tokens}...")
    perplexities = compute_perplexities_for_dataframe(
        model, tokenizer, df,
        device=args.device,
        max_batch_tokens=args.max_batch_tokens,
        use_data_parallel=use_data_parallel,
        model_name=args.model_name
    )

    df['teacher_reasoning_perplexity'] = perplexities

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'{args.nickname}_perplexity.pickle')
    print(f"Saving results to {output_path}...")
    df.to_pickle(output_path)

    print(f"\nPerplexity statistics:")
    print(f"  Mean:   {np.mean(perplexities):.4f}")
    print(f"  Median: {np.median(perplexities):.4f}")
    print(f"  Std:    {np.std(perplexities):.4f}")
    print(f"  Min:    {np.min(perplexities):.4f}")
    print(f"  Max:    {np.max(perplexities):.4f}")
    print("\nDone!")


if __name__ == '__main__':
    # Set multiprocessing start method for CUDA compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    main()
