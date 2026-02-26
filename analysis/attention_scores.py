import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import pickle
from functools import partial
import gc

# --- CONFIG ---
CHUNK_SIZE = 512 
PROMPT_COL = 'prompt'
RESPONSE_COL = 'post_distraction_response'

def custom_select_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['original_ratio'] == 0.0].reset_index(drop=True)

def find_token_boundaries(prompt, response, tokenizer):
    if '</think>' in response: response = response.split('</think>')[0]
    full_text = prompt + response
    full_tokens = tokenizer(full_text, return_tensors='pt', add_special_tokens=False)
    
    template_result = tokenizer.apply_chat_template([{"role": "user", "content": "HANDLE"}], tokenize=False, add_generation_prompt=True)
    try:
        _, suffix = template_result.split("HANDLE")
        input_prompt_text, _ = prompt.split(suffix, 1)
    except:
        suffix = "\n\n"
        input_prompt_text = prompt

    len_problem = len(tokenizer(input_prompt_text + suffix, add_special_tokens=False)['input_ids'])
    len_prompt = len(tokenizer(prompt, add_special_tokens=False)['input_ids'])
    len_total = full_tokens['input_ids'].shape[1]

    return {
        'slices': {
            'problem': (0, len_problem),
            'reasoning': (len_problem, len_prompt),
            'response': (len_prompt, len_total)
        },
        'full_tokens': full_tokens
    }

def attention_agg(final_tensor):
    """
    Input: (Seq, Layers, Heads, 3)
    Output: Dict with flattened averages and metadata
    """
    seq_len, num_layers, num_heads, _ = final_tensor.shape

    flat_tensor = final_tensor.view(seq_len, num_layers * num_heads, 3)
    
    res = {
        'n_layers': num_layers,
        'n_heads': num_heads,
        'overall_avg_scores': flat_tensor.mean(dim=0),
        'first_128_avg_scores': flat_tensor[:128].mean(dim=0),
        'first_256_avg_scores': flat_tensor[:256].mean(dim=0),
        'first_512_avg_scores': flat_tensor[:512].mean(dim=0),
        'first_1024_avg_scores': flat_tensor[:1024].mean(dim=0)
    }
    return res

class AttentionBlockProcessor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        self.stats_buffer = {} 
        self.hooks = []
        self.current_slices = None
        self.current_global_offset = 0 

    def _hook_fn(self, layer_idx, module, args, output):
        attn_matrix = output[1]
        
        if attn_matrix is None: return output

        s = self.current_slices
        p_s, p_e = s['problem']
        x_s, x_e = s['reasoning']
        r_s, r_e = s['response']
        
        device = attn_matrix.device
        dtype = attn_matrix.dtype
        chunk_rows = attn_matrix.shape[-2]
        
        row_idx = torch.arange(
            self.current_global_offset, 
            self.current_global_offset + chunk_rows, 
            device=device
        ).view(1, 1, -1, 1)
        
        col_idx = torch.arange(
            r_s, 
            r_s + attn_matrix[..., :, r_s:].shape[-1], 
            device=device
        ).view(1, 1, 1, -1)

        block_result = torch.stack(
            [
                attn_matrix[..., :, p_s:p_e].sum(dim=-1),
                attn_matrix[..., :, x_s:x_e].sum(dim=-1),
                torch.einsum('...ij, ...ij -> ...i', 
                    attn_matrix[..., :, r_s:], 
                    (col_idx <= row_idx).to(dtype)
                )
            ],
             dim=-1).cpu()

        if layer_idx not in self.stats_buffer:
            self.stats_buffer[layer_idx] = []
        self.stats_buffer[layer_idx].append(block_result)

        attn_matrix.set_(torch.tensor([], dtype=dtype, device=device))
        
        return (output[0], None) + output[2:]

    def run(self, input_ids, slices):
        self.stats_buffer = {}
        self.current_slices = slices
        r_start = slices['response'][0]
        
        prompt_ids = input_ids[:, :r_start]
        response_ids = input_ids[:, r_start:]
        num_resp_tokens = response_ids.shape[1]

        with torch.inference_mode():
            # Phase 1: Prefill
            outputs = self.model(prompt_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            del outputs
            torch.cuda.empty_cache()

            # Phase 2: Block Decoding
            for i, layer in enumerate(self.model.model.layers):
                self.hooks.append(layer.self_attn.register_forward_hook(partial(self._hook_fn, i)))

            for i in range(0, num_resp_tokens, CHUNK_SIZE):
                self.current_global_offset = r_start + i
                chunk_ids = response_ids[:, i : i + CHUNK_SIZE]
                
                outputs = self.model(
                    chunk_ids, 
                    past_key_values=past_key_values, 
                    output_attentions=True, 
                    use_cache=True
                )
                past_key_values = outputs.past_key_values
                del outputs
            
            for h in self.hooks: h.remove()
            self.hooks = []

        if not self.stats_buffer: return None

        # Phase 3: Aggregate
        layers_data = []
        for i in range(len(self.stats_buffer)):
            layers_data.append(torch.cat(self.stats_buffer[i], dim=2))
            
        # Stack Layers -> (Layers, Batch=1, Heads, Seq, 3)
        final_tensor = torch.stack(layers_data, dim=0).squeeze(1)
        
        # Permute to: (Seq, Layers, Heads, 3)
        final_tensor = final_tensor.permute(2, 0, 1, 3)

        # Normalize
        final_tensor = final_tensor / (final_tensor.sum(dim=-1, keepdim=True) + 1e-9)
        return final_tensor.float()

def analyze_dataframe(model, tokenizer, df, device='cuda'):
    results = []
    model.eval()
    
    processor = AttentionBlockProcessor(model, tokenizer)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            data = find_token_boundaries(row[PROMPT_COL], row[RESPONSE_COL], tokenizer)
            input_ids = data['full_tokens']['input_ids'].to(device)
            
            if (data['slices']['response'][1] - data['slices']['response'][0]) <= 0: continue

            final_tensor = processor.run(input_ids, data['slices'])
            agg_data = attention_agg(final_tensor)
            
            res = {
                'row_idx': idx,
                'problem': row['problem'],
                'source': row['source'],
                'solve_n': row.get('solve_n', None),
                'distractor_solve_n': row.get('distractor_solve_n', None),
                'model_is_correct': row.get('model_is_correct', None),
                'token_counts': {k: v[1]-v[0] for k,v in data['slices'].items()}
            }
            res.update(agg_data)
            results.append(res)

            if idx % 20 == 0: 
                gc.collect()
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error row {idx}: {e}")
            torch.cuda.empty_cache()
            
    return results

def main():
    """
    python analysis/attention_scores.py \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --nickname R1-Distill-Qwen-7B \
        --input_dir ./results/allmath/inject_distractor \
        --output_dir ./results/inject_distractor/attention_scores
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--nickname', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_rows', type=int, default=None)
    parser.add_argument('--overwrite', type=bool, default=False)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if not args.overwrite and os.path.exists(os.path.join(output_dir, f'{args.nickname}.pickle')):
        print(f"Results already exist for {args.nickname}")
        exit()

    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation="eager" 
    )

    df = pd.read_pickle(os.path.join(args.input_dir, f'{args.nickname}.pickle'))
    df = custom_select_rows(df)
    if args.max_rows: df = df.head(args.max_rows)

    results = analyze_dataframe(model, tokenizer, df, device=args.device)

    with open(os.path.join(output_dir, f'{args.nickname}.pickle'), 'wb') as f:
        pickle.dump(results, f)
    print("Done.")

if __name__ == '__main__':
    main()