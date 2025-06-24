"""
openaiapi.py

A production-ready client wrapper for calling OpenAI and third-party LLM APIs.
Provides functions for single and parallel chat completions with support for configurable models,
rate limiting, and efficient client reuse.

Author: Aochong Oliver Li
Date: 2025-04-22
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


'''Default client is OpenAI'''
def create_openai_client(model: str) -> OpenAI:
    """
    Create and return an OpenAI client for the specified model.

    Args:
        model (str): The model identifier (e.g., 'gpt-4o', 'deepseek', 'togetherai').

    Returns:
        OpenAI: Configured OpenAI client instance.
    """
    if 'gpt' in model.lower() or 'o1' in model.lower():
        ORG_ID = os.environ['OPENAI_ORG_ID']
        PROJECT_ID = os.environ['OPENAI_PROJECT_ID']
        OPENAI_API_KEY =  os.environ['OPENAI_API_KEY']
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            organization=ORG_ID,
            project=PROJECT_ID
            )
    elif 'deepseek' in model.lower():
        DEEPSEEK_API_KEY = os.environ['DEEPSEEK_API_KEY']
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )
    else:
        TOGETHERAI_API_KEY = os.environ['TOGETHERAI_API_KEY']

        client = OpenAI(
            api_key=TOGETHERAI_API_KEY,
            base_url = 'https://api.together.xyz/v1'
        )

    return client

# default client is gpt
client = create_openai_client('gpt')

'''Model Generate Response'''
def generate_chat_completion(
    input_prompt: str,
    developer_message: str = 'You are a helpful assistant',
    model: str = 'gpt-4o',
    temperature: float = 0.0,
    max_tokens: int = 1024,
    n: int = 1,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stop: Optional[list[str]] = None
) -> Union[str, List[str]]:
    """
    Generate a single or multiple chat completion(s) using the specified model.

    Args:
        input_prompt (str): The user prompt.
        developer_message (str): System or developer message for the model.
        model (str): Model identifier.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum tokens in the response.
        n (int): Number of completions to generate.
        top_p (float): Nucleus sampling parameter.
        frequency_penalty (float): Frequency penalty.
        presence_penalty (float): Presence penalty.
        stop (list[str], optional): Sequence(s) where the API will stop generating further tokens.

    Returns:
        str or List[str]: Generated completion text(s).
    """
    client = create_openai_client(model)
    
    # o-series models
    if 'o3' in model or 'o1' in model:
        raise NotImplementedError
    # gpt models or open source models on TogetherAI
    elif model == 'deepseek-reasoner':
        messages = [
            {"role": "system", "content": developer_message}, 
            {"role": "user", "content": input_prompt}
        ]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens
        )
        
        return [
            {
                'think': choice.message.reasoning_content,
                'content': choice.message.content
            } 
                for choice in response.choices
            ]
    else:
        messages = [
            {"role": "developer", "content": developer_message}, 
            {"role": "user", "content": input_prompt}
        ]
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n = n,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop
            )
        except Exception as e:
            print(e)
            return None
        
        return [choice.message.content for choice in response.choices]

def process_chunk_wrapper(args: Tuple[List[Dict[str, Any]], int]) -> List[Tuple[int, Optional[str]]]:
    """
    Process a chunk of API requests in parallel.

    Args:
        args: A tuple containing the chunk of request objects and the chunk ID.

    Returns:
        List of tuples mapping request index to response content.
    """
    chunk, chunk_id = args
    model = chunk[0]['body']['model']
    results: List[Tuple[int, Optional[str]]] = []
    for input_object in tqdm(chunk, desc=f"Process-{chunk_id}", position=chunk_id):
        # Extract parameters from the input object
        _id = int(input_object['custom_id'].replace('idx_', ''))
        input_prompt = input_object['body']['messages'][1]['content']
        developer_message = input_object['body']['messages'][0]['content']
        temperature = input_object['body']['temperature']
        max_tokens = input_object['body']['max_tokens']
        n = input_object['body']['n']
        top_p = input_object['body']['top_p']
        frequency_penalty = input_object['body']['frequency_penalty']
        presence_penalty = input_object['body']['presence_penalty']
        stop = input_object['body']['stop']

        # Call your generate_chat_completion function
        try:
            response = generate_chat_completion(
                input_prompt=input_prompt, 
                developer_message=developer_message, 
                model=model,
                temperature=temperature, 
                max_tokens=max_tokens,
                n=n, 
                top_p=top_p,
                frequency_penalty=frequency_penalty, 
                presence_penalty=presence_penalty, 
                stop=stop
            )
        except:
            response = None
        results.append((_id, response))
        time.sleep(1)
    return results

def generate_parallel_completions(
    input_filepath: str,
    cache_filepath: str,
    num_processes: int = 20
) -> None:
    """
    Execute chat completions in parallel using multiple processes and cache results.

    Args:
        input_filepath (str): Path to the newline-delimited JSON input file.
        cache_filepath (str): Path to save the pickled DataFrame of responses.
        num_processes (int): Number of worker processes.
    """    
    with open(input_filepath, 'r') as f:
        batch_input = [json.loads(line) for line in f]
    # Resume from existing cache if available
    if os.path.exists(cache_filepath):
        df_cached = pd.read_pickle(cache_filepath)
        # Load existing results as list of tuples (index, response)
        results = [(row['idx'], row['response']) for _, row in df_cached.iterrows() if row['response'] != None]
        success_idx = [item[0] for item in results]
        batch_input = [input_obj for input_obj in batch_input
                       if int(input_obj['custom_id'].split("_")[1]) not in success_idx
                       ]
    else:
        results = []
    
    # Split the data into chunks for parallel processing
    chunk_size = max(1, len(batch_input) // num_processes)
    chunks = [batch_input[i:i + chunk_size] for i in range(0, len(batch_input), chunk_size)]
        # Set up multiprocessing
    args = [(chunk, i) for i, chunk in enumerate(chunks)]

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_chunk_wrapper, arg) for arg in args]
        for future in as_completed(futures):
            results.extend(future.result())
            # Save intermediate checkpoint after each chunk completes
            sorted_results = sorted(results, key=lambda x: x[0])
            pd.DataFrame(
                {
                    "idx": [idx for idx, _ in sorted_results],
                    'response': [resp for _, resp in sorted_results]
                 }
                 
                 ).to_pickle(cache_filepath)

def minibatch_stream_generate_response(input_filepath: str,
                                       batch_log_filepath: str = None,
                                       minibatch_filepath: str = '/home/al2644/research/openai_batch_io/minibatchinput.jsonl',
                                       batch_size: int = 10,
                                       completion_window: str = '24h',
                                       failed_batch_start: int = None,
                                       failed_batch_end: int = None,
                                       batch_rate_limit: int = None):
    batch_logs = {}
    with open(input_filepath, 'r') as f:
        batch_input = [json.loads(line) for line in f]
        client = create_openai_client(batch_input[0]['body']['model'])

        if failed_batch_start is not None and failed_batch_end is not None:
            batch_input = batch_input[failed_batch_start: failed_batch_end]

    while len(batch_logs) * batch_size < len(batch_input):
        batch_idx = batch_size * len(batch_logs)

        with open(minibatch_filepath, 'w') as f:
            for item in batch_input[batch_idx : batch_idx + batch_size]:
                f.write(json.dumps(item) + '\n')
        
        # uplaod batch input files
        batch_input_file = client.files.create(
            file=open(minibatch_filepath, "rb"),
            purpose="batch"
        )
        
        # create batch
        batch_input_file_id = batch_input_file.id

        batch_log = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata={
            "description": f"minibatch_{batch_idx}"
            }
        )
        print(f'batch {batch_log.id} is created')

        batch_logs[batch_idx] = batch_log.id

        if batch_rate_limit is not None and len(batch_logs) % batch_rate_limit == 0:
            time.sleep(30)

        with open(batch_log_filepath, 'w') as f:
            json.dump(batch_logs, f)

def minibatch_retrieve_response(output_dict: dict = None):
    
    model_outputs = {}
    for _, output_file_id in output_dict.items():
        try:
            file_response = client.files.content(output_file_id)
            print(f'Retrieving output {output_file_id}')
            
            text_responses = file_response.text.split('\n')[:-1]
            json_responses = [json.loads(x) for x in text_responses]
            
            for output in json_responses:
                custom_id = int(output['custom_id'].replace('idx_', ''))
                content = output['response']['body']['choices'][0]['message']['content']
                model_outputs[custom_id] = content
        except:
            continue
    
    return pd.DataFrame.from_dict(model_outputs, orient='index', columns = ['response'])        

def minibatch_stream_retry (batch_log_filepath: str, batch_rate_limit: int = None):
    failed_batch_logs = {}
    retry_batch_logs = {}

    with open(batch_log_filepath, 'r') as f:
        batch_logs = json.load(f)
    
    for batch_idx, batch_log_id in batch_logs.items():
        status = check_batch_status(batch_log_id)
        if status == 'failed':
            failed_batch_logs[batch_idx] = batch_log_id
    
    for batch_idx, batch_log_id in failed_batch_logs.items():
        print(f'Retrying batch {batch_idx}')
        
        batch_log = client.batches.retrieve(batch_log_id)
        batch_input_file_id = batch_log.input_file_id
        completion_window = batch_log.completion_window

        batch_log = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata={
            "description": f"minibatch_{batch_idx}"
            }
        )
        print(f'batch {batch_log.id} is created')

        retry_batch_logs[batch_idx] = batch_log.id

        if batch_rate_limit is not None and len(retry_batch_logs) % batch_rate_limit == 0:
            time.sleep(30)
        
        batch_logs.update(retry_batch_logs)

        with open(batch_log_filepath, 'w') as f:
            json.dump(batch_logs, f)

'''
Utils for OpenAI BatchAPI
'''

def batch_query_template(input_prompt: str, developer_message: str = 'You are a helpful assistant', model: str = 'gpt-4o', custom_id: str = None,
                         temperature: float = 0.0, max_tokens: int = 1024, n: int = 1, top_p: float = 1.0, frequency_penalty: float = 0.0,
                         presence_penalty: float = 0.0, stop: Optional[list[str]] = None):
    query_template = {"custom_id": custom_id,
                  "method": "POST",
                  "url": "/v1/chat/completions",
                   "body": {"model": model,
                            "temperature": temperature,
                            "messages": [{"role": "developer", "content": developer_message},
                                         {"role": "user", "content": input_prompt}
                                        ],
                            "max_tokens": max_tokens,
                            "n": n,
                            "top_p": top_p,
                            "frequency_penalty": frequency_penalty,
                            "presence_penalty": presence_penalty,
                            "stop": stop}
                 }
    
    return query_template

def retrieve_batch_output_file_id(batch_log_id: str, model = 'gpt'):
    client = create_openai_client(model)
    batch_log = client.batches.retrieve(batch_log_id)
    
    return batch_log.output_file_id

def check_batch_status(batch_log_id: str, model = 'gpt'):
    client = create_openai_client(model)
    batch_log = client.batches.retrieve(batch_log_id)
    
    return batch_log.status

def check_batch_error(batch_log_id: str, model = 'gpt'):
    client = create_openai_client(model)
    batch_log = client.batches.retrieve(batch_log_id)

    if batch_log.status == 'failed':
        print(f'Batch {batch_log_id} failed with error: {batch_log.errors}')
        return batch_log.errors
    else:
        return None
    
def cancel_batch(batch_log_id: str, model = 'gpt'):
    client = create_openai_client(model)
    client.batches.cancel(batch_log_id)
    
    return f'Batch {batch_log_id} is cancelled'

def cache_batch_query(filepath: str, query: dict):
    with open(filepath, 'a') as f:
        f.write(json.dumps(query) + '\n')
