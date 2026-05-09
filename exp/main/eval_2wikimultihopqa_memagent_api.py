import sys, os
sys.path.insert(0, os.getcwd())
from exp.interface.memagent_api import MemAgent_API
import os, csv, json
import argparse
import time
import glob
from tqdm import tqdm
from datasets import load_dataset
import re
from openai import OpenAI
from transformers import AutoTokenizer
# pyrefly: ignore [missing-import]
import tiktoken
import torch.multiprocessing as mp

from exp.tools import *
from exp.tools.llminfer import *
from exp.tools.llminfer.semaphore import set_max_concurrent

set_max_concurrent(8)


ARGS = None


def get_prompt(prompt, model, tokenizer, args):
    input_ids = tokenizer.encode(prompt)
    return prompt, input_ids


def process_data(item):
    if "input" in item and "answers" in item:
        item['task_type'] = 'custom_qa'
        item['question'] = item['input'] + " Please answer the question directly and concisely."
        item['golden'] = item['answers'] 
        return item

    context = item['context']
    item['golden'] = item['golden'][0]
    
    return item


def chunk_worker(item, model, tokenizer):
    item = process_data(item)

    item['context'], context_ids = get_prompt(item['context'], model, tokenizer, None)
    tokenized_chunks = [context_ids[i: i + ARGS.chunk] for i in range(0, len(context_ids), ARGS.chunk)]
    item['context_chunks'] = [tokenizer.decode(ck) for ck in tokenized_chunks]
    
    return item


@async_tqdm
async def get_pred_one(item, agent: MemAgent_API, args, fout, subpbar):
    context = item['context']
    prompt = item['question']
    try:
        subpbar.update(1)
        output, clues = await agent(
            question=prompt,
            context=context,
            context_chunks=item['context_chunks'],
        )
    except Exception as e:
        print("-" * 50)
        print("=" * 50)
        raise e
    if not output: output = ''
    response = output.strip()
    item['pred'] = response.split("</think>")[-1].split("\\boxed")[-1]

    if item['task_type'] == 'custom_qa':
        pred_lower = item['pred'].lower()
        item['judge'] = any(str(ans).lower() in pred_lower for ans in item['golden'])
    else:
        item['judge'] = item['golden'] in item['pred']

    item['record'] = clues
    item.pop('response', None)
    item.pop('context', None)
    item.pop('context_ids', None)
    item.pop('context_chunks', None)
    item['response'] = response
    item['context'] = context[:500]
    try:
        if not item['pred']: return
        with Synchronizer(fout.strip(".jsonl")):
            push_jsonl(item, fout)
    except Exception as e:
        print(item)
        raise e


async def main(args, num_proc, rank, file_path, task_name, ctx_length, min_dist):
    print(f"Loading data from: {file_path}")
    datas = read_json(file_path)
    datas = [datas[_] for _ in range(len(datas))]

    out_filename = f"{ctx_length}.jsonl" if not min_dist else f"{ctx_length}_min_distance_{min_dist}.jsonl"

    out_dir = os.path.join(args.save_dir, args.model.split("/")[-1] + ('_cot' if args.cot else '') + ('_dis' if args.disable_thinking else '') + f"_{args.chunk}" + f"_{args.save_suf}", task_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, out_filename)

    fout = out_file

    if IO.exists(out_file):
        has_data_ids = set(k['id'] for k in read_jsonl(out_file))
    else:
        has_data_ids = []

    local_ids = []
    for i in range(len(datas)):
        if datas[i]["id"] not in has_data_ids:
            local_ids += [i]
    chunks_data = list(chunks(local_ids, num_proc))
    if rank >= len(chunks_data):
        local_ids = []
    else:
        local_ids = chunks_data[rank]
    
    data = [datas[i] for i in local_ids]

    tokenizer_model = args.tokenizer_model
    data = mapping(
        data,
        chunk_worker,
        tool=lambda: (args.model, AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)),
    )

    agent = MemAgent_API(
        tokenizer_model=tokenizer_model,
        model_name=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        enable_thinking=args.cot,
        chunk_size=args.chunk,
    )
    
    sub_datas = [data]
    with tqdm(total=len(data)) as pbar:
        with tqdm(total=len(data), desc=f"Testing {out_filename}:") as subpbar:
            for subdata in sub_datas:
                tasks = [async_create_task(get_pred_one(item, agent, args, fout, subpbar, pbar=pbar)) for item in subdata]
                await async_gather(*tasks)
    await close_redis_semaphore()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="./exp/main/results/custom_qa")
    parser.add_argument("--model", "-m", type=str, default="")
    parser.add_argument("--tokenizer-model", "-tm", type=str, default="")
    parser.add_argument("--base-url", "-bs", type=str, default="")
    parser.add_argument("--api-key", "-ak", type=str, default="")
    parser.add_argument("--chunk", "-c", type=int, default=5000)
    parser.add_argument("--data-root", type=str, default="./datas/2wikimultihopqa")
    parser.add_argument(
        "--ctxs",
        nargs="+",
        type=str,
        default=["8k", "16k", "32k", "64k", "128k", "256k", "512k", "1M"],
    )
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--no-chunk-prompt", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--cot", "-cot", action='store_true') 
    parser.add_argument("--no_context", "-nc", action='store_true') 
    parser.add_argument("--rag", "-rag", type=int, default=0) 
    parser.add_argument("--n_proc", "-n", type=int, default=4)
    parser.add_argument("--save-suf", type=str, default="memagent_api")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()

    ARGS = args
    num_proc = args.n_proc

    group_map = {
        '8k': 0, '16k': 0, '32k': 0, '64k': 0,
        '128k': 0, '256k': 1, '512k': 1, '1M': 1
    }

    tasks_by_group = {0: [], 1: [], 2: []}

    TARGET_TASK = "2wikimultihopqa"
    print(f"Scanning directory: {args.data_root} for task: {TARGET_TASK}")
    print("CTXS:", args.ctxs)
    
    all_json_files = glob.glob(os.path.join(args.data_root, f"eval_{TARGET_TASK}_*.json"))
    
    if not all_json_files:
        print(f"No evaluation files found for {TARGET_TASK} in the specified directory!")
        sys.exit(0)

    filename_pattern = re.compile(r"^eval_(.+?)_(\d+[kM])(?:_min_distance_(\d+[kM]))?\.json$")

    for file_path in all_json_files:
        filename = os.path.basename(file_path)
        match = filename_pattern.match(filename)
        
        if not match:
            print(f"Skipping file with unrecognized format: {filename}")
            continue
            
        task_name = match.group(1)   
        ctx_length = match.group(2)  
        min_dist = match.group(3)    
        print("TCM:", task_name, ctx_length, min_dist)
        
        if task_name != TARGET_TASK:
            continue
        if ctx_length not in set(args.ctxs):
            continue
            
        if ctx_length not in group_map:
            print(f"Warning: Unknown context length {ctx_length} in {filename}, skipping.")
            continue
            
        node_idx = group_map[ctx_length]
        
        for i in range(num_proc):
            tasks_by_group[node_idx].append((args, num_proc, i, file_path, task_name, ctx_length, min_dist))
            
    print(f"Successfully loaded {len(all_json_files)} test files for {TARGET_TASK}.")

    for node_id, tasks in tasks_by_group.items():
        if not tasks: 
            continue
        mapping(tasks, worker=lambda x: async_run(main(*x)), num_processes=num_proc)
