import sys, os
sys.path.insert(0, os.getcwd())
from exp.interface.mem_reread import MemReread
from exp.interface.vllm_deploy import VllmWorker
import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from openai import OpenAI
from transformers import AutoTokenizer
import tiktoken
import torch.multiprocessing as mp

from exp.tools import *
from exp.tools.llminfer import *

ARGS = None

def get_prompt(prompt, model, tokenizer, args):
    input_ids = tokenizer.encode(prompt)
    return prompt, input_ids


def process_data(item):
    context = item['context']
    item['golden'] = item['golden'][0]
    
    item['task'] = item['type']['yaml_entry_name']
    item['task_type'] = 'global-reasoning'
    item['question'] = item['question'] + "Please use Arabic numerals for your answer."

    return item


def chunk_worker(item, model, tokenizer):
    item = process_data(item)

    item['context'], context_ids = get_prompt(item['context'], model, tokenizer, None)
    tokenized_chunks = [context_ids[i: i + ARGS.chunk] for i in range(0, len(context_ids), ARGS.chunk)]
    item['context_chunks'] = [tokenizer.decode(ck) for ck in tokenized_chunks]
    
    return item


@async_tqdm
async def get_pred_one(item, agent: "Agent", args, fout, subpbar):
    context = item['context']
    prompt = item['question']
    try:
        subpbar.update(1)
        output, clues = await agent(question = prompt, context = context, context_chunks = item['context_chunks'])
    except Exception as e:
        print("-" * 50)
        print("=" * 50)
        raise e
    if not output: output = ''
    response = output.strip()
    item['pred'] = response.split("</think>")[-1].split("\\boxed")[-1]

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

FILE_DICT = dict(
    gr = 'global_reasoning'

)

async def main(args, num_proc, rank, ctx_length, file_name):
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)


    file_name = FILE_DICT[file_name]
    name_template = 'global_reasoning_{ctx}.json'
    dpath = path_join(args.data_root, file_name, name_template.format(ctx = ctx_length))
    print(dpath)
    datas = read_json(dpath)


    out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + ('_cot' if args.cot else '') + ('_dis' if args.disable_thinking else '') + f"_{args.chunk}" + f"_{args.save_suf}", file_name, f"{ctx_length}.jsonl")

    fout = out_file


    if IO.exists(out_file):
        has_data_ids = set(k['id'] for k in read_jsonl(out_file))
    else: has_data_ids  = []

    local_ids = []
    for i in range(len(datas)):
        ids = datas[i]['id']
        if not isinstance(ids, str):
            ids = ids[0]
        if ids not in has_data_ids:
            local_ids+= [i]
    local_ids = list(chunks(local_ids, num_proc))[rank]
    
    data = [datas[i] for i in local_ids]

    tokenizer_model = args.model

    data = mapping(data, chunk_worker, tool = lambda : (args.model, AutoTokenizer.from_pretrained(args.model)))

    agent = MemReread(
        tokenizer_model,
        args.model, 
        args.port, 
        enable_thinking = args.cot, 
        max_chunk = args.chunk, 
        max_depth = args.depth, 
        max_width = args.width, 
        max_nodes = args.nodes
    )

    sub_datas = [data]
    with tqdm(total = len(data)) as pbar:
        with tqdm(total = len(data), desc = "process data:") as subpbar:
            for subdata in sub_datas:
                tasks = [async_create_task(get_pred_one(item, agent, args, fout, subpbar, pbar = pbar)) for item in subdata]
        
                await async_gather(*tasks)

    await close_redis_semaphore()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="./exp/prelinminary/results/pre")
    parser.add_argument("--model", "-m", type=str, default="")
    parser.add_argument("--port", "-p", type=int, default=7103)
    parser.add_argument("--chunk", "-c", type=int, default=5000)
    parser.add_argument("--depth", "-d", type=int, default=1)    
    parser.add_argument("--width", "-w", type=int, default=4)   
    parser.add_argument("--nodes", "-nodes", type=int, default=4)
    parser.add_argument("--data-root", type = str, default = "./datas")
    parser.add_argument("--ctxs", nargs = '+', type = str, default = ['1k','2k','4k','8k','16k','32k','64k','128k','256k','512k','1M'])
    parser.add_argument("--tasks", nargs = '+', type = str, default = ['gr'])
    parser.add_argument("--disable-thinking", action = "store_true")
    # parser.add_argument("--root-width", "-rw", type = int, default = 5)
    parser.add_argument("--no-chunk-prompt", action = "store_true")
    parser.add_argument("--max-model-len", type = int, default = 40960)
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    parser.add_argument("--n_proc", "-n", type=int, default=16)
    parser.add_argument("--save-suf", type = str, default = "mem_reread")
    parser.add_argument("--gpus", type = str, default = "0,1,2,3,4,5,6,7")
    args = parser.parse_args()

    ARGS = args
    num_proc = 16

    group = {
        '1k': 0,
        '2k':0,
        '4k':0,
        '8k':0,
        '16k':0,
        '32k':0,
        '64k':0,
        '128k':0,
        '256k':1,
        '512k':1,

    }

    ctxs_list =[[], [], [], []]

    for ctx_length in args.ctxs:
        ctxs_list[group[ctx_length]] += [ctx_length]

    with VllmWorker(args.model, args.port, max_model_len = args.max_model_len, gpus = args.gpus, timeout = 300):

        for ctxs in ctxs_list:
            if not ctxs: continue
            tasks = []
            for ctx_length in ctxs:
                for file_name in args.tasks:
                    for i in range(0, num_proc):
                        tasks += [(args, num_proc, i, ctx_length, file_name)]
            mapping(tasks, worker = lambda x: async_run(main(*x)), num_processes = num_proc)