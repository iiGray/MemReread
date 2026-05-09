#!/bin/bash

python ./exp/main/eval_2wikimultihopqa_memreread_api.py \
    --model xxx \
    --tokenizer-model Qwen/Qwen3-0.6B \
    --base-url https://xxx \
    --api-key the-api-key \
    --ctxs '8k','16k','32k','64k','128k' \
    --n_proc 32

