#!/bin/bash

python ./exp/main/eval_2wikimultihopqa_memreread.py \
    --model /path/to/the/model \
    --ctxs '8k','16k','32k','64k','128k' \
    --n_proc 16
