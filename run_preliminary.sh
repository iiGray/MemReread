#!/bin/bash

python ./exp/preliminary/eval_global_reasoning_mem_reread.py \
    --model /path/to/the/model \
    --ctxs '8k' \
    --n_proc 16
