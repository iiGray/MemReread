#!/bin/bash
set -x

NNODES=1
NGPUS_PER_NODE=8
PROJ_ROOT=./
DATASET_ROOT="./datas/training_data"

MODEL_PATH=Qwen/Qwen3-4B
VAL_PATH="${DATASET_ROOT}/hotpotqa_dev.parquet"
TRAIN_PATH="${DATASET_ROOT}/hotpotqa_train.parquet"



STRICT_MATH=True
FACTOR=0.1

MAX_DEPTH=1
MAX_WIDTH=3
MAX_NODES=3
MAX_CONVS=-1
ROLLOUT_N=4


EXP="memory_agent/MemReread-4B"

PROJ_DIR=${PROJ_ROOT}/${EXP}


MAXLEN=10240
MAX_NEW_TOKEN=1024

export VLLM_LOGGING_LEVEL=DEBUG

export RAY_BACKEND_LOG_LEVEL=debug
export VLLM_RPC_TIMEOUT=360000
export VLLM_USE_V1=1
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.chat_scheduler=recurrent.async_utils.ChatCompletionProxy \
    recurrent.memory.config.chunk_size=5000 \
    reward_model.reward_kwargs.org_fn=${STRICT_MATH} \
    custom_reward.n_factor=${FACTOR} \
    custom_reward.theta=0.9 \
    custom_reward.alpha=0.05 \
    actor_rollout_ref.rollout.max_depth=${MAX_DEPTH} \
    actor_rollout_ref.rollout.max_width=${MAX_WIDTH} \
    actor_rollout_ref.rollout.max_nodes=${MAX_NODES} \
    actor_rollout_ref.rollout.max_convs=${MAX_CONVS} \
    actor_rollout_ref.rollout.update_thinking=False \
    algorithm.use_ReA=True \
    algorithm.adv_estimator=grpo \
    algorithm.grpo_use_adv=False \
    trainer.save_freq=10 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    trainer.logger=['console','tensorboard'] \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.clip_ratio_high=0.20 \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    data.train_files=$TRAIN_PATH \
    data.val_files=$VAL_PATH \
    data.shuffle=False \
    data.filter_overlong_prompts=True \
    data.train_batch_size=64 \
    data.truncation='center' \
    +data.context_key='context' \
    data.max_prompt_length=$MAXLEN \
    data.max_response_length=$MAX_NEW_TOKEN \
    reward_model.reward_manager='thread' \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=40960 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=20480 \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.project_name='mem_reread' \
    trainer.experiment_name=${EXP} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.test_freq=5 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$PROJ_DIR \
    trainer.total_epochs=30