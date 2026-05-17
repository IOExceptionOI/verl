#!/bin/bash
# Hermeneutic Search: RL Training with GRPO (3 GPUs: 5,6,7)
# Uses verl's sglang multi-turn rollout with hermeneutic search tool
# Usage: bash run_train_3gpu.sh

set -x
ulimit -n 65535

export CUDA_VISIBLE_DEVICES=5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_P2P_LEVEL=NVL

PROJECT_DIR=/workspace/verl
CONFIG_PATH=$PROJECT_DIR/examples/hermeneutic_search/config
TOOL_CONFIG=$CONFIG_PATH/tool_config/hermeneutic_tool_config.yaml

# Use hermeneutic-formatted data (run prepare_data.py first)
TRAIN_DATA=$PROJECT_DIR/data/hermeneutic_search/train.parquet
VAL_DATA=$PROJECT_DIR/data/hermeneutic_search/test.parquet

MODEL=/workspace/models/Qwen2.5-3B

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='hermeneutic_search_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=128 \
    data.val_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    'trainer.logger=[console,wandb]' \
    trainer.project_name=HermeneuticSearch \
    trainer.experiment_name=hs-qwen2.5-3b-base-grpo-3gpu \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.max_actor_ckpt_to_keep=3 \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG \
    trainer.total_epochs=1 \
    "$@"
