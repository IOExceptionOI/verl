#!/bin/bash
# Hermeneutic Search GRPO training with TRUE context reset
# Uses custom agent loop + attention mask patch for per-cycle isolation
set -x
ulimit -n 65535

export NCCL_P2P_LEVEL=NVL
export CUDA_DEVICE_ORDER=PCI_BUS_ID

PROJECT_DIR=/workspace/verl
CONFIG_PATH=$PROJECT_DIR/examples/hermeneutic_search/config
TOOL_CONFIG=$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml

TRAIN_DATA=$PROJECT_DIR/data/hermeneutic_search/train.parquet
VAL_DATA=$PROJECT_DIR/data/hermeneutic_search/test.parquet

MODEL=/workspace/models/Qwen2.5-3B-Instruct

# Use custom entry point that registers hermeneutic_agent + patches attention mask
python3 -m examples.hermeneutic_search.main_hermeneutic \
    --config-path="$CONFIG_PATH" \
    --config-name='hermeneutic_search_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=128 \
    data.val_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=3000 \
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
    actor_rollout_ref.rollout.max_model_len=15000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.agent.default_agent_loop=hermeneutic_agent \n    actor_rollout_ref.rollout.agent.agent_loop_config_path=/workspace/verl/examples/hermeneutic_search/config/agent_loop_config.yaml \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.test_freq=50 \
    'trainer.logger=[console,wandb]' \
    trainer.project_name=HermeneuticSearch \
    trainer.experiment_name=hs-qwen2.5-3b-instruct-grpo-ctx-reset \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=3 \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG \
    trainer.total_epochs=1 \
    "$@"
