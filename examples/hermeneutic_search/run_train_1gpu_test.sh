#!/bin/bash
# Single GPU test for Hermeneutic Search (debug mode)
# Uses container's device 2 (host GPU 7)
set -x
ulimit -n 65535

export CUDA_VISIBLE_DEVICES=2
export NCCL_P2P_LEVEL=NVL
export CUDA_DEVICE_ORDER=PCI_BUS_ID

PROJECT_DIR=/workspace/verl
CONFIG_PATH=$PROJECT_DIR/examples/hermeneutic_search/config
TOOL_CONFIG=$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml
AGENT_LOOP_CONFIG=$PROJECT_DIR/examples/hermeneutic_search/config/agent_loop_config.yaml

TRAIN_DATA=$PROJECT_DIR/data/hermeneutic_search/train.parquet
VAL_DATA=$PROJECT_DIR/data/hermeneutic_search/test.parquet
MODEL=/workspace/models/Qwen2.5-3B

python3 -m examples.hermeneutic_search.main_hermeneutic \
    --config-path="$CONFIG_PATH" \
    --config-name='hermeneutic_search_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=1500 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_model_len=8000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
    actor_rollout_ref.rollout.agent.default_agent_loop=hermeneutic_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$AGENT_LOOP_CONFIG \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=examples.hermeneutic_search.tools.hermeneutic_agent_worker.HermeneuticAgentLoopManager \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.test_freq=50 \
    trainer.log_val_generations=5 \
    data.val_max_samples=32 \
    'trainer.logger=[console,wandb]' \
    trainer.project_name=HermeneuticSearch \
    trainer.experiment_name=hs-3b-base-v9 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    data.train_files=$TRAIN_DATA \
    data.val_files=$TRAIN_DATA \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG \
    reward.custom_reward_function.path=/workspace/verl/verl/utils/reward_score/hermeneutic_qa_em.py \
    reward.custom_reward_function.name=compute_score \
    reward.reward_manager.source=importlib \
    reward.reward_manager.name=HermeneuticRewardManager \
    reward.reward_manager.module.path=/workspace/verl/examples/hermeneutic_search/tools/hermeneutic_reward_manager.py \
    trainer.total_epochs=1 \
    "$@"
