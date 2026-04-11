#!/bin/bash
# ============================================================
# GSM8K + Qwen2.5-0.5B-Instruct + PPO 单卡训练
# ============================================================
# 用法:
#   bash my-scripts/run_gsm8k_qwen0.5b_ppo_1gpu.sh
#
# 切换 GPU:
#   CUDA_VISIBLE_DEVICES=5 bash my-scripts/run_gsm8k_qwen0.5b_ppo_1gpu.sh
#
# 追加/覆盖参数 (Hydra 风格, 通过 $@ 透传):
#   bash my-scripts/run_gsm8k_qwen0.5b_ppo_1gpu.sh trainer.total_epochs=5
# ============================================================

set -x

# 默认 GPU 编号 (可通过环境变量覆盖)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
export PYTHONUNBUFFERED=1

export RAY_TMPDIR=/dev/shm/ray_tmp

# 路径配置 (容器内绝对路径, 对应宿主机挂载卷)
DATA_DIR=/root/data/gsm8k
MODEL_PATH=/root/models/Qwen2.5-0.5B-Instruct

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.model.path=${MODEL_PATH} \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    critic.optim.lr=1e-5 \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_my_gsm8k' \
    trainer.experiment_name='qwen2.5_0.5b_ppo_1gpu' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 $@
