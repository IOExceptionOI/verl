"""
Custom RewardManager that injects `global_steps` into each sample's extra_info.

This is verl's canonical pattern for passing per-step training state to the
reward function. Inspired by verl's own NaiveRewardManager — the only change
is a single extra `extra_info["global_steps"] = ...` line. Other managers
(visual, remote, gdpo) follow the same pattern for injecting different fields.

The trainer sets `batch.meta_info["global_steps"]` at ray_trainer.py:1367.
We read it here and forward to compute_score via extra_info.

Wire in config:
    reward.reward_manager.source=importlib
    reward.reward_manager.name=HermeneuticRewardManager
    reward.reward_manager.module.path=.../hermeneutic_reward_manager.py
"""

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager


class HermeneuticRewardManager(NaiveRewardManager):
    """NaiveRewardManager + inject global_steps into extra_info."""

    async def run_single(self, data: DataProto) -> dict:
        # Read global_steps from batch meta_info (set by trainer).
        global_steps = data.meta_info.get("global_steps", 0)

        # Inject into per-sample extra_info before parent runs compute_score.
        data_item = data[0]
        existing_extra = data_item.non_tensor_batch.get("extra_info", {})
        if existing_extra is None:
            existing_extra = {}
        if not isinstance(existing_extra, dict):
            existing_extra = dict(existing_extra)
        existing_extra["global_steps"] = global_steps
        # Mutate in place so NaiveRewardManager.run_single picks it up.
        data_item.non_tensor_batch["extra_info"] = existing_extra

        return await super().run_single(data)
