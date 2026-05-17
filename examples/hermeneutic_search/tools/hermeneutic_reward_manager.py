"""
Custom RewardManager that injects `global_steps` into each sample's extra_info.

Why this exists:
verl's default reward managers don't pass `global_steps` to the reward function
— only data-level fields. But we want step-aware reward shaping (format_score
that decays over training). This manager reads `data.meta_info["global_steps"]`
(set by trainer) and injects it into extra_info before calling compute_score.

Built on verl 0.8's experimental RewardManagerBase interface (run_single per sample).

Wire via:
    reward.reward_manager.source=importlib
    reward.reward_manager.name=HermeneuticRewardManager
    reward.reward_manager.module.path=.../hermeneutic_reward_manager.py
"""

import inspect

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score


class HermeneuticRewardManager(RewardManagerBase):
    """Inject global_steps into extra_info, then call compute_score."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        # Decode response
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = data_item.batch["responses"][:valid_response_length]
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

        # Build extra_info with step injected
        extra_info = data_item.non_tensor_batch.get("extra_info", None)
        extra_info = dict(extra_info) if extra_info else {}

        # Read global_steps from batch meta_info (set by trainer at ray_trainer.py:1367)
        global_steps = data.meta_info.get("global_steps", 0)
        extra_info["global_steps"] = global_steps

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        # Call compute_score
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                ),
            )

        if isinstance(result, dict):
            reward = float(result.get("score", 0.0))
            reward_extra_info = {k: v for k, v in result.items() if k != "score"}
        else:
            reward = float(result)
            reward_extra_info = {}

        return {
            "reward": reward,
            "valid_response_length": int(valid_response_length),
            "reward_extra_info": reward_extra_info,
        }
