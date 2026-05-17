"""
Main entry point for Hermeneutic Search GRPO training.

This file:
1. Registers the hermeneutic_agent loop (with true context reset)
2. Patches the rollout output to fix attention_mask at cycle boundaries
3. Runs standard verl PPO training

Usage:
    python -m examples.hermeneutic_search.main_hermeneutic --config-path=... --config-name=...
"""

import os
import sys

# Register the hermeneutic agent loop BEFORE verl imports
# This ensures the @register("hermeneutic_agent") decorator fires
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import examples.hermeneutic_search.tools.hermeneutic_agent_loop  # noqa: F401

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.utils.device import auto_set_device


class HermeneuticTaskRunner(TaskRunner):
    """
    Custom TaskRunner that applies the cycle attention mask patch
    after the trainer is initialized.
    """

    def run(self, config):
        """Override run to apply hermeneutic patches."""
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        
        # Monkey-patch RayPPOTrainer.fit to fix attention masks
        original_fit = RayPPOTrainer.fit
        
        def patched_fit(trainer_self):
            """Wrap the trainer's generate_sequences to fix cycle attention masks."""
            from examples.hermeneutic_search.tools.cycle_attention_patch import fix_cycle_attention_mask
            from transformers import AutoTokenizer
            
            # Get pad_token_id from the model's tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            
            # Patch the rollout manager
            original_generate = trainer_self.async_rollout_manager.generate_sequences
            
            def patched_generate(batch, *args, **kwargs):
                result = original_generate(batch, *args, **kwargs)
                return fix_cycle_attention_mask(result, pad_token_id)
            
            trainer_self.async_rollout_manager.generate_sequences = patched_generate
            print(f"[HermeneuticSearch] Patched generate_sequences (pad_token_id={pad_token_id})")
            
            # Run original fit
            return original_fit(trainer_self)
        
        RayPPOTrainer.fit = patched_fit
        
        # Now run the standard training setup
        return super().run(config)


@hydra.main(config_path="config", config_name="hermeneutic_search_grpo", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    
    # Use custom TaskRunner
    task_runner_class = ray.remote(num_cpus=1)(HermeneuticTaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
