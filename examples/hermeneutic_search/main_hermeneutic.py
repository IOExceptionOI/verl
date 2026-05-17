"""
Entry point for Hermeneutic Search GRPO training.

Just imports the custom agent loop module to trigger registration, then runs
the standard verl main entrypoint. All custom logic is wired via config:

    actor_rollout_ref:
      rollout:
        agent:
          default_agent_loop: hermeneutic_agent
          agent_loop_manager_class: examples.hermeneutic_search.tools.hermeneutic_agent_worker.HermeneuticAgentLoopManager
          agent_loop_config_path: examples/hermeneutic_search/config/agent_loop_config.yaml

No monkey-patching. No custom TaskRunner needed.
"""

import os
import sys

# Make examples package importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Trigger @register("hermeneutic_agent") in main process
import examples.hermeneutic_search.tools.hermeneutic_agent_loop  # noqa: F401

# Run standard verl main (handles hydra, ray init, TaskRunner, etc.)
from verl.trainer.main_ppo import main

if __name__ == "__main__":
    main()
