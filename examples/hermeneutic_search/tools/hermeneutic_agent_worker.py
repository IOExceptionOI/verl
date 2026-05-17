"""
Custom AgentLoopWorker for Hermeneutic Search.

Uses verl's official extension point `agent_loop_manager_class` (configured in
yaml as `actor_rollout_ref.rollout.agent.agent_loop_manager_class`).

Overrides `_agent_loop_postprocess` to recompute `position_ids` so each cycle's
positions start from 0. This triggers transformers' `_is_packed_sequence`
detection, which makes flash_attention apply block-diagonal attention isolation
between cycles (training/inference consistency).

No monkey-patching. No PAD separators. Explicit cycle boundary metadata flows
from agent loop → worker via `extra_fields["cycle_boundary_positions"]`.

Reference: examples/hermeneutic_search/varlen_cu_seqlens_explained.md
"""

import ray
import torch

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopWorker,
)


def compute_position_ids_for_cycles(
    attention_mask: torch.Tensor,
    cycle_boundaries: list[int],
    prompt_length: int,
) -> torch.Tensor:
    """
    Compute position_ids with per-cycle reset.

    For each sample:
    - Left-padded prompt positions: zeros
    - Cycle 0 (prompt + cycle_0_response): standard increasing positions starting from 0
    - Cycle i (i>=1): positions reset to 0 at cycle boundary

    Args:
        attention_mask: [bsz, seq_len], 1 for content, 0 for left/right padding
        cycle_boundaries: list of int, position in response_ids where each cycle starts.
                          E.g., [0, 250] means cycle 0 starts at response[0], cycle 1 at response[250].
        prompt_length: length of the prompt portion in the input_ids (left-padded part length).

    Returns:
        position_ids: [bsz, seq_len], int64
    """
    bsz, seq_len = attention_mask.shape
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)

    # Per-sample computation (cycle_boundaries can differ per sample if we ever batch
    # different trajectories; for now assumed same-shape).
    for b in range(bsz):
        mask = attention_mask[b]
        # Effective sequence: indices where mask == 1
        valid_indices = mask.nonzero(as_tuple=True)[0]
        if valid_indices.numel() == 0:
            continue

        first_valid = valid_indices[0].item()
        # Cycle 0 = prompt + first response cycle.
        # Absolute positions in the input_ids: [first_valid, first_valid+1, ...]
        # Cycle i's absolute start position in input_ids:
        #   prompt_end + cycle_boundaries[i]
        # where prompt_end = first_valid + (prompt_length - left_pad_count)
        #                  = position where response starts.

        # Within the input_ids sequence:
        # - input_ids[: prompt_length] = left padding + prompt
        # - input_ids[prompt_length :] = response (right-padded)
        # The actual prompt content begins at first_valid (left-padding ends there).
        # The response begins at index `prompt_length` (always).

        response_start = prompt_length  # absolute index in input_ids

        # Cycle starts within input_ids:
        # cycle 0 starts at first_valid (the prompt's first non-pad token)
        # cycle i (i>=1) starts at response_start + cycle_boundaries[i]
        cycle_start_in_input = [first_valid]
        for cb in cycle_boundaries[1:]:
            cycle_start_in_input.append(response_start + cb)

        # Assign positions: within each cycle, positions go 0, 1, 2, ...
        for ci, start in enumerate(cycle_start_in_input):
            # End of this cycle: start of next cycle, or end of valid content
            if ci + 1 < len(cycle_start_in_input):
                end = cycle_start_in_input[ci + 1]
            else:
                # Last valid token + 1
                end = valid_indices[-1].item() + 1

            length = end - start
            if length > 0:
                position_ids[b, start:end] = torch.arange(length, dtype=torch.long)

    return position_ids


class HermeneuticAgentLoopWorker(AgentLoopWorker):
    """
    Reshapes position_ids in agent loop output so each cycle starts at position 0.

    This makes transformers' flash_attention detect packed sequences and apply
    block-diagonal attention isolation across cycles.
    """

    async def _agent_loop_postprocess(self, output, validate, **kwargs):
        # 1. Let parent do standard padding / mask construction
        internal = await super()._agent_loop_postprocess(output, validate, **kwargs)

        # 2. Get cycle boundaries from agent loop's extra_fields
        boundaries = output.extra_fields.get("cycle_boundary_positions")
        prompt_len = output.extra_fields.get("prompt_length")

        if boundaries is None or len(boundaries) <= 1 or prompt_len is None:
            # Single-cycle trajectory — no reset needed. Standard position_ids works.
            return internal

        # 3. Recompute position_ids with per-cycle reset
        new_position_ids = compute_position_ids_for_cycles(
            internal.attention_mask, boundaries, prompt_len
        )

        # Apply to the internal output
        internal.position_ids = new_position_ids

        return internal


class HermeneuticAgentLoopManager(AgentLoopManager):
    """
    Wires the HermeneuticAgentLoopWorker via verl's `for recipe to change` hook.
    """

    def __init__(self, *args, **kwargs):
        # Override the worker class before parent init reads it.
        self.agent_loop_workers_class = ray.remote(HermeneuticAgentLoopWorker)
        super().__init__(*args, **kwargs)
        print("[HermeneuticSearch] AgentLoopManager initialized with custom worker")
