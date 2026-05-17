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
    response_start_abs: int,
) -> torch.Tensor:
    """
    Compute position_ids with per-cycle reset.

    Args:
        attention_mask: [bsz, seq_len], 1 for content, 0 for left/right padding.
            Final layout: [left_pad..., prompt_tokens, response_tokens, right_pad...]
            with response starting at absolute index `response_start_abs`.
        cycle_boundaries: positions in response_ids where each cycle starts
            (relative to the response, not the full input).
            E.g., [0, 250] means cycle 0 at response[0], cycle 1 at response[250].
        response_start_abs: absolute index in input_ids where the response begins.
            This is the PADDED prompt width (e.g. rollout_config.prompt_length),
            NOT the unpadded prompt length.

    Returns:
        position_ids: [bsz, seq_len], int64. Each cycle's tokens have positions
        starting from 0. Padded positions stay 0 (will be dropped by unpad_input).
    """
    bsz, seq_len = attention_mask.shape
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)

    for b in range(bsz):
        mask = attention_mask[b]
        valid_indices = mask.nonzero(as_tuple=True)[0]
        if valid_indices.numel() == 0:
            continue
        first_valid = valid_indices[0].item()
        last_valid = valid_indices[-1].item()

        # Filter boundaries: must be within valid response region.
        # Cycle i (i >= 1) starts at response_start_abs + cb.
        # We need that position to be valid (within last_valid).
        # Cycle 0 always starts at first_valid (the prompt's first non-pad token).
        valid_starts = [first_valid]
        for cb in cycle_boundaries[1:]:
            absolute_start = response_start_abs + cb
            # Must be (a) within the sample, (b) at a valid (mask=1) position,
            # and (c) strictly after the previous start (monotonic).
            if (
                absolute_start <= last_valid
                and mask[absolute_start].item() == 1
                and absolute_start > valid_starts[-1]
            ):
                valid_starts.append(absolute_start)

        # Assign positions: within each cycle's segment, positions go 0, 1, 2, ...
        for ci, start in enumerate(valid_starts):
            if ci + 1 < len(valid_starts):
                end = valid_starts[ci + 1]
            else:
                end = last_valid + 1

            length = end - start
            if length > 0:
                # Only write positions where attention_mask is actually 1
                # (defensive: avoids putting positive position ids on right-pad).
                segment_mask = mask[start:end].bool()
                positions = torch.arange(length, dtype=torch.long)
                # Where mask is 1, use the position; where mask is 0, keep 0.
                position_ids[b, start:end] = torch.where(
                    segment_mask, positions, torch.zeros_like(positions)
                )

    return position_ids


class HermeneuticAgentLoopWorker(AgentLoopWorker):
    """
    Reshapes position_ids in agent loop output so each cycle starts at position 0.

    This makes transformers' flash_attention detect packed sequences and apply
    block-diagonal attention isolation across cycles.
    """

    async def _agent_loop_postprocess(self, output, validate, **kwargs):
        # 1. Let parent do standard padding / mask construction.
        internal = await super()._agent_loop_postprocess(output, validate, **kwargs)

        # 2. Get cycle boundaries from agent loop's extra_fields.
        boundaries = output.extra_fields.get("cycle_boundary_positions")

        if boundaries is None or len(boundaries) <= 1:
            # Single-cycle trajectory — no reset needed.
            return internal

        # 3. The response starts at the PADDED prompt width, not the unpadded length.
        #    `internal.prompt_ids` is left-padded to `rollout_config.prompt_length`.
        response_start_abs = internal.prompt_ids.shape[1]

        # 4. Filter boundaries beyond the actual response length.
        #    response_ids was truncated to max_total_response in agent loop,
        #    but boundaries may still point past that.
        response_length = internal.response_ids.shape[1]
        boundaries_clipped = [b for b in boundaries if 0 <= b < response_length]
        if len(boundaries_clipped) <= 1:
            # After clipping, only one cycle survives — no reset needed.
            return internal

        # 5. Recompute position_ids with per-cycle reset.
        new_position_ids = compute_position_ids_for_cycles(
            internal.attention_mask, boundaries_clipped, response_start_abs
        )

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
