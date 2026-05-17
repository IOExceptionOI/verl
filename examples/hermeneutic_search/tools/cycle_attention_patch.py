"""
Patch for cycle-aware attention mask in Hermeneutic Search training.

After rollout, the batch contains response_ids with PAD tokens between cycles.
This patch:
1. Detects PAD tokens within the response portion
2. Sets attention_mask=0 at those positions (creating cu_seqlens breaks for flash-attn)
3. Recomputes position_ids to reset at cycle boundaries

Usage: import this module BEFORE training starts. It patches the rollout manager's
generate_sequences to apply the fix automatically.

    from examples.hermeneutic_search.tools.cycle_attention_patch import patch_rollout_manager
    patch_rollout_manager(trainer.async_rollout_manager, tokenizer.pad_token_id)
"""

import torch
from verl.utils.model import compute_position_id_with_mask


def fix_cycle_attention_mask(batch, pad_token_id: int):
    """
    Fix attention_mask for hermeneutic search cycles.
    
    Finds PAD tokens within the response portion of input_ids and sets
    their attention_mask to 0, creating flash-attn document boundaries.
    Also recomputes position_ids to reset at cycle boundaries.
    """
    input_ids = batch.batch["input_ids"]  # [bsz, seq_len]
    attention_mask = batch.batch["attention_mask"]  # [bsz, seq_len]
    
    prompt_length = batch.batch["prompts"].shape[1] if "prompts" in batch.batch else 0
    
    # Find PAD tokens in the response portion (after prompt)
    # These are cycle separators inserted by HermeneuticAgentLoop
    if prompt_length > 0:
        response_ids = input_ids[:, prompt_length:]
        response_attn = attention_mask[:, prompt_length:]
    else:
        response_ids = input_ids
        response_attn = attention_mask
    
    # Mask: PAD tokens within the response that currently have attention_mask=1
    pad_in_response = (response_ids == pad_token_id) & (response_attn == 1)
    
    if not pad_in_response.any():
        return batch  # No cycle separators, nothing to fix
    
    # Zero out attention_mask at separator positions
    if prompt_length > 0:
        attention_mask[:, prompt_length:][pad_in_response] = 0
    else:
        attention_mask[pad_in_response] = 0
    
    # Recompute position_ids from the updated attention_mask
    batch.batch["attention_mask"] = attention_mask
    batch.batch["position_ids"] = compute_position_id_with_mask(attention_mask)
    
    return batch


def patch_rollout_manager(rollout_manager, pad_token_id: int):
    """
    Monkey-patch the rollout manager to apply cycle attention mask fix
    after generate_sequences.
    """
    original_generate = rollout_manager.generate_sequences
    
    def patched_generate(batch, *args, **kwargs):
        result = original_generate(batch, *args, **kwargs)
        return fix_cycle_attention_mask(result, pad_token_id)
    
    rollout_manager.generate_sequences = patched_generate
    print("[HermeneuticSearch] Patched generate_sequences with cycle attention mask fix")
