"""
Reward function for Hermeneutic Search training.

Computes reward based on:
1. Answer correctness (F1/EM against ground truth) - primary signal
2. Format compliance bonus (valid actions throughout)

Registers as a verl reward function via the standard interface.
"""

import re
from typing import Any


def compute_f1(prediction: str, ground_truths: list) -> float:
    def tokenize(s):
        return s.lower().split()
    
    pred_tokens = tokenize(prediction)
    if not pred_tokens:
        return 0.0
    
    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = tokenize(gt)
        if not gt_tokens:
            continue
        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            continue
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


def compute_em(prediction: str, ground_truths: list) -> float:
    pred_norm = prediction.lower().strip()
    return float(any(pred_norm == gt.lower().strip() for gt in ground_truths))


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def hermeneutic_reward_function(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any] = None,
    **kwargs,
) -> float:
    """
    Reward function compatible with verl's reward_function interface.
    
    Args:
        data_source: dataset name
        solution_str: full model output (all turns concatenated)
        ground_truth: dict with 'target' key containing list of acceptable answers
        extra_info: additional info from dataset
    
    Returns:
        reward score in [0, 1]
    """
    targets = list(ground_truth.get("target", []))
    if not targets:
        return 0.0
    
    answer = extract_answer(solution_str)
    if not answer:
        return 0.0
    
    # Use F1 as primary reward (smoother signal than EM)
    f1 = compute_f1(answer, targets)
    em = compute_em(answer, targets)
    
    # Reward = max(F1, EM) to give credit for exact matches even with tokenization differences
    reward = max(f1, em)
    
    return reward
