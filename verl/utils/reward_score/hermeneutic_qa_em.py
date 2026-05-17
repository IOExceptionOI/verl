"""
Reward for Hermeneutic Search — staged outcome-based reward.

Design:
- During warmup (first `format_warmup_steps` steps): give small format_score (0.05)
  to bootstrap base model that doesn't yet know the answer format.
- After warmup: format_score linearly decays to 0 over `format_decay_steps`.
- Correct answer always gets correct_score (1.0).
- Transform bonus (+0.1) only when correct AND used <transform>.
- Anti-hack: multi-<answer> tags → 0 regardless of correctness.

This matches the curriculum learning / staged-reward pattern from R1-Searcher
(stage 1 teaches format, stage 2 teaches answer correctness) but as a continuous
schedule instead of two separate trainings.

To use, set reward.reward_manager.name=hermeneutic in the training script so
the HermeneuticRewardManager injects `global_steps` into extra_info.
"""

import random
import re
import string


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return True
    return False


def extract_solution(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if len(matches) < 1:
        return None
    return matches[-1].group(1).strip()


def has_valid_transform(solution_str):
    pattern = r"<transform>(.*?)</transform>"
    matches = list(re.finditer(pattern, solution_str, re.DOTALL))
    return any(len(m.group(1).strip()) > 0 for m in matches)


def get_format_score(step: int) -> float:
    """
    Staged format_score schedule:
    - step 0   - 100: 0.1   (full bootstrap)
    - step 100 - 200: 0.05  (half — gradual handoff)
    - step 200+      : 0     (pure outcome reward)
    """
    if step < 100:
        return 0.1
    if step < 200:
        return 0.05
    return 0.0


def compute_score(
    solution_str,
    ground_truth,
    data_source=None,
    extra_info=None,
    correct_score=1.0,
    transform_bonus=0.1,
    **kwargs,
):
    """
    Outcome-based reward with staged format_score.

    Reward structure:
    - correct answer + used <transform>  → 1.1
    - correct answer (no transform)      → 1.0
    - has <answer> tag but wrong         → format_score (staged: 0.1 → 0.05 → 0)
    - no <answer> tag                    → 0

    NOTE: anti-hack (multi-<answer> → 0) is removed. Base models sometimes
    emit several <answer> tags inline; extract_solution takes the LAST one,
    so taking format_score for "model put SOME answer in" is fine.

    extra_info["global_steps"] is required (injected by HermeneuticRewardManager).
    """
    step = (extra_info or {}).get("global_steps", 0) or 0
    format_score = get_format_score(step)

    answer = extract_solution(solution_str)
    has_transform = has_valid_transform(solution_str)

    do_print = random.randint(1, 32) == 1

    if answer is None:
        if do_print:
            print(f"[REWARD] no_answer (step={step}, format={format_score:.3f}) → 0")
        return 0.0

    if em_check(answer, ground_truth["target"]):
        reward = correct_score + (transform_bonus if has_transform else 0.0)
        print(
            f"[REWARD] CORRECT={reward:.2f} (step={step}) | transform={has_transform} | "
            f"ans={answer[:60]} | gt={list(ground_truth['target'])[:2]}"
        )
        if do_print:
            print(f"[TRAJECTORY] {solution_str[:500]}")
        return reward
    else:
        if do_print:
            print(f"[REWARD] format={format_score:.3f} (step={step}) | transform={has_transform} | ans={answer[:50]}")
        return format_score
