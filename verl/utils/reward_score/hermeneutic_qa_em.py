"""
Reward for Hermeneutic Search — 3-component format reward + outcome.

Design:
- 3 format components, each worth 0.1 in warmup, decaying together:
    * has <answer> tag
    * has valid <transform> (content > 10 chars, prevents 'and' hack)
    * has <tool_call> (used search)
- Format reward max = 0.3 (all three used) in warmup, 0.15 mid, 0 late.
- Correct answer: 1.0 (+0.1 transform_bonus if used) — independent of format.

Schedule:
- step 0-100:   format scale = 1.0  → max format 0.3
- step 100-200: format scale = 0.5  → max format 0.15
- step 200+:    format scale = 0.0  → pure outcome (correct only)
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
    """Valid transform: content > 10 chars (filter out 'and', empty, single token)."""
    pattern = r"<transform>(.*?)</transform>"
    matches = list(re.finditer(pattern, solution_str, re.DOTALL))
    return any(len(m.group(1).strip()) > 10 for m in matches)


def has_search(solution_str):
    """Did the trajectory issue any <tool_call>?"""
    return "<tool_call>" in solution_str and "</tool_call>" in solution_str


def get_format_scale(step: int) -> float:
    """
    Format reward scale:
    - 0-100:   1.0 (full bootstrap)
    - 100-200: 0.5 (half)
    - 200+:    0.0 (pure outcome)
    """
    if step < 100:
        return 1.0
    if step < 200:
        return 0.5
    return 0.0


def compute_score(
    solution_str,
    ground_truth,
    data_source=None,
    extra_info=None,
    correct_score=1.0,
    transform_bonus=0.1,
    component_bonus=0.1,
    **kwargs,
):
    """
    Reward = max(correct_reward, format_reward).

    correct_reward (only if EM): correct_score + transform_bonus (if transform used)
    format_reward (always, scaled by step):
        component_bonus * (has_answer_tag + has_valid_transform + has_search)
        × format_scale(step)
    """
    step = (extra_info or {}).get("global_steps", 0) or 0
    format_scale = get_format_scale(step)

    answer = extract_solution(solution_str)
    has_tr = has_valid_transform(solution_str)
    has_sr = has_search(solution_str)
    has_ans_tag = answer is not None

    do_print = random.randint(1, 32) == 1

    # 3-component format reward (only when format_scale > 0)
    if format_scale > 0:
        n_components = int(has_ans_tag) + int(has_tr) + int(has_sr)
        format_reward = component_bonus * n_components * format_scale
    else:
        format_reward = 0.0

    if not has_ans_tag:
        if do_print:
            print(f"[REWARD] no_answer (step={step}, format={format_reward:.3f}) | search={has_sr} transform={has_tr}")
        return format_reward  # could still earn from transform/search components

    if em_check(answer, ground_truth["target"]):
        reward = correct_score + (transform_bonus if has_tr else 0.0)
        print(
            f"[REWARD] CORRECT={reward:.2f} (step={step}) | transform={has_tr} search={has_sr} | "
            f"ans={answer[:60]} | gt={list(ground_truth['target'])[:2]}"
        )
        if do_print:
            print(f"[TRAJECTORY] {solution_str[:500]}")
        return reward

    # Wrong answer, return format reward (which includes answer tag bonus)
    if do_print:
        print(
            f"[REWARD] wrong format={format_reward:.3f} (step={step}) | "
            f"ans_tag={has_ans_tag} transform={has_tr} search={has_sr} | ans={answer[:50]}"
        )
    return format_reward
