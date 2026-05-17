"""
Reward for Hermeneutic Search — outcome-based, anti-hacking.

Design rationale:
- Previous version gave format_score=0.1 for any trajectory with an <answer> tag.
  This was reward-hacked: model learned to spam <answer>X</answer> repeatedly to
  guarantee 0.1 reward, collapsing training (see training_notes.md "Experiment 4").
- New design follows SR1 (outcome-only) plus a transform bonus to encourage the
  hermeneutic behavior:
    - correct answer + used <transform>     → 1.1
    - correct answer (no transform)         → 1.0
    - everything else (wrong / no answer)   → 0.0
- Anti-hacking: repeated <answer> tags (>1) → reward=0 (kills the spam hack).

Note: there is NO format_score. Wrong answer = 0 reward, regardless of tags.
This matches SR1's design and eliminates the cheap-reward attractor.
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
    """Extract the last <answer>...</answer> content. None if no valid match."""
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if len(matches) < 1:
        return None
    return matches[-1].group(1).strip()


def has_valid_transform(solution_str):
    """Check if a non-empty <transform>...</transform> appeared in the trajectory."""
    pattern = r"<transform>(.*?)</transform>"
    matches = list(re.finditer(pattern, solution_str, re.DOTALL))
    return any(len(m.group(1).strip()) > 0 for m in matches)


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
    Outcome-based reward with transform bonus.

    Args:
        solution_str: full trajectory text
        ground_truth: dict with 'target' (list of acceptable answers)
        correct_score: reward for correct answer (default 1.0)
        transform_bonus: extra reward if a valid <transform> was used (default 0.1)

    Returns:
        1.0 + transform_bonus if correct answer and used transform
        1.0                   if correct answer (no transform)
        0.0                   otherwise (wrong answer, no answer tag, or hacking)
    """
    answer = extract_solution(solution_str)
    has_transform = has_valid_transform(solution_str)
    answer_tag_count = solution_str.count("<answer>")

    do_print = random.randint(1, 32) == 1

    # Anti-hacking: if model spams multiple <answer> tags, treat as gaming.
    if answer_tag_count > 1:
        if do_print:
            print(f"[REWARD] HACK (multi_answer={answer_tag_count}) → 0 | transform={has_transform}")
        return 0.0

    if answer is None:
        if do_print:
            print(f"[REWARD] no_answer → 0 | transform={has_transform} | gt={list(ground_truth['target'])[:2]}")
        return 0.0

    if em_check(answer, ground_truth["target"]):
        reward = correct_score + (transform_bonus if has_transform else 0.0)
        # Always print correct answers for tracking
        print(
            f"[REWARD] CORRECT={reward:.1f} | transform={has_transform} | "
            f"ans={answer[:80]} | gt={list(ground_truth['target'])[:2]}"
        )
        if do_print:
            print(f"[TRAJECTORY] {solution_str[:500]}")
        return reward
    else:
        if do_print:
            print(f"[REWARD] wrong → 0 | transform={has_transform} | ans={answer[:50]} | gt={list(ground_truth['target'])[:2]}")
        return 0.0
