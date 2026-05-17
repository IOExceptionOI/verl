"""
Reward for Hermeneutic Search.
- format_score=0.1: has <answer> tag but wrong answer
- score=1.0: correct answer (EM)
- 0.0: no <answer> tag
Logs format vs answer reward separately for tracking.
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


def compute_score(solution_str, ground_truth, data_source=None, extra_info=None, format_score=0.1, score=1.0, **kwargs):
    """
    Reward: 1.0 correct, 0.1 format-only, 0.0 no answer tag.
    """
    answer = extract_solution(solution_str)
    has_transform = "<transform>" in solution_str and "</transform>" in solution_str
    do_print = random.randint(1, 32) == 1

    if answer is None:
        if do_print:
            print(f"[REWARD] no_answer | transform={has_transform} | gt={ground_truth['target'][:2]}")
        return 0.0

    if em_check(answer, ground_truth["target"]):
        # Always print correct answers for tracking
        print(f"[REWARD] CORRECT=1.0 | transform={has_transform} | ans={answer[:80]} | gt={ground_truth['target'][:2]}")
        if do_print:
            print(f"[TRAJECTORY] {solution_str[:500]}")
        return score
    else:
        if do_print:
            print(f"[REWARD] format=0.1 | transform={has_transform} | ans={answer[:50]} | gt={ground_truth['target'][:2]}")
        return format_score
