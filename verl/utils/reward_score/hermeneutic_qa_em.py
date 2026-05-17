"""
Reward function for Hermeneutic Search.
Based on search_r1_like_qa_em.py, adds format reward for proper <transform> usage.

Reward structure:
- 1.0: correct answer in <answer>...</answer>
- 0.3: correct format (has <answer> tag) but wrong answer, AND used <transform> properly
- 0.1: correct format (has <answer> tag) but wrong answer, no transform
- 0.0: no valid <answer> tag found
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
    """Check if the trajectory used <transform>...</transform> properly."""
    pattern = r"<transform>(.*?)</transform>"
    matches = list(re.finditer(pattern, solution_str, re.DOTALL))
    # Valid: at least one transform with non-empty content
    return any(len(m.group(1).strip()) > 10 for m in matches)


def compute_score(solution_str, ground_truth, method="strict", format_score=0.1, transform_bonus=0.2, score=1.0):
    """
    Reward for hermeneutic search.

    Args:
        solution_str: full trajectory text
        ground_truth: dict with 'target' key
        format_score: reward for correct <answer> format but wrong answer
        transform_bonus: additional reward if <transform> was used properly
        score: reward for correct answer
    """
    answer = extract_solution(solution_str)
    used_transform = has_valid_transform(solution_str)
    open_count = solution_str.count("<answer>")
    close_count = solution_str.count("</answer>")

    do_print = random.randint(1, 64) == 1
    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Used transform: {used_transform}")
        print(f"Solution string (first 300): {solution_str[:300]}")

    if answer is None:
        return 0.0

    if em_check(answer, ground_truth["target"]):
        # Correct answer
        if open_count > 10 or close_count > 10:
            return score / 4
        return score
    else:
        # Wrong answer but correct format
        bonus = transform_bonus if used_transform else 0.0
        return format_score + bonus
