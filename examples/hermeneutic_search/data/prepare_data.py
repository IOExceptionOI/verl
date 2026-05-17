"""
Prepare training data for Hermeneutic Search.
Uses exact same <tool_call> format as SR1 (proven to work with verl's Hermes parser),
adds <transform> as additional action.
"""

import argparse
import pandas as pd
from pathlib import Path


SYSTEM_PROMPT = "You are a helpful and harmless assistant."

# Match SR1's proven format exactly, just add transform option
USER_TEMPLATE = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <tool_call> query </tool_call> and it will return the top searched results between <tool_response> and </tool_response>. You can search as many times as you want.

After searching, you have two choices:
1. If you can answer confidently, provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.
2. If you want to refine your question based on what you learned, use <transform> refined_question </transform>. This reformulates a more specific question that incorporates your findings and drops dead-end paths. After transform, focus on the new question for your next search.

Important: Don't transform forever — commit to an answer when you have enough information.

Question: {question}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} samples from {args.input}")

    new_prompts = []
    for _, row in df.iterrows():
        question = row["extra_info"]["question"]
        new_prompts.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(question=question)},
        ])

    df = df.copy()
    df["prompt"] = new_prompts

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df)} samples to {args.output}")


if __name__ == "__main__":
    main()
