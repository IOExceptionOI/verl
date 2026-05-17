"""
Hermeneutic Search: Inference Evaluation

Runs the hermeneutic cycle (question -> search -> transform/answer) on a base model.
Each cycle: model receives ONLY the current question (context resets on transform).
Tests whether a base model can follow this paradigm without RL training.

Usage:
    python eval_hermeneutic.py \
        --model_url http://localhost:8080/v1 \
        --retriever_url http://localhost:8000/retrieve \
        --data_path /workspace/verl/data/searchR1_processed/test.parquet \
        --max_cycles 5 \
        --num_samples 100
"""

import argparse
import json
import re
import time
from typing import Optional
import requests
import pandas as pd
from collections import defaultdict


SYSTEM_PROMPT = """You are a search agent that answers questions through iterative search refinement.

In each cycle, you receive a question and can either:
1. Search for information: output <search>your query</search>
2. Transform the question into a more refined version (incorporating what you learned): output <transform>refined question</transform>
3. Give the final answer: output <answer>your answer</answer>

IMPORTANT RULES:
- Think step by step inside <think>...</think> before each action.
- After receiving search results, you MUST either <transform> the question or <answer> directly.
- When you <transform>, the old question disappears entirely. Your new question must be self-contained — it should incorporate all useful findings and exclude dead ends.
- A good transformed question narrows the search space while preserving open-endedness.
- When you are confident enough to answer, use <answer> instead of <transform>.
- You can only perform ONE action per turn (search, transform, or answer)."""


def call_model(model_url: str, messages: list, temperature: float = 0.7, max_tokens: int = 2048) -> str:
    resp = requests.post(
        f"{model_url}/chat/completions",
        json={
            "model": "default",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": ["</search>", "</transform>", "</answer>"],
        },
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    # re-append the stop token that was cut
    for tag in ["</search>", "</transform>", "</answer>"]:
        open_tag = tag.replace("/", "")
        if open_tag in content and tag not in content:
            content += tag
            break
    return content


def call_retriever(retriever_url: str, queries: list, topk: int = 3) -> list:
    resp = requests.post(
        retriever_url,
        json={"queries": queries, "topk": topk, "return_scores": True},
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json()["result"]
    formatted = []
    for result_list in results:
        passages = ""
        for idx, doc in enumerate(result_list):
            content = doc["document"]["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            passages += f"Doc {idx+1} (Title: {title})\n{text}\n\n"
        formatted.append(passages.strip())
    return formatted


def parse_action(response: str) -> tuple:
    """Parse model response into (action_type, content)."""
    patterns = [
        ("search", r"<search>(.*?)</search>"),
        ("transform", r"<transform>(.*?)</transform>"),
        ("answer", r"<answer>(.*?)</answer>"),
    ]
    for action_type, pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return action_type, match.group(1).strip()
    return "invalid", ""


def compute_f1(prediction: str, ground_truth: list) -> float:
    """Token-level F1 between prediction and best matching ground truth."""
    def tokenize(s):
        return s.lower().split()
    
    best_f1 = 0.0
    pred_tokens = tokenize(prediction)
    for gt in ground_truth:
        gt_tokens = tokenize(gt)
        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            continue
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(gt_tokens) if gt_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        best_f1 = max(best_f1, f1)
    return best_f1


def compute_em(prediction: str, ground_truth: list) -> float:
    pred_norm = prediction.lower().strip()
    return float(any(pred_norm == gt.lower().strip() for gt in ground_truth))


def run_hermeneutic_cycle(
    question: str,
    model_url: str,
    retriever_url: str,
    max_cycles: int = 5,
    topk: int = 3,
    verbose: bool = False,
) -> dict:
    """Run one hermeneutic search episode."""
    current_question = question
    trajectory = []
    
    for cycle in range(max_cycles):
        # Phase 1: Model sees ONLY the current question, decides to search or answer
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": current_question},
        ]
        
        response = call_model(model_url, messages)
        action, content = parse_action(response)
        trajectory.append({"cycle": cycle, "phase": "initial", "action": action, "content": content, "question": current_question})
        
        if verbose:
            print(f"  [Cycle {cycle}] Q: {current_question[:80]}...")
            print(f"  [Cycle {cycle}] Action: {action} -> {content[:80]}...")
        
        if action == "answer":
            return {"answer": content, "trajectory": trajectory, "cycles": cycle + 1, "status": "answered"}
        
        if action == "search":
            # Phase 2: Execute search
            search_results = call_retriever(retriever_url, [content], topk=topk)
            search_info = search_results[0] if search_results else "No results found."
            
            # Phase 3: Model sees question + search results, must transform or answer
            messages_with_results = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": current_question},
                {"role": "assistant", "content": response},
                {"role": "user", "content": f"<information>\n{search_info}\n</information>\n\nBased on the search results, either <transform> the question into a more refined version, or <answer> if you can."},
            ]
            
            response2 = call_model(model_url, messages_with_results)
            action2, content2 = parse_action(response2)
            trajectory.append({"cycle": cycle, "phase": "post_search", "action": action2, "content": content2})
            
            if verbose:
                print(f"  [Cycle {cycle}] Post-search: {action2} -> {content2[:80]}...")
            
            if action2 == "answer":
                return {"answer": content2, "trajectory": trajectory, "cycles": cycle + 1, "status": "answered"}
            elif action2 == "transform":
                # CONTEXT RESET: new question replaces everything
                current_question = content2
            else:
                # Invalid action after search, force continue with same question
                trajectory.append({"cycle": cycle, "phase": "fallback", "note": "invalid action after search, retrying"})
        
        elif action == "transform":
            # Direct transform without search (unusual but allowed)
            current_question = content
        
        else:
            # Invalid initial action
            trajectory.append({"cycle": cycle, "phase": "fallback", "note": f"invalid action: {action}"})
    
    # Max cycles reached without answer — force a final answer attempt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{current_question}\n\nYou have reached the maximum number of search cycles. You MUST provide your best answer now using <answer>...</answer>."},
    ]
    response = call_model(model_url, messages)
    _, content = parse_action(response)
    trajectory.append({"cycle": max_cycles, "phase": "forced_answer", "content": content})
    return {"answer": content, "trajectory": trajectory, "cycles": max_cycles, "status": "max_cycles"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_url", default="http://localhost:8080/v1")
    parser.add_argument("--retriever_url", default="http://localhost:8000/retrieve")
    parser.add_argument("--data_path", default="/workspace/verl/data/searchR1_processed/test.parquet")
    parser.add_argument("--max_cycles", type=int, default=5)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", default="hermeneutic_eval_results.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(args.data_path)
    samples = df.head(args.num_samples)
    
    results = []
    metrics = defaultdict(list)
    
    for idx, row in samples.iterrows():
        question = row["extra_info"]["question"]
        ground_truth = list(row["reward_model"]["ground_truth"]["target"])
        
        print(f"[{idx+1}/{args.num_samples}] Q: {question[:60]}...")
        
        try:
            result = run_hermeneutic_cycle(
                question=question,
                model_url=args.model_url,
                retriever_url=args.retriever_url,
                max_cycles=args.max_cycles,
                topk=args.topk,
                verbose=args.verbose,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            result = {"answer": "", "trajectory": [], "cycles": 0, "status": "error", "error": str(e)}
        
        f1 = compute_f1(result["answer"], ground_truth)
        em = compute_em(result["answer"], ground_truth)
        
        metrics["f1"].append(f1)
        metrics["em"].append(em)
        metrics["cycles"].append(result["cycles"])
        metrics["status"].append(result["status"])
        
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "prediction": result["answer"],
            "f1": f1,
            "em": em,
            "cycles": result["cycles"],
            "status": result["status"],
            "trajectory": result["trajectory"],
        })
        
        print(f"  -> {result['status']} in {result['cycles']} cycles | F1={f1:.2f} EM={em:.0f} | Pred: {result['answer'][:50]}")
    
    # Summary
    avg_f1 = sum(metrics["f1"]) / len(metrics["f1"]) if metrics["f1"] else 0
    avg_em = sum(metrics["em"]) / len(metrics["em"]) if metrics["em"] else 0
    avg_cycles = sum(metrics["cycles"]) / len(metrics["cycles"]) if metrics["cycles"] else 0
    
    from collections import Counter
    status_counts = Counter(metrics["status"])
    
    summary = {
        "num_samples": len(results),
        "avg_f1": avg_f1,
        "avg_em": avg_em,
        "avg_cycles": avg_cycles,
        "status_distribution": dict(status_counts),
        "model_url": args.model_url,
        "max_cycles": args.max_cycles,
    }
    
    print("\n" + "="*60)
    print(f"RESULTS: {len(results)} samples")
    print(f"  Avg F1:     {avg_f1:.4f}")
    print(f"  Avg EM:     {avg_em:.4f}")
    print(f"  Avg Cycles: {avg_cycles:.2f}")
    print(f"  Status:     {dict(status_counts)}")
    print("="*60)
    
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
