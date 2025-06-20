#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-phase evaluator (DeepSeek API) — Calculate EM / F1 only.

Usage Example
--------
python eval_single_phase.py --input data/2wikimqa.jsonl
"""

import argparse, time, jsonlines, os
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from utils.metrics import qa_em_score, qa_f1_score
from utils.llmjudge import judge_answer_with_api

# -------------------- CLI --------------------
p = argparse.ArgumentParser("Single-phase evaluator")
p.add_argument("--input", required=True, help="Path to the *.jsonl file to evaluate")
p.add_argument("--model",       default="deepseek-r1")
p.add_argument("--temperature", type=float, default=0.5)
p.add_argument("--max_tokens",  type=int,   default=30)
p.add_argument("--sleep",       type=float, default=0.0)
args = p.parse_args()

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY")
)

# -------------------- helper --------------------
def ask(context: str, question: str) -> str:
    """Call DeepSeek to get answer (return final answer only)"""
    messages = [
        {"role": "system",
         "content": ("You are a QA assistant. "
                     "Answer strictly based on the passages; "
                     "output only the final answer.")},
        {"role": "user",
         "content": f"Answer the question and output only the final answer without extra words. Passages:\n{context}\n\nQuestion: {question}\nAnswer:"}
    ]
    resp = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    if not resp.choices[0].message.content:
        return "None"

    return resp.choices[0].message.content.strip()


# -------------------- core eval --------------------
def evaluate_file(path: Path):
    dataset = path.stem
    data = {obj["input"]: obj for obj in jsonlines.open(path)}

    total   = len(data)
    em_hits = 0
    f1_sum  = 0.0

    for q, item in tqdm(data.items(), desc=f"{dataset}"):
        ctx   = item["context"]
        golds = item["answers"] if isinstance(item["answers"], list) else [item["answers"]]

        pred  = ask(ctx, q).split('.', 1)[0]        # Cut off extra explanations
        if pred == "None":
            continue
        em    = max(qa_em_score(pred, g)  for g in golds)
        f1    = max(qa_f1_score(pred, g) for g in golds)

        em_hits += em
        f1_sum  += f1
        if args.sleep:
            time.sleep(args.sleep)

    print(f"\n=== {dataset.upper()} SUMMARY ===")
    print(f"Total samples : {total}")
    print(f"Exact Match   : {em_hits}/{total}  ({em_hits/total:.2%})")
    print(f"Average F1    : {f1_sum/total:.4f}")
    print("-" * 40 + "\n")


# -------------------- run --------------------
input_path = Path(args.input)
if not input_path.exists():
    raise SystemExit(f"File does not exist: {input_path}")

evaluate_file(input_path)