#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate models on a LongBench subset with Exact-Match (EM).
Supports both Qwen3 (Transformers) and other models (vLLM).

Requirements
------------
pip install vllm datasets tqdm transformers accelerate
"""

import argparse, logging, time, torch
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from utils.metrics import qa_em_score
import os

# ---------------------------- CLI ------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--hf_model",
    default="Qwen/Qwen3-8B-Instruct",
    help="Model name or local path")
parser.add_argument("--is_qwen3", action="store_true",
    help="Set this flag if using Qwen3 model (uses Transformers). Otherwise uses vLLM.")
parser.add_argument("--max_new_tokens", type=int, default=20)
parser.add_argument("--max_tokens", type=int, default=20,
    help="For vLLM models (ignored if --is_qwen3)")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--tensor_parallel_size", type=int, default=2,
    help="GPU parallel size for vLLM (ignored if --is_qwen3)")

parser.add_argument("--dataset_repo", default="THUDM/LongBench")
parser.add_argument("--dataset_subset", default="hotpotqa")
parser.add_argument("--split", default="test")
parser.add_argument("--sleep", type=float, default=0.0)
parser.add_argument("--log", default="summary.log")
parser.add_argument("--cuda_devices", default="1,6",
    help="CUDA visible devices")
args = parser.parse_args()

# Set CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

# --------------------------- logging ---------------------------------
logging.basicConfig(
    filename=args.log,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode="a",
)
logging.getLogger().addHandler(logging.StreamHandler())

# ------------------------- dataset -----------------------------------
ds = load_dataset(args.dataset_repo, args.dataset_subset, split=args.split)
total = len(ds)
logging.info("Loaded %d samples from %s/%s[%s]",
             total, args.dataset_repo, args.dataset_subset, args.split)

if args.is_qwen3:
    # ---------------------- Qwen3 with Transformers ----------------------------
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    load_kwargs = dict(
        trust_remote_code=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=torch.float16,
        **load_kwargs
    )

    EOS_ID      = tokenizer.eos_token_id
    THINK_ENDID = 151668  # </think> token id

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
        eos_token_id=EOS_ID,
    )

    # -------------------------- Qwen3 loop -------------------------------------
    correct_em = 0

    for ex in tqdm(ds, desc="Evaluating with Transformers (Qwen3)"):
        q = ex["input"]
        golds = ex["answers"]

        msgs = [
            {"role": "system", "content": "You are a QA assistant."},
            {"role": "user",
             "content": f"Question: {q}\n"
                        "Please reply with *only* the final answer—no extra words."}
        ]
        prompt = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False   # Qwen3 thinking mode
        )
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outs = model.generate(**inputs, **gen_kwargs)[0]

        # Extract newly generated tokens
        new_ids = outs[len(inputs.input_ids[0]):].tolist()

        # Find </think> (if not exist idx=0)
        try:
            idx = len(new_ids) - new_ids[::-1].index(THINK_ENDID)
        except ValueError:
            idx = 0

        content = tokenizer.decode(new_ids[idx:],
                                   skip_special_tokens=True).strip("\n").strip()

        # Only use content for EM comparison
        if any(qa_em_score(content, g) for g in golds):
            correct_em += 1

        if args.sleep:
            time.sleep(args.sleep)

else:
    # ---------------------- Other models with vLLM ----------------------------
    from vllm import LLM, SamplingParams
    
    # Initialize vLLM
    llm = LLM(
        model=args.hf_model,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    sampler = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        stop=["</assistant>", "</s>", "<|end_of_text|>"],
    )

    # -------------------------- vLLM loop -------------------------------------
    correct_em = 0

    for ex in tqdm(ds, desc="Evaluating with vLLM"):
        question = ex["input"]
        golds = ex["answers"]      # list[str]
        
        chat_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            stop=["</s>", "<|end_of_text|>"],   # Safety stop tokens
        )
        
        messages = [
            {"role": "system",
             "content": "You are a QA assistant."},
            {"role": "user",
             "content": f"Question: {question}\n"
                        "Please first reply with *only* the final answer—no extra words.\n Answer:"}
        ] 

        result = llm.chat(messages, sampling_params=chat_params)
        # vLLM returns list[RequestOutput]; take first output's first candidate
        pred = result[0].outputs[0].text.strip()
        print(f"A: {pred}\nG: {golds}\n")

        if any(qa_em_score(pred, g) for g in golds):
            correct_em += 1

        if args.sleep:
            time.sleep(args.sleep)

# -------------------------- result -----------------------------------
em = correct_em / total
model_type = "Qwen3 (Transformers)" if args.is_qwen3 else "vLLM"
logging.info("RESULT | model=%s | type=%s | subset=%s | EM=%.4f",
             args.hf_model, model_type, args.dataset_subset, em)
print(
    f"\n=== SUMMARY ===\n"
    f"Model   : {args.hf_model}\n"
    f"Type    : {model_type}\n"
    f"Subset  : {args.dataset_subset} ({args.split})\n"
    f"EM      : {em:.4f}\n"
    f"(Log in {Path(args.log).resolve()})"
)