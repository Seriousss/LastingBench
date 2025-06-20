
import argparse, os, jsonlines, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from utils.metrics import qa_f1_score, qa_em_score
THINK_END_ID = 151668        # "</think>" token id for Qwen3

# --------------------------------------------------
def strip_think(token_ids):
    try:
        cut = len(token_ids) - token_ids[::-1].index(THINK_END_ID)
        return token_ids[cut:]
    except ValueError:
        return token_ids

def main():
    # ---------- CLI ----------
    parser = argparse.ArgumentParser(
        description="Evaluate HotpotQA JSONL with Transformers + Qwen3-8B"
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Path to input JSONL file")
    parser.add_argument("--model", required=True,
                        help="HF model name, e.g. Qwen/Qwen3-8B")
    parser.add_argument("-d", "--devices", default="0",
                        help="CUDA_VISIBLE_DEVICES (comma-separated)")
    parser.add_argument("-t", "--temperature", type=float, default=0.5,
                        help="Sampling temperature")
    parser.add_argument("-k", "--max_tokens", type=int, default=40,
                        help="max_new_tokens")
    args = parser.parse_args()



    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    gen_cfg = GenerationConfig(
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        do_sample=args.temperature > 0
    )


    with jsonlines.open(args.input) as reader:
        data = list(reader)

    total_f1 = total_em = 0.0

    for idx, item in enumerate(data):
        question = item.get("input", "")
        context  = item.get("context", "")
        answers  = item.get("answers", [])
        if not answers:
            print(f"[{idx}] no gold answer, skip")
            continue
        gold = answers[0]
        print(gold)

        # ----- Prompt -----
        prompt = (
            "Answer the question based on the given passages. "
            "Only give me your answer and do not output any other words.\n"
            "Passages:\n"
            f"{context}\n"
            f"Question: {question}\n"
            "Answer:"
        )
        messages = [{"role": "user", "content": prompt}]
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)


        # ----- Generate -----
        try:
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=args.max_tokens)
        except ValueError as e:
            if "position ids exceed" in str(e).lower() or "sequence length" in str(e).lower():
                print(f"[{idx}] prompt too long – skipped")
                continue
            raise
        print("im here")
        new_ids = outputs[0][len(inputs.input_ids[0]):].tolist() 
        try:
            index = len(new_ids) - new_ids[::-1].index(151668)
        except ValueError:
            index = 0
        answer = tokenizer.decode(new_ids[index:], skip_special_tokens=True).strip("\n")
        answer = answer.strip()  

        # ----- Score -----
        f1 = qa_f1_score(answer, gold)
        em = qa_em_score(answer, gold)
        total_f1 += f1
        total_em += em

        print(f"[{idx}] Q: {question}")
        print(f"    Resp: {answer!r} | Gold: {gold!r}")
        print(f"    F1={f1:.2f}, EM={em:.2f}")

    n = len(data)
    print(f"\nOverall F1: {total_f1/n:.4f}")
    print(f"Overall EM: {total_em/n:.4f}")

if __name__ == "__main__":
    main()