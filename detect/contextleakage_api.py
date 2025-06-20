import os, time, argparse, logging
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from utils.metrics import qa_em_score  

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model",          default="gpt-4o")
parser.add_argument("--dataset_repo",   default="THUDM/LongBench")
parser.add_argument("--dataset_subset", default="hotpotqa")
parser.add_argument("--split",          default="test")
parser.add_argument("--max_tokens",     type=int,   default=30)
parser.add_argument("--temperature",    type=float, default=0.0)
parser.add_argument("--sleep",          type=float, default=0.5,
                    help="seconds to wait between requests")
parser.add_argument("--log",            default="summary.log",
                    help="append overall score here")
args = parser.parse_args()

# ----------------------------------------------------------------------
# Logging (append mode)
# ----------------------------------------------------------------------
logging.basicConfig(
    filename=args.log,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode="a",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# ----------------------------------------------------------------------
# OpenAI client
# ----------------------------------------------------------------------
client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY"), 
  base_url=os.environ.get("OPENAI_BASE_URL")
)

# ----------------------------------------------------------------------
# Load dataset
# ----------------------------------------------------------------------
ds = load_dataset(args.dataset_repo, args.dataset_subset, split=args.split)
total = len(ds)
logging.info("Loaded %d samples from %s/%s[%s]",
             total, args.dataset_repo, args.dataset_subset, args.split)

# ----------------------------------------------------------------------
# Evaluation loop
# ----------------------------------------------------------------------
correct_em = 0

for ex in tqdm(ds, desc="Evaluating"):
    question = ex["input"]
    golds    = ex["answers"]

    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": "You are a QA assistant."},
            {"role": "user",
            "content": f"Question: {question}\n"
            "Please first reply with *only* the final answer—no extra words.\n Answer:"}
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    pred = resp.choices[0].message.content.strip()
    print(f"A: {pred}\n G: {golds}")

    if any(qa_em_score(pred, g) for g in golds):
        correct_em += 1

    time.sleep(args.sleep)

em_score = correct_em / total
logging.info("RESULT | model=%s | subset=%s | EM=%.4f",
             args.model, args.dataset_subset, em_score)

print(f"\n=== SUMMARY ===\nModel   : {args.model}"
      f"\nDataset : {args.dataset_subset} ({args.split})"
      f"\nEM      : {em_score:.4f}\n"
      f"(Appended to {args.log})")