import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert and merge JSONL files with question-answer mappings')
    parser.add_argument('orig_path', help='Path to the original JSONL file')
    parser.add_argument('out_path', help='Path to the output JSONL file')
    parser.add_argument('mapping_paths', nargs='+', help='Path(s) to mapping JSONL file(s)')
    
    args = parser.parse_args()
    
    # Original data file paths from command line arguments
    orig_path = args.orig_path
    out_path = args.out_path
    mapping_paths = args.mapping_paths

    # Step 1: Build question -> {context, answers} mapping
    mapping = {}
    for mp in mapping_paths:
        with open(mp, 'r', encoding='utf-8') as f_map:
            for idx, line in enumerate(f_map):
                obj = json.loads(line)
                q = obj.get("question")
                if q is None:
                    continue
                # Ensure we get the context
                ctx = obj.get("context", "")
                # Some files have "answer" field, some have "answers"
                raw_ans = obj.get("answers", obj.get("answer", []))
                # Normalize answer(s) to list format
                if isinstance(raw_ans, list):
                    ans = raw_ans
                else:
                    ans = [raw_ans]
                # If the same question appears in multiple mapping files, later ones will overwrite earlier ones
                mapping[q] = {"context": ctx, "answers": ans}

    # Step 2: Read original file, perform replacement and write output
    with open(orig_path, 'r', encoding='utf-8') as f_in, \
         open(out_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            item = json.loads(line)
            inp = item.get("input")
            if inp in mapping:
                item["context"] = mapping[inp]["context"]
                item["answers"] = mapping[inp]["answers"]
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Merge completed, output file: {out_path}")

if __name__ == "__main__":
    main()