import json
import time
import re
import os
import argparse
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from utils.util import retriveDoc,compute_best_sentence_f1
from openai import OpenAI
import asyncio, json, torch, math
from typing import List, Tuple
# Hugging Face transformers related
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils.metrics import qa_f1_score
from utils.llmjudge import judge_answer_with_api


client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY")
)
# Load models using transformers

tokenizer1 = AutoTokenizer.from_pretrained("Qwen/Qwen-14B-Chat", trust_remote_code=True)
model1 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-14B-Chat", trust_remote_code=True,device_map="cuda:0",torch_dtype=torch.bfloat16)


tok_qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
model_qwen = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True,
    device_map="cuda:1",torch_dtype=torch.bfloat16
).eval()

def get_transformers_answer(prompt, tokenizer, model, max_new_tokens=100, temperature=0.7, top_p=0.9, retries=3, delay=5):
    """
    Use transformers model.generate method for inference with retry mechanism,
    use chat template to format input, and strip the input prompt part through token-level slicing,
    return the newly generated text.
    """
    import time
    for attempt in range(retries):
        try:
            # Convert original prompt to message format
            messages = [{"role": "user", "content": prompt}]
            
            # Try to use chat template to format input
            try:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"Unable to apply chat template: {e}, falling back to basic text input")
                formatted_prompt = prompt  # Fall back to original prompt as input
            
            # Encode formatted prompt as model input tensor
            model_inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            
            # Call generate, the generated id sequence contains both prompt and subsequent generated text
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            # Calculate the token count corresponding to the prompt
            input_length = model_inputs.input_ids.shape[1]
            
            # Strip the prompt part from the front of the output, keeping only the newly added part
            output_ids = generated_ids[0][input_length:]
            
            # Decode generated text
            answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            return answer
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached, skipping this request.")
                return None

def truncate_answer(answer):
    """Truncate answer, only take the part before the first period"""
    return answer.split('.')[0].strip() if answer else "No answer"

def write_to_log(filename, data):
    """Write data to log file"""
    with open(filename, 'a', encoding='utf-8') as file:
        file.write(data + '\n')

def remove_think_tags(text: str) -> str:
    """Remove all <think> ... </think> blocks"""
    return re.sub(r'<think>(.*?)</think>', '', text, flags=re.DOTALL).strip()

def build_prompt(context: str, question: str) -> str:
    prompt = (
        f"Answer the question based on the given passages. The following are the passages:\n"
        f"{context}\n"
        f"Answer the question based on the given passages.\n"
        f"Question: {question}.\n"
        f"Answer:\n"
        f"Please first provide your answer in the format of Answer:[Your answer]. Then provide your reasoning process step-by-step.(Only include explicit clues) "
        f"At the end of each reasoning step, include a new line that specifies the key information or reference content used in that step. "
        f"Please ensure that the [reference content] you include is the complete original sentence or consecutive sentences from the text. Please do not change the punctuation.  Do not use ellipses inside the sentence. "
        f"Follow this format:\n"
        f"Answer: [Your answer]\n"
        f"Step-by-step Reasoning:\n"
        f"1. [Reasoning step 1]\n"
        f"[replaced by your reference content]\n"
        f"2. [Reasoning step 2]\n"
        f"[replaced by your reference content]\n"
    )
    return prompt

def extract_final_bullet_passage(answer_text: str):
    reasoning_pattern = r"Step-by-step Reasoning:\s*(.*)"
    reasoning_match = re.search(reasoning_pattern, answer_text, flags=re.DOTALL)
    if not reasoning_match:
        return None, None

    reasoning_text = reasoning_match.group(1).strip()
    bullet_pattern = r"(?m)^(\d+\.\s.*?)(?=(?:\n\d+\.\s)|\Z)"
    bullets = re.findall(bullet_pattern, reasoning_text, flags=re.DOTALL)
    if not bullets:
        print("No bullet blocks found.")
        return None, None

    passage_pattern = re.compile(
        r'(?i)(?:\*\*)?passage\s+(\d+)(?:\*\*)?\s*:\s*("([^"]*)"|(.+?))(?=\Z|\n\s*\n|$)',
        flags=re.DOTALL
    )
    
    for bullet in reversed(bullets):
        matches = passage_pattern.findall(bullet)
        if matches:
            last_match = matches[-1]
            passage_number = last_match[0]
            quoted_snippet = last_match[2]
            non_quoted_snippet = last_match[3]
            snippet = non_quoted_snippet.strip() if non_quoted_snippet.strip() else quoted_snippet.strip()
            return passage_number, snippet

    return None, None

def extract_all_bullet_passages(answer_text: str):
    reasoning_pattern = r"Step-by-step Reasoning:\s*(.*)"
    reasoning_match = re.search(reasoning_pattern, answer_text, flags=re.DOTALL)
    if not reasoning_match:
        return []

    reasoning_text = reasoning_match.group(1).strip()
    bullet_pattern = re.compile(r"^(\d+\.\s.*?)(?=^\d+\.\s|\Z)", re.MULTILINE | re.DOTALL)
    bullets = bullet_pattern.findall(reasoning_text)
    if not bullets:
        return []

    results = []
    for bullet_index, bullet_text in enumerate(bullets, start=1):
        results.append({
            'bullet_index': bullet_index,
            'snippet': bullet_text.strip()
        })
    print(results)
    return results

def extract_evidence(answer_text: str):
    reasoning_pattern = r"(?i)Evidence\s*(.*)"
    reasoning_match = re.search(reasoning_pattern, answer_text, flags=re.DOTALL)
    if not reasoning_match:
        return []

    reasoning_text = reasoning_match.group(1).strip()

    # Extract all bullet segments
    bullet_pattern = re.compile(r"^(\d+\.\s.*?)(?=^\d+\.\s|\Z)", re.MULTILINE | re.DOTALL)
    bullets = bullet_pattern.findall(reasoning_text)
    if not bullets:
        return []

    # Find the index of the first bullet starting with 1.
    start_index = -1
    for i, bullet in enumerate(bullets):
        if bullet.strip().startswith("1."):
            start_index = i
            break

    if start_index == -1:
        return []  # No valid starting bullet

    # Only keep the part starting from the first valid bullet
    bullets = bullets[start_index:]

    results = []
    for bullet_index, bullet_text in enumerate(bullets, start=1):
        results.append({
            'bullet_index': bullet_index,
            'snippet': bullet_text.strip()
        })
    return results


def get_answer_with_retry(model, prompt, retries=3, delay=5):
    """Call the model to get the answer based on the prompt, with retry on failure."""
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached, skipping this request.")
                return None

def extract_json_from_gpt_response(text: str) -> dict | None:
    """
    Finds the first JSON block inside ```json ... ``` or ``` … ``` and returns it as a dict.
    """
    # Try to find a ```json … ``` block first
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if not m:
        # Fallback: any ``` … ``` block that looks like JSON
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if not m:
        # Lastly, maybe the model just spit raw JSON without fences
        m = re.search(r"(\{.*?\})", text, flags=re.DOTALL)
    if not m:
        return None

    json_str = m.group(1)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # clean up trailing commas, etc.
        cleaned = re.sub(r",\s*([\]}])", r"\1", json_str)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

async def random_alternative_answer(
    question: str,
    original_context: str,
    unique_sents: List[str],
    correct_answer: str
) -> dict:
    """Generate random alternative answer and modified evidence"""
    
    # Construct GPT-4o prompt
    numbered = "\n\n".join(f"{j+1}. {s}" for j, s in enumerate(unique_sents))
    prompt = (
        "You are a creative assistant. Given the question below and the original answer, propose a plausible alternative answer that is **different** from the original but still reasonable. "
        "Then rewrite the provided sentences to support your alternative answer. When rewriting each sentence, modify only the parts necessary to support the alternative answer. "
        "Parts unrelated to the answer must keep their original meaning. Be sure that the modified evidence sentences are sufficient to answer the original question. "
        "Output must be strictly in the specified JSON format, with no additional text.\n"
        '{\n'
        '  "answer": "<your alternative answer here, just provide the answer phrase, no need for complete sentence>",\n'
        '  "revised": [\n'
        '    "<rewritten sentence 1>",\n'
        '    "<rewritten sentence 2>",\n'
        '    ...\n'
        '  ]\n'
        '}\n\n'
        f"Question:\n{question}\n\n"
        f"Original answer:\n{correct_answer}\n\n"
        f"Sentences to rewrite:\n{numbered}"
    )
    
    print(f"[Alternative Answer] Generating prompt: {prompt}")
    
    rsp = client.chat.completions.create(
        model="gpt-4o", temperature=0.7,
        messages=[{"role":"user","content":prompt}]
    )
    
    js = extract_json_from_gpt_response(rsp.choices[0].message.content)
    if not js:
        print("[Alternative Answer] Failed to parse JSON")
        return {"context": original_context, "answer": "Failed to generate alternative"}
        
    revised = js["revised"]     # List[str]
    alternative = js["answer"]  # Alternative answer
    
    # Create new context
    new_ctx = original_context
    for old, new in zip(unique_sents, revised):
        new_ctx = new_ctx.replace(old, new)
    
    return {"context": new_ctx, "answer": alternative}

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LastingBench random alternative answer generation")
    parser.add_argument("--output", "-o", type=str, default="output_random.jsonl", 
                       help="Output JSONL file path (default: output_random.jsonl)")
    parser.add_argument("--dataset_repo", type=str, default="THUDM/LongBench",
                       help="Dataset repository name (default: THUDM/LongBench)")
    parser.add_argument("--dataset_subset", type=str, default="hotpotqa",
                       help="Dataset subset name (default: hotpotqa)")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split (default: test)")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Starting index for processing (default: 0)")
    parser.add_argument("--max_samples", type=int, default=-1,
                       help="Maximum number of samples to process (-1 for all, default: -1)")
    
    args = parser.parse_args()
    
    out_file = args.output
    # Load dataset
    longbench = load_dataset(args.dataset_repo, args.dataset_subset)[args.split]
    
    print(f"Output file: {out_file}")
    print(f"Dataset: {args.dataset_repo}/{args.dataset_subset}[{args.split}]")
    print(f"Total samples: {len(longbench)}")
    
    count = 0
    
    # Determine processing range
    start_idx = args.start_idx
    end_idx = len(longbench) if args.max_samples == -1 else min(start_idx + args.max_samples, len(longbench))
    
    print(f"Processing samples from index {start_idx} to {end_idx-1}")
    
    for idx in range(start_idx, end_idx):
        example = longbench[idx]
        question = example['input']
        print(f"Question: {question}")
        context = example['context']
        correct_answer = example['answers'][0]

        print(f"Processing example {idx + 1}:")
        print(f"Correct Answer: {correct_answer}")

        # Build prompts
        prompt_with_context = build_prompt(context, question)

        # Get answers using transformers pipelines
        answer_with_context = get_answer_with_retry('deepseek-r1', prompt_with_context) 
        
        # Extract content after "Answer:" from answer_with_context
        answer_with_context_simple = (
            answer_with_context
            .split("Answer:", 1)[-1]          # First keep the part after Answer:
            .split("Step-by-step Reasoning", 1)[0]  # Then cut before Step-by-step Reasoning
            .strip()
        )
        
        print(f"Answer with context: {answer_with_context_simple}") 
        result = judge_answer_with_api(question, correct_answer, answer_with_context_simple)
        print(f"Answer judge result: {result}")
        
        if not result:
            continue

        answer_with_context = remove_think_tags(answer_with_context or "")
        evidence = extract_all_bullet_passages(answer_with_context)

        page_contents = []
        if evidence:
            count += 1
            for ev in evidence:
                snippet = ev['snippet']
                result = retriveDoc(context, snippet)
                # result["context"] is a set of Document objects
                page_contents += [doc.page_content for doc in result]
            
            unique_page_contents = list(dict.fromkeys(page_contents))
            aggregated_content = "\n".join(unique_page_contents)
            
            prompt_final = (
                f"Please answer the question based on the context.\nContext: {aggregated_content}.\n Question: {question}.\n"
                f"Please only provide your answer. "
                f"Your Answer:"
            )
            
            final_answer = get_transformers_answer(prompt_final, tokenizer1, model1)
            
            if judge_answer_with_api(question, correct_answer, final_answer):
                print("correct")
            else:
                print("incorrect")
                result_query = retriveDoc(context, question)
                page_contents += [doc.page_content for doc in result_query]
                
            unique_page_contents = list(dict.fromkeys(page_contents))
            
            # Generate random alternative answer instead of selecting the highest ppl answer
            alternative = asyncio.run(
                random_alternative_answer(
                    question,
                    context,
                    unique_page_contents,
                    correct_answer
                )
            )
            
            record = {
                "question": question,
                "answer": alternative["answer"],
                "context": alternative["context"]
            }

            # Append one line of JSON each loop
            with open(out_file, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main() 