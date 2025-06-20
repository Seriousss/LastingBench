import time
import os
import argparse
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI # Added for GPT-4o rephrasing
from utils.metrics import qa_f1_score, qa_em_score


THINK_END_ID = 151668 # </think> token ID for Qwen models (like Qwen1.5/Qwen2)

# --- OpenAI Client for Rephrasing ---
openai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"), 
    base_url=os.environ.get("OPENAI_BASE_URL")
)

def get_openai_rephrase_response(prompt, model="gpt-4o", retries=3, delay=2):
    """Call OpenAI API for rephrasing."""
    for attempt in range(retries):
        try:
            completion = openai_client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=100
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI Rephrase attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                print(f"Retrying OpenAI rephrase in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries for OpenAI rephrase reached.")
                return "Failed to rephrase question"

def rephrase_question_with_gpt4o(question, rephrase_type="opposite"):
    if rephrase_type == "opposite":
        prompt = f"""Please rephrase the following question to have the exact opposite meaning. 
Question: {question}

Return only the rephrased question with the opposite meaning, without any explanations or other content."""
    elif rephrase_type == "similar":
        prompt = f"""Please rephrase the following question to be synonymous, maintaining the original meaning but using different wording:
Question: {question}

Return only the rephrased question, without any explanations or other content."""
    else:
        raise ValueError(f"Invalid rephrase_type: {rephrase_type}. Must be 'opposite' or 'similar'.")
    
    return get_openai_rephrase_response(prompt)

# --- Qwen3-Specific Hugging Face Model Functions (for Answering) ---
def get_qwen3_hf_response(prompt_text, model, tokenizer, device, max_new_tokens=40, retries=2, delay=5):
    """Generate a response from a Qwen3-like HF model. max_new_tokens default to 30."""
    for attempt in range(retries):
        try:
            messages = [{"role": "user", 'content': prompt_text}]
            
            chat_template_args = {
                "tokenize": False,
                "add_generation_prompt": True
            }
            # Qwen models (like Qwen1.5, Qwen2) often use/support enable_thinking
            # Check if tokenizer's apply_chat_template supports 'enable_thinking'
            # This check is simplified; for robust production, inspect.signature might be better
            # but for Qwen-specific, we assume it or it gracefully ignores.
            try:
                # Attempt to use enable_thinking=False for Qwen models
                processed_prompt = tokenizer.apply_chat_template(
                    messages, **chat_template_args, enable_thinking=False
                )
            except TypeError:
                # Fallback if enable_thinking is not a valid kwarg for the specific tokenizer version
                print("Warning: Tokenizer does not support 'enable_thinking' in apply_chat_template. Proceeding without it.")
                processed_prompt = tokenizer.apply_chat_template(messages, **chat_template_args)
            except Exception as e:
                print(f"Warning: Error applying chat template: {e}. Using raw prompt.")
                processed_prompt = prompt_text # Fallback to raw prompt

            inputs = tokenizer(processed_prompt, return_tensors="pt", padding=True, truncation=True).to(device)
            
            generated_ids_full = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id
            )
            # Get only newly generated tokens
            output_only_ids_list = generated_ids_full[0][inputs.input_ids.shape[1]:].tolist()
            
            # Strip <think>...</think> tags specifically for Qwen
            try:
                # Find the last occurrence of THINK_END_ID and take tokens after it
                cut_index = len(output_only_ids_list) - output_only_ids_list[::-1].index(THINK_END_ID) 
                final_ids_to_decode = output_only_ids_list[cut_index:]
            except ValueError:
                # THINK_END_ID not found, use all generated new tokens
                final_ids_to_decode = output_only_ids_list
            
            response = tokenizer.decode(final_ids_to_decode, skip_special_tokens=True).strip()
            return response
        except Exception as e:
            print(f"Qwen HF Model generation attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries for Qwen HF model reached. Skipping this request.")
                return "Failed to get Qwen HF response"

def answer_question_with_context_qwen3_hf(question, context, model, tokenizer, device):
    """Answer a question with context using a Qwen3-like HF model."""
    prompt = f"""Please answer the question based on the following context:

Context:
{context}

Question: {question}

Only output the answer, no any other text. If the answer is not in the context, please say "I don't know".

Answer:"""
    return get_qwen3_hf_response(prompt, model, tokenizer, device)

def main(args):
    hf_device_setting = "auto"
    print(f"Attempting to use device: {hf_device_setting} for Qwen HF model.")

    print(f"Loading Qwen HF model for Answering: {args.model_name}...")
    hf_model = None
    hf_tokenizer = None
    try:
        hf_tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code_hf)
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, 
            device_map=hf_device_setting, 
            trust_remote_code=args.trust_remote_code_hf,
            torch_dtype="bfloat16" 
        )
        hf_model.eval()



        print(f"Successfully loaded Qwen HF model {args.model_name}.")
    except Exception as e:
        print(f"Failed to load Qwen HF model {args.model_name}: {e}")
        return

    print(f"Loading dataset {args.dataset_name}, subset {args.dataset_subset}...")
    try:
        dataset = load_dataset(args.dataset_name, args.dataset_subset)["test"]
        print(f"Successfully loaded dataset with {len(dataset)} samples.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    em_match_count = 0 # Counter for EM matches
    em_match_original_count = 0 # Counter for EM matches
    successfully_processed_samples = 0 # Counter for successfully processed samples

    num_samples_to_process = len(dataset) if args.sample_count == -1 else min(args.sample_count, len(dataset))
    
    print(f"Processing {num_samples_to_process} samples. Rephrasing with GPT-4o (opposite meaning). Answering with Qwen HF model {args.model_name} (max 30 tokens)...")

    for i in tqdm(range(num_samples_to_process), desc="Processing samples"):
        example = dataset[i]
        original_question = example['input']
        context = example['context']
        ground_truth_answers = example['answers']
        print(original_question)
        
        rephrased_question = rephrase_question_with_gpt4o(original_question, args.rephrase_type)
        print(rephrased_question)
        
        if rephrased_question == "Failed to rephrase question":
            print(f"Skipping sample {i+1} due to rephrasing failure.")
            continue
            
        rephrased_answer = answer_question_with_context_qwen3_hf(rephrased_question, context, hf_model, hf_tokenizer, hf_model.device)
        print(rephrased_answer)
        original_answer = answer_question_with_context_qwen3_hf(original_question, context, hf_model, hf_tokenizer, hf_model.device)
        if not ground_truth_answers:
            print(f"Skipping sample {i+1} due to missing ground truth answers.")
            continue
        print(original_answer)
        successfully_processed_samples += 1
        
        sample_had_em_match = False
        for gt_ans in ground_truth_answers:
            em = qa_em_score(rephrased_answer, gt_ans)
            if em > 0: # Check for exact match (assuming qa_em_score returns 1.0 for EM)
                sample_had_em_match = True
                break
        
        if sample_had_em_match:
            em_match_count += 1
        
        sample_had_em_match = False
        for gt_ans in ground_truth_answers:
            em = qa_em_score(original_answer, gt_ans)
            if em > 0: # Check for exact match (assuming qa_em_score returns 1.0 for EM)
                sample_had_em_match = True
                break
        if sample_had_em_match:
            em_match_original_count += 1

    if successfully_processed_samples > 0:
        print(f"\n--- Evaluation Summary ---")
        print(f"Answering Qwen HF Model: {args.model_name}")
        print(f"Dataset: {args.dataset_name} ({args.dataset_subset})")
        print(f"Successfully Processed Samples for Evaluation: {successfully_processed_samples}")
        print(f"Count of EM with original ground truth (after rephrase): {em_match_count}")
        print(f"Count of EM with original ground truth (before rephrase): {em_match_original_count}")
    else:
        print("\nNo samples were processed adequately to provide an evaluation summary.")
    
    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rephrase with GPT-4o, Answer with local Qwen3-like HF Model, then Evaluate.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-7B-Chat", help="Name of the Qwen3-like Hugging Face model for Answering.")
    parser.add_argument("--trust_remote_code_hf", action="store_true", default=True, help="Set to true if the Hugging Face model requires remote code (default: True for Qwen). Argument is present for explicitness but defaults to True.")
    parser.add_argument("--dataset_name", type=str, default="THUDM/LongBench", help="Name of the Hugging Face dataset.")
    parser.add_argument("--dataset_subset", type=str, default="2wikimqa", help="Subset of the dataset.")
    parser.add_argument("--sample_count", type=int, default=5, help="Number of samples to process. -1 for all. Default: 5.")
    parser.add_argument("--rephrase_type", type=str, default="opposite", choices=["opposite", "similar"], help="Type of rephrasing: 'opposite' for opposite meaning or 'similar' for similar meaning.")
    
    args = parser.parse_args()
    if openai_client.api_key == "your_api_key_here":
        print("CRITICAL ERROR: Please replace 'your_api_key_here' with your actual OpenAI API key in the script.")
    else:
        main(args)
