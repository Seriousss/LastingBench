import json
import time
import os
import argparse
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from utils.metrics import qa_f1_score, qa_em_score # Import evaluation functions

# Configure OpenAI API
client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY"), 
  base_url=os.environ.get("OPENAI_BASE_URL")
)

def get_openai_response(prompt, model="gpt-4o", retries=3, delay=2):
    """Call OpenAI API to get response with retry mechanism"""
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=100
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Skipping this request.")
                return "Failed to get response"

def rephrase_question_api(question, model_name, rephrase_type="opposite"):
    """Use OpenAI API to rephrase question (English prompt)"""
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
    
    return get_openai_response(prompt, model=model_name)

def answer_question_with_context_api(question, context, model_name, max_tokens_for_answer=30):
    """Use OpenAI API to answer question based on context (English prompt)"""
    prompt = f"""Please answer the question based on the following context:

Context:
{context}

Question: {question}

Only output the answer, no any other text. If the answer is not in the context, please say "I don't know".

Answer:"""
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=max_tokens_for_answer
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Answer generation failed for model {model_name}: {e}")
        return "Failed to get answer"

def main(args):
    # Load dataset
    print(f"Loading dataset {args.dataset_name}, subset {args.dataset_subset}...")
    try:
        dataset = load_dataset(args.dataset_name, args.dataset_subset)["test"]
        print(f"Successfully loaded dataset with {len(dataset)} samples.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    em_match_count = 0  # Counter for EM matches
    successfully_processed_samples = 0 # Counter for successfully processed samples

    num_samples_to_process = len(dataset) if args.sample_count == -1 else min(args.sample_count, len(dataset))
    
    print(f"Processing {num_samples_to_process} samples. Rephrasing with GPT-4o (opposite meaning). Answering with {args.model_name} (max 30 tokens for answer)...")

    for i in tqdm(range(num_samples_to_process), desc="Processing samples"):
        example = dataset[i]
        original_question = example['input']
        context = example['context']
        ground_truth_answers = example['answers']

        print(f"Original question: {original_question}")
        
        # Use API to rephrase question, fixed using gpt-4o
        rephrased_question = rephrase_question_api(original_question, "gpt-4o", args.rephrase_type)
        print(f"Rephrased question (opposite): {rephrased_question}")
        
        if rephrased_question == "Failed to get response" or rephrased_question == "Failed to rephrase question": # Broader check
            print(f"Skipping sample {i+1} due to rephrasing failure.")
            continue
            
        # Use rephrased question and context to get answer, using args.model_name, answer length limited to 30 tokens
        rephrased_answer = answer_question_with_context_api(rephrased_question, context, args.model_name, max_tokens_for_answer=30)
        # print(f"Answer to rephrased question: {rephrased_answer}")

        if rephrased_answer == "Failed to get answer":
            print(f"Skipping sample {i+1} due to answer generation failure.")
            continue

        if not ground_truth_answers:
            print(f"Skipping sample {i+1} due to missing ground truth answers.")
            continue
        
        successfully_processed_samples += 1
        sample_had_em_match = False
        for gt_ans in ground_truth_answers:
            em = qa_em_score(rephrased_answer, gt_ans)
            if em > 0: # EM is 1.0 for a match
                sample_had_em_match = True
                break
        
        if sample_had_em_match:
            em_match_count += 1
        # print(f"Sample EM with original GT: {1 if sample_had_em_match else 0}")

    if successfully_processed_samples > 0:
        print(f"\n--- Evaluation Summary ---")
        print(f"Answering Model : {args.model_name}")
        print(f"Dataset         : {args.dataset_name} ({args.dataset_subset})")
        print(f"Successfully Processed Samples for Evaluation: {successfully_processed_samples}")
        print(f"Max Answer Tokens: 30")
        print(f"Count of EM with original ground truth (after rephrase): {em_match_count}")
    else:
        print("\nNo samples were processed adequately to provide an evaluation summary.")
    
    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rephrase questions to opposite meaning with GPT-4o, answer with specified OpenAI model, then count EM against original GT.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Name of the OpenAI model to use for Answering.")
    parser.add_argument("--dataset_name", type=str, default="THUDM/LongBench", help="Name of the Hugging Face dataset.")
    parser.add_argument("--dataset_subset", type=str, default="2wikimqa", help="Subset of the dataset.")
    parser.add_argument("--sample_count", type=int, default=-1, help="Number of samples to process. -1 for all samples.")
    parser.add_argument("--rephrase_type", type=str, default="opposite", choices=["opposite", "similar"], help="Type of rephrasing: 'opposite' for opposite meaning or 'similar' for similar meaning.")
    
    args = parser.parse_args()
    main(args) 