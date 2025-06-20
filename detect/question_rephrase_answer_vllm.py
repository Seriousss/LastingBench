import time
import os
import argparse
# import torch # torch might not be directly needed if vLLM handles all device aspects
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI # For GPT-4o rephrasing
from vllm import LLM, SamplingParams # For vLLM inference
from transformers import AutoTokenizer # Import AutoTokenizer
from utils.metrics import qa_f1_score, qa_em_score

# This will be respected by vLLM if CUDA_VISIBLE_DEVICES is set before vLLM import
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2" # User can set this outside the script

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
                max_tokens=100 # Max tokens for rephrased question
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
    """Rephrase a question using GPT-4o (English prompt)."""
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

# --- vLLM Model Functions (for Answering) ---
def get_vllm_response(prompt_text, llm_instance, sampling_params_instance, retries=2, delay=5):
    """Generate a response from a vLLM instance."""
    for attempt in range(retries):
        try:
            # vLLM generate method expects a list of prompts
            outputs = llm_instance.generate([prompt_text], sampling_params_instance)
            # For a single prompt, the result is in the first element of the output list
            # Each output object has a list of `outputs` (for n>1 in SamplingParams)
            response = outputs[0].outputs[0].text.strip()
            return response
        except Exception as e:
            print(f"vLLM generation attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                print(f"Retrying vLLM generation in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries for vLLM generation reached.")
                return "Failed to get vLLM response"

def answer_question_with_context_vllm(question, context, llm_instance, sampling_params_instance, tokenizer):
    """Answer a question with context using a vLLM model and chat template (English prompt)."""
    # Construct prompt using chat template, similar to evaluation.py
    prompt_content = (
        f"Answer the question based on the given passages. "
        "Only give me your answer and do not output any other words.\\n"
        "The following are given passages:\\n"
        f"{context}\\n"
        "Please strictly follow the context. "
        f"Question: {question}\\n"
        "Answer:"
    )
    messages = [{"role": "user", "content": prompt_content}]
    
    # Apply chat template
    # Note: Some tokenizers might not have a chat template configured, or might have different ways to apply it.
    # This is a common way for many models.
    try:
        final_prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        print(f"Failed to apply chat template: {e}. Falling back to basic prompt string.")
        # Fallback to a simpler prompt if template application fails
        final_prompt_text = f"Context:\\n{context}\\n\\nQuestion: {question}\\n\\nAnswer:"

    return get_vllm_response(final_prompt_text, llm_instance, sampling_params_instance)

def main(args):
    # Load Tokenizer for the vLLM model
    print(f"Loading tokenizer for model: {args.model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
        print("Successfully loaded tokenizer.")
    except Exception as e:
        print(f"Failed to load tokenizer for {args.model_name}: {e}")
        print("Please ensure the model name is correct and the tokenizer can be loaded.")
        return

    # Load vLLM Model (for Answering)
    print(f"Loading vLLM model for Answering: {args.model_name}...")
    print(f"(This may take a while depending on the model size and download speed if not cached).")
    vllm_model = None
    try:
        # You can expose more vLLM LLM parameters as args if needed
        # (e.g., tensor_parallel_size, dtype, gpu_memory_utilization)
        vllm_model = LLM(
            model=args.model_name, 
            trust_remote_code=args.trust_remote_code,
            dtype="bfloat16", # Use dtype from command line arguments
            # Add other vLLM LLM constructor arguments here if needed, e.g.:
            tensor_parallel_size=2
        )
        print(f"Successfully loaded vLLM model {args.model_name} with dtype='{args.dtype}' and tensor_parallel_size={args.tensor_parallel_size}.")
    except Exception as e:
        print(f"Failed to load vLLM model {args.model_name}: {e}")
        print("Please ensure vLLM is installed correctly and the model identifier is valid.")
        return

    # Define Sampling Parameters for vLLM
    # max_tokens is equivalent to max_new_tokens in HF
    # temperature=0.0 for greedy decoding, good for QA tasks for more deterministic output.
    # Adjust temperature (e.g., 0.7) and top_p (e.g., 0.95) for more diverse outputs if needed.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=30) # Set temperature to 0.0 for deterministic QA

    # Load dataset
    print(f"Loading dataset {args.dataset_name}, subset {args.dataset_subset}...")
    try:
        dataset = load_dataset(args.dataset_name, args.dataset_subset)["test"]
        print(f"Successfully loaded dataset with {len(dataset)} samples.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    em_match_count = 0  # Counter for EM matches
    em_match_original_count = 0 # Counter for EM matches
    successfully_processed_samples = 0 # Counter for successfully processed samples

    num_samples_to_process = len(dataset) if args.sample_count == -1 else min(args.sample_count, len(dataset))
    
    print(f"Processing {num_samples_to_process} samples. Rephrasing with GPT-4o (opposite meaning). Answering with vLLM model {args.model_name} (max 30 tokens)...")

    for i in tqdm(range(num_samples_to_process), desc="Processing samples with vLLM"):
        example = dataset[i]
        original_question = example['input']
        context = example['context']
        ground_truth_answers = example['answers']
        
        rephrased_question = rephrase_question_with_gpt4o(original_question, args.rephrase_type) # Use new rephrasing
        
        if rephrased_question == "Failed to rephrase question":
            print(f"Skipping sample {i+1} due to rephrasing failure.")
            continue
            
        rephrased_answer = answer_question_with_context_vllm(rephrased_question, context, vllm_model, sampling_params, tokenizer)
        # print(f"Rephrased question: {rephrased_question}") # Optional: for debugging
        # print(f"Answer to rephrased: {rephrased_answer}") # Optional: for debugging

        original_answer = answer_question_with_context_vllm(original_question, context, vllm_model, sampling_params, tokenizer)
        # print(f"Original question: {original_question}") # Optional: for debugging
        # print(f"Answer to original: {original_answer}") # Optional: for debugging
        
        if not ground_truth_answers:
            print(f"Skipping sample {i+1} due to missing ground truth answers.")
            continue
        print(original_answer)
        successfully_processed_samples += 1
        
        sample_had_em_match = False


        

        em_match_count += qa_em_score(rephrased_answer, ground_truth_answers[0])
        
        sample_had_em_match = False


        print(original_answer)
        print(ground_truth_answers[0])

        em_match_original_count += qa_em_score(original_answer, ground_truth_answers[0])

    if successfully_processed_samples > 0:
        print(f"Answering vLLM Model: {args.model_name}")
        print(f"Dataset             : {args.dataset_name} ({args.dataset_subset})")
        print(f"Successfully Processed Samples for Evaluation: {successfully_processed_samples}")
        print(f"Max Answer Tokens   : 30") # Reflects SamplingParams
        print(f"Count of EM with original ground truth (after rephrase): {em_match_count}")
        print(f"Count of EM with original ground truth (before rephrase): {em_match_original_count}")
    else:
        print("\nNo samples were processed adequately to provide an evaluation summary.")
    
    print("vLLM processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rephrase with GPT-4o, Answer with local vLLM-hosted Model, then Evaluate.")
    parser.add_argument("--model_name", type=str, default="facebook/opt-125m", help="Name/path of the Hugging Face model for Answering via vLLM (e.g., 'mistralai/Mistral-7B-Instruct-v0.1').")
    parser.add_argument("--dataset_name", type=str, default="THUDM/LongBench", help="Name of the Hugging Face dataset.")
    parser.add_argument("--dataset_subset", type=str, default="2wikimqa", help="Subset of the dataset.")
    parser.add_argument("--sample_count", type=int, default=3, help="Number of samples to process. -1 for all. Default: 3 for quick testing.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Set to true if the Hugging Face model for vLLM requires remote code.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM.")
    parser.add_argument("--dtype", type=str, default="auto", help="Data type for the model. Examples: 'auto', 'half', 'float16', 'bfloat16', 'float', 'float32'. Default is 'auto'.")
    parser.add_argument("--rephrase_type", type=str, default="opposite", choices=["opposite", "similar"], help="Type of rephrasing: 'opposite' for opposite meaning or 'similar' for similar meaning.")
    
    args = parser.parse_args()
    main(args) 