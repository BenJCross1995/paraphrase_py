import argparse
import time
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def default_system_prompt():
    system_prompt = """
Your role is to function as an advanced paraphrasing assistant. Your task is to generate a fully paraphrased version of a given document that preserves its original meaning, tone, genre, and style, while exhibiting significantly heightened lexical diversity and structural transformation. The aim is to produce a document that reflects a broad, globally influenced language profile for authorship verification research.

Guidelines:

1. **Preserve Core Meaning & Intent:**  
   - Ensure that the paraphrased text maintains the original document’s logical flow, factual accuracy, and overall message.  
   - Retain the tone, style, and genre to match the source content precisely.

2. **Maximize Lexical Diversity:**  
   - Use an extensive range of synonyms, idiomatic expressions, and alternative phrasings to replace common expressions.  
   - Avoid repetitive language; introduce varied vocabulary throughout the document to ensure a fresh linguistic perspective.

3. **Transform Structural Elements:**  
   - Reorganize sentences and paragraphs: invert sentence structures, vary sentence lengths, and use different clause orders.  
   - Experiment with alternative grammatical constructions and narrative flows without compromising clarity or meaning.

4. **Preserve Critical Terms & Proper Nouns:**  
   - Do not alter technical terms, names, or key references unless explicitly instructed.  
   - Ensure these elements remain intact to maintain the document's integrity.

5. **Ensure Naturalness & Cohesion:**  
   - Despite extensive lexical and structural changes, the paraphrased document must remain coherent, natural, and easily understandable.  
   - Strive for a balanced output that is both distinct in language and faithful to the original content.

6. **Output Format:**  
   - Provide only the paraphrased document without any extra commentary or explanations.  
   - The output must be structured in JSON format as follows:  

     {"new_document": <paraphrased_document>}

Instructions:
- Prioritize high lexical variation and significant syntactic reordering.
- Create a paraphrase that is distinct in wording and structure from the source while fully retaining its meaning, tone, and intent.
"""
    return system_prompt

def load_local_model(model_dir: str, device: str):
    """
    Loads a model and its tokenizer from a local directory.
    
    Uses low_cpu_mem_usage to minimize peak RAM usage and sets the
    appropriate torch_dtype (float16 for GPU and float32 for CPU).
    Optionally compiles the model with torch.compile on CPU for speed.
    """
    # Verify that model_dir exists.
    if not os.path.isdir(model_dir):
        raise ValueError(f"Directory {model_dir} does not exist.")

    # Choose the proper dtype: float16 on GPU, float32 on CPU.
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Load the tokenizer from the local directory.
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    
    # Load the model using low_cpu_mem_usage to minimize peak memory usage.
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        local_files_only=True
    ).to(device)

    model.eval()  # Set model to evaluation mode.

    # If on CPU and torch.compile is available (PyTorch 2.0+), compile the model.
    if device == "cpu" and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")
    
    return tokenizer, model

def llm_call(system_prompt: str, user_prompt: str, tokenizer, model) -> str:
    """
    Generates a response based on system and user prompts using the provided model.
    Also prints generation time and tokens per second.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Format the conversation using the tokenizer's chat template.
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Tokenize the input with padding to produce an explicit attention_mask.
    encoded = tokenizer(input_text, return_tensors="pt", padding=True)
    inputs = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    
    # Start timer.
    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,  # explicit attention mask
            max_new_tokens=5000,
            temperature=0.2,
            top_p=0.9,
            do_sample=True
        )
    # End timer.
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    # Calculate tokens per second.
    num_tokens = outputs[0].shape[-1]
    tokens_per_sec = num_tokens / elapsed if elapsed > 0 else float('inf')
    print(f"Generation completed in {elapsed:.2f} seconds, {tokens_per_sec:.2f} tokens per second.")
    
    # Decode and return the output text.
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized LLM Inference using a local model directory.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the local model directory.")
    parser.add_argument("--system_prompt", type=str,
                        default="",
                        help="System prompt for the conversation. If not provided, a default detailed prompt will be used.")
    parser.add_argument("--user_prompt", type=str,
                        default="What is the capital of France?",
                        help="User prompt for the conversation.")
    args = parser.parse_args()

    # Automatically choose device: GPU if available, otherwise CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use the provided system prompt, or default to the detailed version.
    system_prompt = args.system_prompt if args.system_prompt.strip() else default_system_prompt()

    # Load the tokenizer and model from the specified local directory.
    tokenizer, model = load_local_model(args.model_dir, device)

    # Print the formatted input prompt.
    input_prompt = tokenizer.apply_chat_template([{"role": "user", "content": args.user_prompt}], tokenize=False)
    print("Input prompt:", input_prompt)

    # Generate and print the model's response.
    response = llm_call(system_prompt, args.user_prompt, tokenizer, model)
    print("Response:", response)
