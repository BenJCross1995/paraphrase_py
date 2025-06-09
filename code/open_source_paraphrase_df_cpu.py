import argparse
import time
import math
import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import read_and_write_docs
import re
import json
import sys
from torch import nn
from functools import partial

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

document:

"""
    return system_prompt

def default_generation_system_prompt():
    system_prompt = """
    You are an expert re-writer. A *reference text* will be provided; study it only to understand the author’s voice, level of formality, pacing, and the information it conveys.  
Then compose an entirely new document that:

1. **Matches Style & Tone**  
   • Mirror the register, rhythm, and genre conventions (e.g. academic essay, news article, casual blog).  
   • Preserve the emotional tenor (serious, playful, persuasive, etc.).

2. **Maintains Conceptual Fidelity, Not Surface Similarity**  
   • Convey the same core ideas, arguments, and key facts.  
   • Feel free to reorder, reframe, condense, or expand points for natural flow.  
   • Avoid sentence-level paraphrase; do **not** echo more than four consecutive words from the source.

3. **Encourages Creative Re-expression**  
   • Introduce fresh transitions, examples, metaphors, or analogies where helpful.  
   • Use new phrasing rather than one-for-one synonym substitutions.

4. **Allows Structural Freedom**  
   • Reorganize paragraphs, combine or split ideas, adjust headings, or deploy different rhetorical devices.  
   • The result should read like an original piece written after performing the same research—not like a transformed clone.

5. **Protects Critical Details**  
   • Keep technical terms, proper nouns, data, citations, and direct quotations accurate and unaltered unless explicitly instructed.

6. **Output Format**  
   • Return **only** the freshly written document—no commentary—in JSON:  
     {"new_document": <your_document_here>}

Instructions:  
Work as a seasoned editor rewriting an article from scratch. Emphasize originality in wording and structure while safeguarding the source’s communicative intent.

reference text:

"""
    return system_prompt
    

def load_local_model(model_dir: str, device: str):
    if not os.path.isdir(model_dir):
        raise ValueError(f"Directory {model_dir} does not exist.")

    # Make things deterministic if you really need it
    torch.backends.cudnn.deterministic = True

    # Optional: speed-up with ORT if installed
    try:
        from torch_ort import ort_setup
        ort_setup()
    except ImportError:
        pass

    # Choose dtypes
    if device == "cpu":
        dtype = torch.bfloat16
    elif device.startswith("cuda"):
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare from_pretrained kwargs
    load_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype,
        "local_files_only": True,
        # for CPU offloading you can still use device_map="auto"
        "device_map": "auto" if device == "cpu" else None,
    }

    # Instantiate the model
    model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)

    # If this was a safetensors‐backed model, decompose() will unpack it
    if hasattr(model, "decompose"):
        model.decompose()

    # Move + eval
    model.to(device)
    model.eval()

    # Enable gradient checkpointing if supported
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Device‐specific extra tweaks
    if device == "cpu":
        # e.g. xFormers, Intel extensions, etc.
        if hasattr(model, "optimize"):
            model.optimize(optimization_type="speed")
    else:  # cuda
        model = model.half()  # ensure float16
        if hasattr(model, "optimize_for_inference"):
            model = model.optimize_for_inference()

    # Finally, compile if available
    if hasattr(torch, "compile"):
        compile_mode = "max-autotune" if device.startswith("cuda") else "default"
        model = torch.compile(model, mode=compile_mode)

    return tokenizer, model


def prepare_inputs(system_prompt: str, user_prompt: str, tokenizer, device):
    """
    Prepares tokenized inputs and attention mask from the system and user prompts.
    """
    start_time = time.time()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
	
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    inputs = encoded["input_ids"].to(device, non_blocking=True)
    attention_mask = encoded["attention_mask"].to(device, non_blocking=True)

    elapsed_time = time.time() - start_time
    print(f"Time taken for prepare_inputs: {elapsed_time:.4f} seconds")

    return inputs, attention_mask, input_text

def compute_effective_max_tokens(user_prompt: str, tokenizer, default_max: int, ratio: float = 0.2) -> int:
    """
    Computes max_new_tokens as the minimum of default_max and ratio * (number of tokens in user_prompt).
    """
    original_length = len(tokenizer.encode(user_prompt))
    computed = int(original_length * (1 + ratio))
    return min(default_max, computed) if computed > 0 else default_max

def batch_llm_call(inputs, attention_mask, tokenizer, model, n: int,
                   max_new_tokens: int, temperature: float, top_p: float):
    """
    Executes the generation call n times using the same inputs and returns a list of results.
    Each result contains the iteration, generated text, time taken, and tokens per second.
    """

	# Record the prompt length so that we can start from the next token
    prompt_length = inputs.shape[1]

    results = []

    for i in range(n):
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        num_tokens = outputs[0].shape[-1]
        tokens_per_sec = num_tokens / elapsed if elapsed > 0 else float('inf')
        generated_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
		
        results.append({
            "iteration": i + 1,
            "generated_text": generated_text,
            "time_sec": elapsed,
            "tokens_per_sec": tokens_per_sec
        })
        print(f"Iteration {i+1}: {elapsed:.2f} sec, {tokens_per_sec:.2f} tokens/sec")
    return results

def batch_llm_call(inputs, attention_mask, tokenizer, model, n: int, max_new_tokens: int,
				   temperature: float, top_p: float, batch_size: int = None):
    """
    Executes the generation call n times using the same inputs and returns a list of results.
    If batch_size is provided, the n responses are generated in batches.
    Each result contains the iteration, generated text, time taken, and tokens per second.
    """
    # Record the prompt length so that we can start from the next token
    prompt_length = inputs.shape[1]

    results = []

    # Use batch_size = n if not provided to generate all responses in one go
    batch_size = batch_size or n
    total_batches = math.ceil(n / batch_size)
    iteration = 0

    for b in range(total_batches):
        # Create a batch by repeating the inputs and attention_mask
        current_batch_size = min(batch_size, n - iteration)
        batch_inputs = inputs.repeat(current_batch_size, 1)
        batch_attention = attention_mask.repeat(current_batch_size, 1)

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                batch_inputs,
                attention_mask=batch_attention,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        end_time = time.perf_counter()
        elapsed = end_time - start_time

        # Process each output in the batch
        for i in range(current_batch_size):
            num_tokens = outputs[i].shape[-1]
            tokens_per_sec = num_tokens / elapsed if elapsed > 0 else float('inf')
            generated_text = tokenizer.decode(outputs[i][prompt_length:], skip_special_tokens=True)
            iteration += 1
            result = {
                "iteration": iteration,
                "generated_text": generated_text,
                "time_sec": elapsed / current_batch_size,  # approximate per-sample time
                "tokens_per_sec": tokens_per_sec
            }
            results.append(result)
            print(f"    Iteration {iteration}: {result['time_sec']:.2f} sec, {tokens_per_sec:.2f} tokens/sec")
    
    return results

def sequential_unique_llm_call(
    inputs,
    attention_mask,
    tokenizer,
    model,
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float
):
    """
    Repeatedly samples from the model (one at a time) until we have `n` distinct
    generations.  Returns a list of dicts with keys:
      - iteration:      1-based count of unique outputs so far
      - generated_text: the text of this new unique sample
      - time_sec:       time taken for this single generate() call
      - tokens_per_sec: tokens generated / time_sec

    Duplicate texts are skipped (not counted toward `n`), and we keep sampling
    until `n` unique generations have been seen.
    """
    # how many tokens in the prompt so we can strip them off
    prompt_length = int(attention_mask[0].sum().item())

    seen_texts = set()
    results = []
    unique_count = 0

    # we assume inputs is a single prompt, i.e. batch_size==1
    while unique_count < n:
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        elapsed = time.perf_counter() - start

        # strip off the prompt tokens, decode new tokens
        gen = tokenizer.decode(out[0][prompt_length:], skip_special_tokens=True).strip()

        if gen in seen_texts:
            continue

        seen_texts.add(gen)
        unique_count += 1

        total_tokens = out[0].size(-1)
        tps = total_tokens / elapsed if elapsed > 0 else float("inf")

        result = {
            "iteration": unique_count,
            "generated_text": gen,
            "time_sec": elapsed,
            "tokens_per_sec": tps
        }
        results.append(result)
        print(f"Iteration {unique_count}: {elapsed:.2f}s, {tps:.2f} tok/s")

    return results
	
def main():
    parser = argparse.ArgumentParser(
        description="Optimized LLM Inference using a local model directory and JSONL input/output."
    )
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the local model directory.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input JSONL file (with a 'text' field).")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save the output JSONL file.")
    parser.add_argument("--system_prompt_loc", type=str, default="",
                        help="System prompt for the conversation. If not provided, a default prompt is used.")
    parser.add_argument("--n", type=int, default=1,
                        help="Number of iterations to run the generation.")
    parser.add_argument("--max_new_tokens", type=int, default=5000,
                        help="Default maximum number of new tokens to generate (overridden if computed value is lower).")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperature for generation.")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling probability.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Optional batch size for parallel generation. If not provided, either n is used or a safe default.")
    parser.add_argument("--num_threads", type=int, default=None,
						help="Optional: number of CPU threads to use for PyTorch operations.")
    parser.add_argument("--batch", action="store_true",
                        help="Use batch mode (batch_llm_call) instead of sequential sampling.")
    parser.add_argument("--generation_type", type=str, default="paraphrase", help="Whether to use the paraphrase or generate system prompt")
    args = parser.parse_args()

    # If the file exists then exit before compiling any code.
    if os.path.exists(args.output_file):
        print(f"Output file '{args.output_file}' already exists, quitting.")
        sys.exit(0)
        
    process_start_time = time.time()
    
    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"
		
    print(f"Using Device: {device}")

    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(args.num_threads)
		
    if args.system_prompt_loc and args.system_prompt_loc.strip():
        system_prompt_loc = args.system_prompt_loc.strip()
        
        with open(system_prompt_loc, "r", encoding="utf-8") as f:

            system_prompt = f.read()  
    else:
        system_prompt = default_system_prompt()

    # Load the input JSONL file using the external module.
    df = read_and_write_docs.read_jsonl(args.input_file)

    if "text" not in df.columns:
        raise ValueError("The input JSONL must have a 'text' column.")

	# The dataframe is duplicated and we only want to tokenise the first row
    user_prompt = df.loc[0, "text"]

    # Load the model and output the time taken
    model_load_start = time.time()
    tokenizer, model = load_local_model(args.model_dir, device)
    model_load_time = round(time.time() - model_load_start, 2)
    print(f"Model Loaded: Time Taken {model_load_time}")

    # Want to limit the max new tokens to certain ratio above original text
    effective_max_tokens = compute_effective_max_tokens(user_prompt, tokenizer, args.max_new_tokens, ratio=0.2)
    print(f"Using max_new_tokens = {effective_max_tokens} based on user prompt length.")

    inputs, attention_mask, input_text = prepare_inputs(system_prompt, user_prompt, tokenizer, device)
	
    print("Beginning paraphrasing...")
    # Choose generation mode
    if args.batch:
        results = batch_llm_call(
            inputs,
            attention_mask,
            tokenizer,
            model,
            n=args.n,
            max_new_tokens=effective_max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size
        )
    else:
        results = sequential_unique_llm_call(
            inputs,
            attention_mask,
            tokenizer,
            model,
            n=args.n,
            max_new_tokens=effective_max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
	
    # Replicate the original row details with each generated result.
    original_row = df.iloc[0].to_dict()
    expanded_rows = []
    for res in results:
        row = original_row.copy()

        row.update({
            "generated_text": res["generated_text"],
			"top_p": args.top_p,
			"temperature": args.temperature,
            "time_sec": res["time_sec"],
            "tokens_per_sec": res["tokens_per_sec"]
        })
        expanded_rows.append(row)
    output_df = pd.DataFrame(expanded_rows)
    
    # Ensure output directory exists.
    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(output_dir, exist_ok=True)
    
    # Write the final output using the external module.
    read_and_write_docs.write_jsonl(output_df, args.output_file)
    print(f"Saved {len(output_df)} rows to {args.output_file}")

    process_duration = time.time() - process_start_time
    print(f"Total time taken to process document: {process_duration}")

if __name__ == "__main__":
    main()
