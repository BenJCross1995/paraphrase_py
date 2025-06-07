import argparse
import time
import math
import os
import torch
import pandas as pd
import json
import sys
import vllm
from vllm import LLM, SamplingParams

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
       • Mirror the register, rhythm, and genre conventions (e.g.\ academic essay, news article, casual blog).  
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
    """
    Loads a model using vllm from a local directory.
    """
    if not os.path.isdir(model_dir):
        raise ValueError(f"Directory {model_dir} does not exist.")

    # Load the model using vllm
    model = LLM(
        model=model_dir,
        device=device,
        quantization="auto",
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    return model

def prepare_vllm_inputs(system_prompt: str, user_prompt: str):
    """
    Prepares inputs for vllm in the required format.
    """
    prompt = f"{system_prompt}{user_prompt}"

    return prompt

def compute_effective_max_tokens(user_prompt: str, tokenizer, default_max: int, ratio: float = 0.2) -> int:
    """
    Computes max_new_tokens as the minimum of default_max and ratio * (number of tokens in user_prompt).
    """
    original_length = len(tokenizer.encode(user_prompt))
    computed = int(original_length * (1 + ratio))
    return min(default_max, computed) if computed > 0 else default_max

def batch_vllm_call(llm, prompts, n: int, max_new_tokens: int, temperature: float, top_p: float, batch_size: int = None):
    """
    Executes the generation call using vllm in batch mode.
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens
    )

    results = []
    total_batches = math.ceil(n / batch_size) if batch_size else 1
    batch_size = batch_size or n

    for b in range(total_batches):
        batch_prompts = prompts[b*batch_size:(b+1)*batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)

        for i, output in enumerate(outputs):
            elapsed = output.finish_time - output.start_time
            generated_text = output.text
            tokens_per_sec = (len(output.token_ids) / elapsed) if elapsed > 0 else float('inf')

            results.append({
                "iteration": b*batch_size + i + 1,
                "generated_text": generated_text,
                "time_sec": elapsed,
                "tokens_per_sec": tokens_per_sec
            })

    return results

def sequential_unique_vllm_call(llm, prompt, n: int, max_new_tokens: int, temperature: float, top_p: float):
    """
    Repeatedly samples from the model until we have `n` distinct generations.
    """
    seen_texts = set()
    results = []
    unique_count = 0

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens
    )

    while unique_count < n:
        output = llm.generate([prompt], sampling_params)[0]
        generated_text = output.text.strip()

        if generated_text in seen_texts:
            continue

        seen_texts.add(generated_text)
        unique_count += 1

        elapsed = output.finish_time - output.start_time
        tokens_per_sec = (len(output.token_ids) / elapsed) if elapsed > 0 else float('inf')

        results.append({
            "iteration": unique_count,
            "generated_text": generated_text,
            "time_sec": elapsed,
            "tokens_per_sec": tokens_per_sec
        })

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
        
    if args.system_prompt_loc and args.system_prompt_loc.strip():
        system_prompt_loc = args.system_prompt_loc.strip()
        with open(system_prompt_loc, "r", encoding="utf-8") as f:
            system_prompt = f.read()  
    else:
        system_prompt = default_system_prompt()

    # Load the model using vllm
    model_load_start = time.time()
    llm = load_local_model(args.model_dir, device)
    model_load_time = round(time.time() - model_load_start, 2)
    print(f"Model Loaded: Time Taken {model_load_time}")

    # Load the input JSONL file
    df = pd.read_json(args.input_file, lines=True)
    if "text" not in df.columns:
        raise ValueError("The input JSONL must have a 'text' column.")
    user_prompt = df.loc[0, "text"]

    # Create prompts for vllm
    prompts = [prepare_vllm_inputs(system_prompt, user_prompt)]

    # Generation mode
    if args.batch:
        if not args.batch_size:
            args.batch_size = args.n
        results = batch_vllm_call(llm, prompts, args.n, args.max_new_tokens, args.temperature, args.top_p, args.batch_size)
    else:
        results = sequential_unique_vllm_call(llm, prompts[0], args.n, args.max_new_tokens, args.temperature, args.top_p)
    
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
    
    # Write the final output
    output_df.to_json(args.output_file, lines=True, orient='records')
    print(f"Saved {len(output_df)} rows to {args.output_file}")

    process_duration = time.time() - process_start_time
    print(f"Total time taken to process document: {process_duration}")

if __name__ == "__main__":
    main()

# cd \Users\benjc\Documents\environments
# .\para_llm\Scripts\activate
# cd \Users\benjc\Documents\GitHub\paraphrase_py
# python code/vllm_paraphrase_df.py --model_dir "C:\Users\benjc\Documents\local models\Qwen2.5-1.5B-Instruct"  --input_file "C:\Users\benjc\Documents\test datasets\142_196_88_228_text_1.jsonl" --output_file "C:\Users\benjc\Documents\test datasets\142_196_88_228_text_1_qwen2.5_1.5B_20250605_1.jsonl" --system_prompt_loc "C:\Users\benjc\Documents\GitHub\paraphrase_py\prompts\default_paraphrase_prompt.txt" --n 1 --temperature 0.7 --top_p 0.9
