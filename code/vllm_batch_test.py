#!/usr/bin/env python
import argparse
import time
from vllm import LLM, SamplingParams
from read_and_write_docs import read_jsonl, write_jsonl

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

Document: <<input>>
"""
    return system_prompt
    
def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Batch process texts with vllm.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input JSONL file location")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSONL file location")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Local model directory")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for processing")
    args = parser.parse_args()


    # Initialize the LLM engine with the given model.
    llm = LLM(model=args.model_dir,
              device="gpu",
              enable_prefix_caching=True,
              disable_sliding_window=True,
              disable_async_output_proc=True)

    sampling_params = SamplingParams(
        max_tokens=100,    # Adjust as needed.
        temperature=0.7,
        top_p=0.95
    )

    # Define a basic system prompt.
    system_prompt = default_system_prompt()

    # Read the input JSONL file.
    df = read_jsonl(args.input_file)
    prompts = [system_prompt.replace('<<input>>', t) for t in df['text']]

    start_time = time.time()
    outputs = llm.generate(prompt=full_prompt, sampling_params=sampling_params)
    end_time = time.time()
    elapsed_time = end_time = start_time
    df['new_document'] = [output.outputs[0].text for output in outputs]

    # Write out the updated dataframe.
    write_jsonl(df, args.output_file)

    # Print performance metrics.
    print("Batch processing complete.")
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))

if __name__ == "__main__":
    main()
