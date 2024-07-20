import argparse
import copy
import json
import os
import re
import read_and_write_docs

import pandas as pd

from llama_cpp import Llama, LlamaGrammar

def filter_dataframe_by_folder(df_path, folder_path):
    """Filters a DataFrame by removing rows with doc_ids that already exist as files in a specified folder.
    
    Args:
        df_path (str): Path to the JSONL file containing the DataFrame.
        folder_path (str): Path to the folder containing files named in the format "doc_{doc_id}.jsonl".
        
    Returns:
        pd.DataFrame: Filtered DataFrame with rows removed where doc_ids already exist in the folder.
    """
    # Read the dataframe from the provided location
    df = read_and_write_docs.read_jsonl_file(df_path)

    # List all files in the provided folder
    files_in_folder = os.listdir(folder_path)

    # Extract doc_ids from file names (assuming files are named as "doc_{doc_id}")
    existing_doc_ids = set()
    for filename in files_in_folder:
        if filename.startswith("doc_") and filename.endswith(".jsonl"):
            try:
                doc_id = int(filename[4:-6])  # 'doc_' is 4 characters and '.jsonl' is 6 characters
                existing_doc_ids.add(doc_id)
            except ValueError:
                continue  # Skip files that don't follow the naming pattern

    # Filter the dataframe to keep only rows where doc_id is not in existing_doc_ids
    filtered_df = df[~df['doc_id'].isin(existing_doc_ids)]
    
    # Calculate the number of unique doc_ids removed
    unique_doc_ids_removed = len(set(df['doc_id']) & existing_doc_ids)
    
    # Print the number of unique doc_ids filtered out if more than 1
    if unique_doc_ids_removed > 1:
        print(f'{unique_doc_ids_removed} unique doc_ids removed as already exist')

    return filtered_df

def check_starting_bracket(llm_output):
    """
    Ensures the string starts with '[' if it doesn't already.
    """
    if not llm_output.startswith('['):
        llm_output = '[' + llm_output
    return llm_output

def check_trailing_bracket(llm_output):
    """
    Ensures the string ends with ']' and handles incomplete entries by removing them.
    """
    llm_output = llm_output.strip()
    if not llm_output.endswith(']'):
        # Find the last comma
        last_comma_index = llm_output.rfind(',')
        if last_comma_index != -1:
            # Get the substring after the last comma
            after_last_comma = llm_output[last_comma_index + 1:].strip()
            # Check if the length of the substring after the last comma is less than 3 characters
            if len(after_last_comma) < 3:
                # Remove the substring after the last comma
                llm_output = llm_output[:last_comma_index].strip()
            # Remove any incomplete entries by checking for open quotes
            while after_last_comma.count('"') % 2 != 0:
                llm_output = llm_output[:last_comma_index].strip()
                last_comma_index = llm_output.rfind(',')
                if last_comma_index == -1:
                    break
                after_last_comma = llm_output[last_comma_index + 1:].strip()
        # Append the closing bracket
        llm_output += ']'

    return llm_output

def format_output(llm_output, error_loc):
    """
    Formats the LLM output to ensure it is a valid JSON list.
    Tries to parse it as a JSON list and returns the parsed list if successful.
    If parsing fails, attempts to replace single quotes with double quotes and try again.
    If parsing still fails, prints the result and returns an empty list.
    """

    result = check_starting_bracket(llm_output)
    result = check_trailing_bracket(result)
    try:
        # Attempt to parse the JSON string
        parsed_result = json.loads(result)
        return parsed_result
    except json.JSONDecodeError as e:
        # Print detailed error information
        print("JSONDecodeError:")
        read_and_write_docs.save_error_as_txt(result, error_loc)
        return []

def call_local_llm(llm, messages, sentence, grammar=None, temperature=0.7):

    # Create a copy of the messages list
    messages_copy = copy.deepcopy(messages)
    
    # Append the new message to the copied list
    new_message = {"role": "user", "content": sentence}
    messages_copy.append(new_message)

    response = llm.create_chat_completion(messages=messages_copy, grammar=grammar, temperature=temperature)
    
    # Extract the answer and finish reason
    answer = response['choices'][0]['message']['content']
    
    return answer
	
def paraphrase_llm(read_loc, write_loc, error_loc, llm, messages, grammar_path=None, temperature=0.7, num_iterations=10):
    """
    Paraphrases documents using an LLM and saves the results.

    Args:
        read_loc (str): File path to the input data.
        write_loc (str): Folder path to save the output data.
        error_loc (str): Folder path to save error logs.
        llm (object): LLM object used for paraphrasing.
        messages (list): Input messages list for the LLM.
        temperature (float, optional): Temperature setting for the LLM. Defaults to 0.7.
        num_iterations (int, optional): Number of iterations to generate paraphrases for each sentence. Defaults to 10.

    Returns:
        None
    """
    filtered_df = filter_dataframe_by_folder(read_loc, write_loc)
    
    if len(filtered_df) > 0:

        if grammar_path:
            print("Grammar Path Found")
            grammar=LlamaGrammar.from_file(grammar_path, verbose=False)
        
        # Get the remaining docs as a list to loop through
        remaining_docs = filtered_df['doc_id'].unique()
        total_docs = len(remaining_docs)
        print(f"{total_docs} Documents Left to Paraphrase")

        for doc_id in remaining_docs:
            # Initialize an empty list to store dictionaries
            result_data = []

            # Filter rows for the current document
            doc_rows = filtered_df[filtered_df['doc_id'] == doc_id]
            num_chunks = len(doc_rows)

            for chunk_id, sentence in enumerate(doc_rows['text']):
                print(f"Processing Document ID: {doc_id}, Chunk ID: {chunk_id} Out Of {num_chunks}")

                for i in range(num_iterations):
                    print(f"Iteration: {i + 1}")
                    if grammar_path:
                        result = call_local_llm(llm, messages, sentence, grammar, temperature)
                    else:
                        result = call_local_llm(llm, messages, sentence, temperature=temperature)
                        
                    formatted_result = format_output(result, error_loc)

                    # Append each result as a new dictionary to result_data
                    for item in formatted_result:
                        result_data.append({'doc_id': doc_id, 'chunk_id': chunk_id, 'result': item})

            # Create DataFrame from list of dictionaries for the current doc_id
            result_df = pd.DataFrame(result_data)

            # Save results for current doc_id to JSON Lines file
            read_and_write_docs.save_as_jsonl(result_df, f"{write_loc}/doc_{doc_id}.jsonl")

        print("All documents processed.")

    else:
        print("No documents to process.")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Paraphrase documents using a local LLM.")

	parser.add_argument("read_loc", type=str, help="File path to the input data.")
	parser.add_argument("write_loc", type=str, help="Folder path to save the output data.")
	parser.add_argument("error_loc", type=str, help="Folder path to save error logs.")
	parser.add_argument("model_path", type=str, help="Path to the local LLM model.")
	parser.add_argument("grammar_path", type=str, default=None, help="Path to the local grammar file.")
	parser.add_argument("--temperature", type=float, default=0.7, help="Temperature setting for the LLM. Defaults to 0.7.")
	parser.add_argument("--num_iterations", type=int, default=10, help="Number of iterations to generate paraphrases for each sentence. Defaults to 10.")
	
	args = parser.parse_args()
	
	system_prompt = (
	    "You are a paraphrasing assistant, given a sentence generate as many paraphrased "
	    "sentences as possible while preserving the original semantic meaning and style. "
	    "Return the rephrased sentences as a Python list. Aim for AT LEAST TWENTY sentences. "
	    "DO NOT INCLUDE ANY NOTE OR ADDITIONAL TEXT IN THE OUTPUT. "
	    "Make sure to WRAP ALL SENTENCES IN DOUBLE QUOTES AND USE ESCAPED SINGLE QUOTES INSIDE THEM. "
	    "If there are NAMED ENTITIES in the sentence DO NOT change the name."
	)
	
	# Convert the text to a single line string
	system_prompt = repr(system_prompt)
	
	example_input = "Known for being very delicate, the skill could take a lifetime to master."
	
	# List of paraphrased sentences
	paraphrased_sentences = [
	    "Famed for its delicacy, mastering the skill could take a lifetime.",
	    "Renowned for its fragility, the skill might require a lifetime to perfect.",
	    "Recognized for being extremely delicate, mastering this skill could take an entire lifetime.",
	    "Known for its delicate nature, the skill might take a lifetime to master.",
	    "Esteemed for its delicate nature, it could take a lifetime to master this skill.",
	    "Noted for its delicateness, the skill could take a whole lifetime to master.",
	    "Celebrated for being very delicate, mastering the skill might take a lifetime.",
	    "Admired for its fragility, the skill could take an entire lifetime to perfect.",
	    "Acclaimed for its delicate quality, mastering this skill might take a lifetime.",
	    "Distinguished by its delicacy, it could take a lifetime to master the skill.",
	    "Famous for being very delicate, the skill might take a lifetime to perfect.",
	    "Acknowledged for its delicate nature, mastering the skill could take a whole lifetime.",
	    "Well-known for its fragility, the skill might require a lifetime to master.",
	    "Recognized for its delicate quality, it could take a lifetime to master the skill.",
	    "Notable for being very delicate, the skill could require a lifetime to master.",
	    "Renowned for its delicateness, mastering the skill might take a whole lifetime.",
	    "Famed for its fragile nature, the skill could take a lifetime to master.",
	    "Celebrated for its delicate nature, mastering the skill could take an entire lifetime.",
	    "Noted for being extremely delicate, the skill might take a lifetime to master.",
	    "Esteemed for its fragility, the skill might require an entire lifetime to perfect.",
	    "Acclaimed for being very delicate, mastering this skill could take a whole lifetime.",
	    "Admired for its delicate quality, it might take a lifetime to master the skill."
	]
	
	# Convert the list to a single line string
	example_output = repr(paraphrased_sentences)
	
	input_messages = [
	    {"role": "system", "content": system_prompt},
	    {"role": "user", "content": example_input},
	    {"role": "assistant", "content": example_output}
	]


	llm = Llama(
	    model_path=args.model_path,
	    n_ctx=4096,
	    n_threads=10,
	    n_gpu_layers=-1,
	    verbose=False,
	    # flash_attn=True
	)
	
	paraphrase_llm(read_loc=args.read_loc,
				   write_loc=args.write_loc,
				   error_loc=args.error_loc,
				   llm=llm,
				   messages=input_messages,
				   num_iterations=args.num_iterations,
				   grammar_path=args.grammar_path,
				   temperature=args.temperature)