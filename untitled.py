import json
import os
import read_and_write_docs

import pandas as pd

from llama_cpp import Llama
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

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

system_prompt = """
Given the sentence, generate as many paraphrased sentences as possible while preserving the original semantic meaning and style. 
Return the rephrased sentences in a python list format. Aim for AT LEAST TWENTY sentences. DO NOT INCLUDE ANY NOTES OR ADDITIONAL TEXT IN THE OUTPUT.

An example is below:
--------
Sentence: ```"Known for being very delicate, the skill could take a lifetime to master."```

Rephrased Sentences: ```["The skill is well known for its delicacy and could require a lifetime to perfect.", "The skill's reputation for delicateness suggests that it could take a whole lifetime to master.", "It may take a lifetime to master the skill, which is renowned for its delicacy.", "The delicacy of the skill means it could take a lifetime to master."]```
--------
Sentence: ```{original_user_supplied_sentence}```
"""

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{original_user_supplied_sentence}"),
    ]
)

def convert_to_phi(original_sentence, prompt_input=final_prompt):

    messages = prompt_input.messages
    
    formatted_messages = ""

    for message in messages:
        if isinstance(message, SystemMessagePromptTemplate):
            formatted_messages += f"<|assistant|>\n{message.prompt.template.replace('\n', '')} <|end|>\n"
        elif isinstance(message, FewShotChatMessagePromptTemplate):
            formatted_messages += f"<|user|>\n{message.examples[0]['original_user_supplied_sentence'].replace('\n', '')} <|end|>\n"
            formatted_messages += f"<|assistant|>\n{message.examples[0]} <|end|>\n"
        elif isinstance(message, HumanMessagePromptTemplate):
            formatted_messages += f"<|user|>\n{message.prompt.template.replace('\n', '')} <|end|>\n"
    
    formatted_messages += f"<|assistant|>"

    formatted_prompt = formatted_messages.replace("<|user|>\n{original_user_supplied_sentence} <|end|>", f"<|user|>\n{original_sentence} <|end|>")
    
    return formatted_prompt

def initialise_llm(model_path, grammar_path):
    
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=4096,
        n_threads=10,
        n_gpu_layers=-1,
        n_batch=512,
        f16_kv=True,
        verbose=False,
        # flash_attn=True,
        grammar_path=grammar_path
    )

    return llm

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

def format_output(llm_output):
    """
    Formats the LLM output to ensure it is a valid JSON list.
    Tries to parse it as a JSON list and returns the parsed list if successful.
    If parsing fails, prints the result and returns an empty list.
    """
    
    result = check_starting_bracket(llm_output)
    result = check_trailing_bracket(result)

    try:
        parsed_result = json.loads(result)
        return parsed_result
    except json.JSONDecodeError:
        print("JSONDecodeError")
        read_and_write_docs.save_error_as_txt(result, error_loc)
        return []

def paraphrase_llm(read_loc, write_loc, chain, num_iterations):
    filtered_df = filter_dataframe_by_folder(read_loc, write_loc)

    if len(filtered_df) > 0:
        # Get the remaining docs as a list to loop through
        remaining_docs = filtered_df['doc_id'].unique()
        print(f"{len(remaining_docs)} Documents Left to Paraphrase")

        for doc_id in remaining_docs:
            # Initialize an empty list to store dictionaries
            result_data = []

            # Filter rows for the current document
            doc_rows = filtered_df[filtered_df['doc_id'] == doc_id]
            num_chunks = len(doc_rows)

            for chunk_id, sentence in enumerate(doc_rows['text']):
                print(f"Processing Document ID: {doc_id}, Chunk ID: {chunk_id} Out Of {num_chunks}")

                for i in range(num_iterations):
                    print(f"Iteration {i+1}:")
                    result = chain.invoke({"original_user_supplied_sentence": sentence})

                    # Append each result as a new dictionary to result_data
                    for item in result:
                        result_data.append({'doc_id': doc_id, 'chunk_id': chunk_id, 'result': item})

            # Create DataFrame from list of dictionaries for the current doc_id
            result_df = pd.DataFrame(result_data)

            # Save results for current doc_id to JSON Lines file
            read_and_write_docs.save_as_jsonl(result_df, f"{write_loc}/doc_{doc_id}.jsonl")

        print("All documents processed.")

    else:
        print("No documents to process.")

def main():
    """Main function to parse arguments and process the input file.
    
    Parses command line arguments, reads the input file, processes the sentences,
    and saves the output to the specified file path.
    """

    start_time = time.time()
	
    # Parse arguments from user
    parser = argparse.ArgumentParser(description='Paraphrase a dataframe using an LLM')
    parser.add_argument('--read_loc', type=str, help='Path to data', required=True)
    parser.add_argument('--write_loc', type=str, help='Path to write_data', required=True)
    parser.add_argument('--model_loc', type=str, help='Output filepath', required=True)
    parser.add_argument('--num_impostors', type=int, default=10, help='The number of impostors for each sentence.')
    args = parser.parse_args()

    # Pull in the unknown and rephrased docs
    try:
        unknown = read_and_write_docs.read_jsonl_file(args.unknown_file_path)
        rephrased = read_and_write_docs.read_jsonl_file(args.rephrased_file_path)
    except:
        unknown = pd.read_csv(args.unknown_file_path)
        rephrased = pd.read_csv(args.rephrased_file_path)

    # Log each doc in the unknown whether to keep or remove
    print("Logging which sentences to keep based on chunks to threshold...")
    start_log_time = time.time()
    logged_unknown = filter_chunked_unknown(unknown)
    end_log_time = time.time()
    print(f"Time taken to log sentences: {end_log_time - start_log_time:.2f} seconds")

    # Run the function to chunk the rephrased docs
    print("Chunking the rephrased data...")
    start_chunk_time = time.time()
    result = chunk_rephrased(logged_unknown, rephrased, args.num_impostors)
    end_chunk_time = time.time()
    print(f"Time taken to chunk rephrased data: {end_chunk_time - start_chunk_time:.2f} seconds")

    read_and_write_docs.save_as_jsonl(result, args.output_file_path)

    print("Rephrasing complete!")
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()