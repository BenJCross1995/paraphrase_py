import os
import json
import argparse
import logging
import pandas as pd
from scorer import ParaphraseScorer  # Ensure scorer.py is accessible

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def read_jsonl(file_path):
    """Reads a JSONL file and converts it into a pandas DataFrame."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parsed_line = json.loads(line)
                if isinstance(parsed_line, list) and len(parsed_line) == 1:
                    data.append(parsed_line[0])
                else:
                    data.append(parsed_line)
        return pd.DataFrame(data)
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

def write_jsonl(data, output_file_path):
    """Writes a pandas DataFrame to a JSONL file."""
    try:
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for _, row in data.iterrows():
                json.dump(row.to_dict(), file)
                file.write('\n')
        logging.info(f"File saved: {output_file_path}")
    except Exception as e:
        logging.error(f"Error writing {output_file_path}: {e}")

def process_file(input_file, output_file, model_type):
    """Processes a single JSONL file and saves the results."""
    
    print(f"Processing file: {input_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load input data
    print("Reading input file...")
    df = read_jsonl(input_file)

    print(f"DataFrame loaded: {df.shape}")  # Debug

    if df.empty:
        logging.error("Input file is empty or could not be read. Skipping processing.")
        return

    # Rename columns if necessary
    print("Renaming columns if needed...")
    df.rename(columns={
        "original_sentence": "original",
        "text": "original",
        "paraphrased_text": "rephrased"
    }, inplace=True)
    
    # Load model
    print("Initializing ParaphraseScorer...")
    parascore_free = ParaphraseScorer(score_type='parascore_free', model_type=model_type)
    print(f"{model_type} Scorer Loaded")

    # Process file
    try:
        print("Calculating scores...")
        df_with_score = parascore_free.calculate_score(df)
        print("Score calculation completed.")

        print(f"Writing output to {output_file}...")
        write_jsonl(df_with_score, output_file)
        print(f"Processing complete: {output_file}")

    except Exception as e:
        logging.error(f"Error during processing: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single JSONL file using a specified model.")
    parser.add_argument("--input_file", type=str, required=True, help="Absolute path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Absolute path to save the output JSONL file.")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model name or local model path.")

    args = parser.parse_args()
    process_file(args.input_file, args.output_file, args.model)
