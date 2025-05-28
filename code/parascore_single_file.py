import os
import json
import argparse
import logging
import pandas as pd
import sys
from scorer import ParaphraseScorer
from read_and_write_docs import read_jsonl, write_jsonl

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_file(input_file, output_file, model_type, num_layers=None,
                 text_column='text', rephrased_column='paraphrased_text'):
    """
    Processes a single JSONL file and saves the results.

    Parameters:
        input_file (str): Path to input JSONL.
        output_file (str): Path for output JSONL.
        model_type (str): Model name or path.
        num_layers (int, optional): Number of model layers.
        text_column (str): Column name for original text.
        rephrased_column (str): Column name for paraphrased text.
    """
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
    rename_mapping = {
        "original_sentence": "original",
        text_column: "original",
        rephrased_column: "rephrased"
    }
    df.rename(columns=rename_mapping, inplace=True)
    print(f"Columns after renaming: {list(df.columns)}")
    
    # Load model
    print("Initializing ParaphraseScorer...")
    parascore_free = ParaphraseScorer(
        score_type='parascore_free',
        model_type=model_type,
        num_layers=num_layers
    )
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
    parser.add_argument(
        "--num_layers", type=int, default=None, help="Number of layers to use in the model (optional, default: None)."
    )
    parser.add_argument(
        "--text_column", type=str, default="text",
        help="Column name for original text (default: 'text')."
    )
    parser.add_argument(
        "--rephrased_column", type=str, default="paraphrased_text",
        help="Column name for paraphrased text (default: 'paraphrased_text')."
    )

    args = parser.parse_args()

    # Check whether input file exists and skip if it does not
    if not os.path.isfile(args.input_file):
        logging.warning(f"Input file not found – skipping: {args.input_file}")
        sys.exit(0)
    
    # Check whether ParaScore file already exists and exit if it does
    if os.path.exists(args.output_file):
        logging.info(f"Output already exists – skipping: {args.output_file}")
        sys.exit(0)
        
    process_file(
        args.input_file,
        args.output_file,
        args.model,
        args.num_layers,
        args.text_column,
        args.rephrased_column
    )
