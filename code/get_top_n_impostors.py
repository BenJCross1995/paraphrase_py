import json
import os
import argparse

from read_and_write_docs import read_jsonl, write_jsonl

def get_top_n_impostors_parascore(input_file, output_file, num_impostors):

    if not os.path.isfile(input_file):
        print(f"Error: input file '{input_file}' does not exist. Aborting.")
        sys.exit(1)

    if os.path.exists(output_file):
        print(f"Error: output file '{output_file}' already exists. Aborting to avoid overwriting.")
        sys.exit(1)
        
    print(f"Processing file: {input_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print("Reading input file...")
    df = read_jsonl(input_file)

    print("Removing duplicate rows based on 'rephrased' and 'parascore_free'...")
    df = df.drop_duplicates(subset=['rephrased', 'parascore_free'])
    
    print(f"Getting top {num_impostors} impostors...")
    top_n = df.sort_values(['parascore_free'], ascending=[False]).head(num_impostors)
    result = top_n.reset_index(drop=True)

    
    print(f"Writing output to {output_file}")
    write_jsonl(result, output_file)
    
    print(f"Processing complete: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the top n impostors from a parascore file and save it.")
    parser.add_argument("--input_file", type=str, required=True, help="Absolute path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Absolute path to save the output JSONL file.")
    parser.add_argument("--num_impostors", type=int, required=True, default=500, help="Number of impostors to keep (default: 500).")

    args = parser.parse_args()
    get_top_n_impostors_parascore(args.input_file, args.output_file, args.num_impostors)