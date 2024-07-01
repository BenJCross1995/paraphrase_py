from parrot import Parrot
import argparse
import json
import torch
import warnings
import read_and_write_docs

import pandas as pd

warnings.filterwarnings("ignore")

def parrot_paraphrase(phrase, n_iterations, model):
  stored_phrases = []

  for i in range(1, n_iterations):
    paraphrases = model.augment(input_phrase=phrase,
                                use_gpu=True,
                                diversity_ranker="levenshtein",
                                do_diverse=False,
                                max_return_phrases=100,
                                max_length=1000,
                                adequacy_threshold=0.75,
                                fluency_threshold=0.70)

    if paraphrases is not None:
      num_phrases = len(paraphrases)

      for j in range(1, num_phrases):
        paraphrase = paraphrases[j][0]

        stored_phrases.append(paraphrase)
    else:
      stored_phrases = []

  result = list(set(stored_phrases))

  return result


def paraphrase_dataframe(df, save_location, model, n_iterations=10, read_previous=None):

  if read_previous is not None:
    completed_df = read_and_write_docs.read_jsonl_file(read_previous)
    completed_ids = set(completed_df['doc_id'].astype(str) + '_' + completed_df['chunk_id'].astype(str))
  else:
    completed_df = pd.DataFrame() # Initialise empty dataframe
    completed_ids = set()  # Initialize the set to an empty set

  new_rows = []
  for index, row in df.iterrows():

    row_id = str(row['doc_id']) + '_' + str(row['chunk_id'])
    if row_id in completed_ids:
      continue # Skip rows that have already been paraphrased
    print(f"Paraphrasing sentence {index + 1} out of {df.index.max() + 1}")
    result = parrot_paraphrase(row['text'], n_iterations, model)
    for paraphrase in result:
      new_row = {
          'index': index,
          'doc_id': row['doc_id'],
          'author_id': row['author_id'],
          'chunk_id': row['chunk_id'],
          'gender': row['gender'],
          'age': row['age'],
          'topic': row['topic'],
          'sign': row['sign'],
          'date': row['date'],
          'paraphrase': paraphrase
      }
      new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)
    completed_df = pd.concat([completed_df, new_df], ignore_index=True)

    # Save the DataFrame to a CSV file after each iteration
    read_and_write_docs.save_as_jsonl(completed_df)

def main():
    """Main function to parse arguments and process the input file.
    
    Parses command line arguments, reads the input file, processes the sentences,
    and saves the output to the specified file path.
    """
	
    # Parse arguments from user
    parser = argparse.ArgumentParser(description='Paraphrase sentences using the Parrot framework')
    parser.add_argument('--read_file_path', type=str, help='Path to rephrased jsonl file', required=True)
    parser.add_argument('--write_file_path', type=str, help='Path to write jsonl file', required=True)
    parser.add_argument('--model_path', type=str, help='Path to the model', required=True)
    parser.add_argument('--num_iterations', type=int, default=10, help='The number of model iterations.')
    args = parser.parse_args()

    # Pull in the unknown and rephrased docs
    rephrased = read_and_write_docs.read_jsonl_file(args.read_file_path)

    parrot = Parrot(args.model_path)

    print("Model Loaded")
          
    paraphrase_dataframe(rephrased, args.write_file_path, parrot, n_iterations=args.num_iterations)

if __name__ == '__main__':
    main()