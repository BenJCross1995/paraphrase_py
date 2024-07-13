import argparse
import read_and_write_docs
import warnings

import pandas as pd

from parrot import Parrot

warnings.filterwarnings("ignore")

def parrot_paraphrase(phrase, n_iterations, model, diverse=False):
    """
    Generates paraphrases for a given phrase using the Parrot paraphraser.

    Args:
        phrase (str): The phrase to be paraphrased.
        n_iterations (int): Number of iterations to generate paraphrases.
        diverse (bool, optional): Flag to enable diverse paraphrasing. Defaults to False.

    Returns:
        list: A list of unique paraphrases.

    Raises:
        ValueError: If `diverse` is not a boolean value.
    """
    
    # Set thresholds based on the diversity flag
    if diverse == False:
        diverse = False
        ad_thresh = 0.99
        fl_thresh = 0.9
    elif diverse == True:
        diverse = True
        ad_thresh = 0.7
        fl_thresh = 0.7
    else:
        raise ValueError("The 'diverse' argument must be a boolean value.")

    # Initialize the list to store paraphrases
    stored_phrases = []

    # Generate paraphrases for the given number of iterations
    for i in range(1, n_iterations):
        paraphrases = model.augment(input_phrase=phrase,
                                    use_gpu=True,
                                    diversity_ranker="levenshtein",
                                    do_diverse=diverse,
                                    max_return_phrases=100,
                                    max_length=1000,
                                    adequacy_threshold=ad_thresh,
                                    fluency_threshold=fl_thresh)

        # If paraphrases are generated, add them to the stored phrases list
        if paraphrases is not None:
            num_phrases = len(paraphrases)
            for j in range(1, num_phrases):
                paraphrase = paraphrases[j][0]
                stored_phrases.append(paraphrase)
        else:
            stored_phrases = []

    # Remove duplicates by converting the list to a set and back to a list
    result = list(set(stored_phrases))

    return result

def paraphrase_dataframe(df, save_location, n_iterations=10, model=None, diverse=False):
    """
    Paraphrases the text in the DataFrame and saves the results to JSONL files by document ID.

    Args:
        df (pd.DataFrame): The DataFrame containing the text to be paraphrased.
        save_location (str): The location to save the JSONL files.
        n_iterations (int, optional): Number of iterations to generate paraphrases. Defaults to 10.
        model: The Parrot model to use for paraphrasing.
        diverse (bool, optional): Flag to enable diverse paraphrasing. Defaults to False.

    Returns:
        None
    """
    # Get unique document IDs
    unique_doc_ids = df['doc_id'].unique()

    # Loop through each document ID
    for doc_id in unique_doc_ids:
        # Filter the DataFrame by the current document ID
        doc_df = df[df['doc_id'] == doc_id]
        
        new_rows = []
        
        # Loop through each row in the filtered DataFrame
        for index, row in doc_df.iterrows():
            print(f"Paraphrasing sentence {index + 1} out of {len(doc_df)} for doc_id {doc_id}")  # Added print statement
            result = parrot_paraphrase(row['text'], n_iterations, model, diverse)
            
            # Add the paraphrases to the new rows list
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
                    'text': paraphrase
                }
                new_rows.append(new_row)
        
        # Save the paraphrased results to a JSONL file for the current document ID
        jsonl_path = f"{save_location}/doc_{doc_id}.jsonl"
        read_and_write_docs.save_as_jsonl(new_rows, jsonl_path)

    print("Paraphrasing complete.")

def main():
    parser = argparse.ArgumentParser(description="Paraphrase text in a DataFrame and save the results to JSONL files.")
    parser.add_argument('input_file', type=str, help='Path to the input jsonl file.')
    parser.add_argument('save_location', type=str, help='Directory where the JSONL files will be saved.')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations to generate paraphrases. Defaults to 10.')
    parser.add_argument('--model_tag', type=str, default="prithivida/parrot_paraphraser_on_T5", help='Model tag for the Parrot paraphraser. Defaults to "prithivida/parrot_paraphraser_on_T5".')
    parser.add_argument('--diverse', action='store_true', help='Enable diverse paraphrasing. Defaults to False.')

    args = parser.parse_args()

    # Initialize the Parrot model
    parrot = Parrot(model_tag=args.model_tag)

    # Read the input CSV file into a DataFrame
    df = read_and_write_docs.read_jsonl_file(args.input_file)

    # Paraphrase the DataFrame and save the results
    paraphrase_dataframe(df, args.save_location, args.iterations, parrot, args.diverse)

if __name__ == "__main__":
    main()