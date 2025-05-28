import json
import re
import os
import random
import pandas as pd
from read_and_write_docs import read_jsonl, write_jsonl
import argparse

def create_temp_doc_id(input_text):
    """Create a new doc id by preprocessing the current id"""
    
    # Extract everything between the brackets
    match = re.search(r'\[(.*?)\]', input_text)
    
    if match:
        extracted_text = match.group(1)
        # Replace all punctuation and spaces with "_"
        cleaned_text = re.sub(r'[^\w]', '_', extracted_text)
        # Replace multiple underscores with a single "_"
        final_text = re.sub(r'_{2,}', '_', cleaned_text)
        return final_text.lower()
        
    return None

def apply_temp_doc_id(df):
    """Apply the doc id function on the dataframe"""
    
    # Rename doc_id to orig_doc_id first    
    df.rename(columns={'doc_id': 'orig_doc_id'}, inplace=True)

    # Create the new doc_id column directly
    df['doc_id'] = df['orig_doc_id'].apply(create_temp_doc_id)

    df.drop("orig_doc_id", axis=1, inplace=True)
    
    # Move the new doc_id column to the front
    cols = ['doc_id'] + [col for col in df.columns if col not in ['doc_id', 'text']] + ['text']

    df = df[cols]

    return df

def get_known_author_profile(loc):

    df = read_jsonl(loc)

    # update doc_id
    df = apply_temp_doc_id(df)

    df_sorted = df.sort_values(by=['author', 'doc_id'])

    group_cols = [col for col in df_sorted.columns if col not in ['doc_id', 'text']]

    df_grouped = df_sorted.groupby(group_cols, as_index=False).agg(
        compiled_docs=('doc_id', list),
        compiled_files=('doc_id', lambda x: [f"{doc}.jsonl" for doc in x]),
        text=('text', lambda texts: ' '.join(texts))
    )
    
    return df_grouped

def filter_authors_with_complete_files(df: pd.DataFrame, root_dir: str):
    """
    Split *df* into two DataFrames:

    - valid_df   → every file in `compiled_files` exists under *root_dir*
    - missing_df → at least one file is missing

    Returns (valid_df, missing_df).
    """
    def has_all_files(file_list):
        return all(os.path.isfile(os.path.join(root_dir, f)) for f in file_list)

    mask = df['compiled_files'].apply(has_all_files)
    return df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True)

def impostor_profile(df, file_directory, save_directory, n):
    """
    For each author in df, generate n compiled document samples.
    
    For each file in the author's 'compiled_files':
      - Load the file using read_and_write_docs.read_jsonl() (returns a DataFrame).
      - Sample n rows (with replacement) to get 'rephrased' and 'parascore_free' values.
    
    For each sample index (0 to n-1):
      - Concatenate sampled texts (in file order) into 'rephrased'.
      - Collect parascore values into 'parascore_list' and compute their average ('parascore_free').
    
    The original "text" column is renamed to "original". The resulting DataFrame for each author
    is saved as <save_directory>/<author_name>.jsonl using read_and_write_docs.write_jsonl().
    """
    os.makedirs(save_directory, exist_ok=True)

    total_authors = len(df)
    
    for idx, row in df.iterrows():
        file_list = row['compiled_files']
        per_file_text_samples = []      # One list per file, each of length n
        per_file_parascore_samples = [] # One list per file, each of length n
        
        for file_name in file_list:
            file_path = os.path.join(file_directory, file_name)
            try:
                data_df = read_jsonl(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                per_file_text_samples.append([''] * n)
                per_file_parascore_samples.append([None] * n)
                continue

            if data_df is None or data_df.empty:
                per_file_text_samples.append([''] * n)
                per_file_parascore_samples.append([None] * n)
                continue

            sampled_df = data_df.sample(n=n, replace=True)
            texts = list(sampled_df['rephrased'].fillna(''))
            parascores = []
            for ps in list(sampled_df['parascore_free']):
                try:
                    parascores.append(float(ps) if ps is not None else None)
                except ValueError:
                    parascores.append(None)
            per_file_text_samples.append(texts)
            per_file_parascore_samples.append(parascores)
        
        new_rows = []
        for sample_index in range(n):
            sample_texts = [per_file_text_samples[i][sample_index] for i in range(len(file_list))]
            sample_parascores = [per_file_parascore_samples[i][sample_index] for i in range(len(file_list))]
            compiled_text = " ".join(sample_texts)
            valid_parascores = [p for p in sample_parascores if isinstance(p, (int, float))]
            avg_score = sum(valid_parascores) / len(valid_parascores) if valid_parascores else None
            
            new_row = row.to_dict()
            new_row['rephrased'] = compiled_text
            new_row['parascore_list'] = sample_parascores
            new_row['parascore_free'] = avg_score
            new_rows.append(new_row)
        
        new_df = pd.DataFrame(new_rows)
        new_df.rename(columns={'text': 'original'}, inplace=True)
        author_name = row.get('author', f"author_{idx}")
        save_path = os.path.join(save_directory, f"{author_name}.jsonl")
        write_jsonl(new_df, save_path)

        print(f"Author {idx+1} out of {total_authors} complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate impostor profiles from known author profiles.")
    parser.add_argument("--known_loc", required=True, help="Path to the known author profile JSONL file.")
    parser.add_argument("--known_save_loc", required=True, help="Path to save the known author profile JSONL file.")
    parser.add_argument("--parascore_dir", required=True, help="Directory containing parascore JSONL files.")
    parser.add_argument("--save_dir", required=True, help="Directory to save the output JSONL files.")
    parser.add_argument("--n", type=int, required=True, help="Number of compiled document samples per author.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling.")
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    
    if os.path.isfile(args.known_save_loc):
        print(f"📄 Found existing known-author profile → {args.known_save_loc}")
        known_df = read_jsonl(args.known_save_loc)
    else:
        known_df = get_known_author_profile(args.known_loc)
        write_jsonl(known_df, args.known_save_loc)
        print("✅ Known author profile saved to disk")
    
    # Check that all of the files for each author exist, we do not want to build a half full profile etc.
    valid_df, missing_df = filter_authors_with_complete_files(known_df, args.parascore_dir)
    
    if not missing_df.empty:
        skipped = ", ".join(missing_df['author'].astype(str).unique())
        print(f"⚠️  Skipping impostor generation for authors "
              f"(missing parascore files): {skipped}")
    impostor_profile(valid_df, args.parascore_dir, args.save_dir, args.n)