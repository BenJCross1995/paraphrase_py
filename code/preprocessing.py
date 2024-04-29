"""Python pre-processing script"""

# ----IMPORT LIBRARIES---- #
import argparse
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import read_and_write_docs

# ----FUNCTIONS TO SAMPLE DATA---- #
def sample_equal_distribution(df, total_rows):
    # Check if total_rows is even, if not, make it even
    if total_rows % 2 != 0:
        total_rows += 1
        print("The total number of rows should be even for equal distribution. Adjusting to the next even number:", total_rows)
    
    # Filter rows where 'same' is True and sample
    same_true_sample = df[df['same'] == True].sample(n=total_rows // 2, replace=True)
    
    # Filter rows where 'same' is False and sample
    same_false_sample = df[df['same'] == False].sample(n=total_rows // 2, replace=True)
    
    # Concatenate and shuffle the samples
    sample = pd.concat([same_true_sample, same_false_sample]).sample(frac=1).sort_values(by='id').reset_index(drop=True)
    
    return sample

def filter_ids_from_df(df, truth_sample):
    filtered_df = df[df['id'].isin(truth_sample['id'])].sort_values(by='id').reset_index(drop=True)
    return filtered_df

# ----SENTENCE SPLITTING---- #
def split_sentences(text):
    
    text = re.sub(r'[\'"]', '', text)
    # Define the regular expression pattern to split at sentence-ending punctuation marks
    pattern = r'(?<=[.!?]) +'
    # Split the text using the regular expression pattern
    sentences = re.split(pattern, text)
    # Strip leading and trailing whitespace from each sentence
    sentences = [sentence.strip() for sentence in sentences]
    return sentences

# Function to count words in a text
def count_words(text):
    return len(text.split())

def apply_sentence_split(df, input_col='text', output_col='text'):
    
    df['sentence'] = df[input_col].apply(split_sentences)
    
    df_expanded = df.explode('sentence').reset_index(drop=True)
    
    df_expanded = df_expanded.drop(columns=input_col)
    
    df_expanded.rename(columns={'sentence':output_col}, inplace=True)
    
    df_expanded.insert(1, 'chunk_id', df_expanded.groupby('id').cumcount())
    
    df_expanded.insert(4, 'word_count', df_expanded['text'].apply(count_words))
    
    df_expanded = df_expanded[df_expanded['word_count'] >= 1]
    
    return df_expanded

def split_rows_by_word_count(df, n):
    """
    Split rows in DataFrame where word_count exceeds 'n' words into chunks of at most 'n' words.
    Keeps the same values for all other columns apart from chunk_id.
    """
    
    max_words = df['word_count'].max()
    
    if max_words >= n:
    
        new_rows = []  # Initialize an empty list to store the new rows
        drop_indices = []  # Initialize a list to store indices of rows to be dropped
        chunk_counter = {}  # Initialize a dictionary to keep track of chunk counts for each chunk_id
    
        # Iterate over the DataFrame
        for index, row in df.iterrows():
            if row['word_count'] > n:
                words = row['text'].split()  # Split the text into words
                num_chunks = len(words) // n  # Calculate the number of chunks
                if len(words) % n != 0:
                    num_chunks += 1
                
                chunk_id = row['chunk_id']
                if chunk_id not in chunk_counter:
                    chunk_counter[chunk_id] = 1
                
                for i in range(num_chunks):
                    start_idx = i * n
                    end_idx = min((i + 1) * n, len(words))
                    new_row = row.copy()  # Create a copy of the original row
                    new_row['text'] = ' '.join(words[start_idx:end_idx])  # Assign the chunked text
                    new_row['subchunk_id'] = chunk_counter[chunk_id]
                    new_rows.append(new_row)  # Append the new row to the list
                    chunk_counter[chunk_id] += 1
    
                drop_indices.append(index)  # Add the index of the original row to the list of indices to be dropped
    
        # Drop the original rows from the DataFrame
        updated_df = df.drop(drop_indices)
    
        # Concatenate the new rows into the updated DataFrame
        updated_df = pd.concat([updated_df, pd.DataFrame(new_rows)])

    else:
        updated_df = df.copy()
        updated_df['subchunk_id'] = 0
    
    updated_df['word_count'] = updated_df['text'].apply(count_words)
    updated_df['input_length'] = updated_df['text'].apply(len)
    updated_df['chunk_id'] = updated_df.groupby('id').cumcount()
    updated_df['subchunk'] = updated_df['subchunk_id'].apply(lambda x: 1 if x >= 1 else 0)

    updated_df.reset_index(inplace=True)  # Bring the original index in as a regular column
    updated_df.sort_values(by=['index', 'chunk_id', 'subchunk_id'], inplace=True)
    updated_df.reset_index(drop=True, inplace=True)  # Reset the index to create a new one

    df_text = updated_df.pop('text')
    updated_df['text'] = df_text
    

    return updated_df

def main():
    parser = argparse.ArgumentParser(description='Preprocessing steps for paraphrasing using LLMs')
    parser.add_argument('--file_path', type=str, help='Path to jsonl file')
    parser.add_argument('--output_file_path', type=str, help='Output filepath')
    args = parser.parse_args()

    print("Beginning Preprocessing")
    df = read_and_write_docs.read_jsonl_file(args.file_path)
    
    split_df = apply_sentence_split(df)

    split_row_by_wc_data = split_rows_by_word_count(split_df, n=200)

    read_and_write_docs.save_as_jsonl(split_row_by_wc_data, args.output_file_path)

    print("Preprocessing Complete")

if __name__ == '__main__':
    main()