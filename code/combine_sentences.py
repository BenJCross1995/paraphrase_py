"""Script to concatenate sentences to a certain length"""
import argparse
import pandas as pd
import read_and_write_docs

# Function to count words in a text
def count_words(text):
    return len(text.split())


def concatenate_sentences(df, length_threshold=50):
    """
    Concatenate sentences in DataFrame if their input_length is less than the length_threshold.
    Group by 'id' column and concatenate sentences accordingly.
    """
    concatenated_sentences = []  # List to store the concatenated sentences
    additional_columns = {}  # Dictionary to store additional column values
    
    # Iterate over the DataFrame grouped by 'id'
    for id, group in df.groupby('id'):
        # Iterate over each row in the group
        i = 0
        while i < len(group):
            j = i + 1
            current_row_data = group.iloc[i]
            current_sentence = current_row_data['text'].strip()
            current_length = len(current_sentence)
            
            # Loop to concatenate sentences until the length threshold is reached or no more rows are available
            while current_length < length_threshold and j < len(group):
                new_row = group.iloc[j]
                new_sentence = current_sentence + " " + new_row['text'].strip()
                current_length = len(new_sentence)
                current_sentence = new_sentence
                j+=1

            # Add the current concatenated sentence to the list
            concatenated_sentences.append(current_sentence.strip())
            # Store additional column values for the current concatenated sentence
            for col in df.columns:
                if col != 'text':
                    if col not in additional_columns:
                        additional_columns[col] = []
                    additional_columns[col].append(current_row_data[col])
            i += 1
    
    # Add additional column values to the resulting dataframe
    additional_columns['text'] = concatenated_sentences

    result_df = pd.DataFrame(additional_columns)
    result_df['input_length'] = result_df['text'].apply(len)
    result_df['word_count'] = result_df['text'].apply(count_words)
    result_df.drop('subchunk', inplace=True, axis=1)

    # Filter out final sentences shorter than the threshold.
    result_df = result_df[result_df['input_length'] >= length_threshold]

    return result_df


def main():

    # Parse arguments from user
    parser = argparse.ArgumentParser(description='Combine sentences to a certain length')
    parser.add_argument('--file_path', type=str, help='Path to jsonl file', required=True)
    parser.add_argument('--output_file_path', type=str, help='Output filepath', required=True)
    parser.add_argument('--length_threshold', type=int, default=50, help='The minimum input length of phrases.')
    args = parser.parse_args()

    df = read_and_write_docs.read_jsonl_file(args.file_path)
    
    concat_df = concatenate_sentences(df, args.length_threshold)

    read_and_write_docs.save_as_jsonl(concat_df, args.output_file_path)

    print("Preprocessing Complete")

if __name__ == '__main__':
    main()