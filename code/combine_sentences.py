import argparse
import pandas as pd
import read_and_write_docs

def count_words(text):
    """Count the number of words in a text string.

    Args:
        text (str): The input text string.

    Returns:
        int: The number of words in the text.
    """
    return len(text.split())

def concatenate_sentences(df, length_threshold=50, threshold_type='word'):
    """Concatenate sentences in a DataFrame if their length is less than the length_threshold.
    
    Group sentences by the 'id' column and concatenate them accordingly.

    Args:
        df (pd.DataFrame): The input DataFrame containing sentences.
        length_threshold (int): The minimum length of concatenated sentences.
        threshold_type (str): The type of length threshold, either 'char' for characters or 'word' for words.

    Returns:
        pd.DataFrame: A DataFrame with concatenated sentences.
    
    Raises:
        ValueError: If threshold_type is not 'char' or 'word'.
    """
    threshold_type = threshold_type.lower()
    if threshold_type not in {'char', 'word'}:  # Using a set for efficient membership testing
        raise ValueError("threshold_type must be either 'char' or 'word'")

    concatenated_sentences = []  # List to store the concatenated sentences
    additional_columns = {}  # Dictionary to store additional column values
    row_counts = [] # Store the number of chunks
    start_points = [] # Store the original sentence
    
    # Iterate over the DataFrame grouped by 'id'
    for id, group in df.groupby('id'):
        i = 0
        while i < len(group):
            j = i + 1
            current_row_data = group.iloc[i]
            current_sentence = current_row_data['text'].strip()
            r_count = 1
            start_point = current_sentence
            
            if threshold_type == 'char':
                current_length = len(current_sentence)
            elif threshold_type == 'word':
                current_length = count_words(current_sentence)
            
            # Loop to concatenate sentences until the length threshold is reached or no more rows are available
            while ((current_length < length_threshold) and (j < len(group))):
                new_row = group.iloc[j]
                new_sentence = current_sentence + " " + new_row['text'].strip()
                
                if threshold_type == 'char':
                    current_length = len(new_sentence)
                elif threshold_type == 'word':
                    current_length = count_words(new_sentence)
                
                current_sentence = new_sentence
                j += 1
                r_count += 1

            # Add the current concatenated sentence to the list
            concatenated_sentences.append(current_sentence.strip())
            row_counts.append(r_count)
            start_points.append(start_point)
            # Store additional column values for the current concatenated sentence
            for col in df.columns:
                if col != 'text':
                    if col not in additional_columns:
                        additional_columns[col] = []
                    additional_columns[col].append(current_row_data[col])
            i += 1
    
    # Add additional column values to the resulting dataframe
    additional_columns['chunk_count'] = row_counts
    additional_columns['original_sentence'] = start_points
    additional_columns['text'] = concatenated_sentences
	
    result_df = pd.DataFrame(additional_columns)
    result_df['input_length'] = result_df['text'].apply(len)
    result_df['word_count'] = result_df['text'].apply(count_words)
    result_df.drop('subchunk', inplace=True, axis=1)

    # Filter out final sentences shorter than the threshold.
    if threshold_type == 'char':
        result_df = result_df[result_df['input_length'] >= length_threshold]
    elif threshold_type == 'word':
        result_df = result_df[result_df['word_count'] >= length_threshold]

    return result_df

def main():
    """Main function to parse arguments and process the input file.
    
    Parses command line arguments, reads the input file, processes the sentences,
    and saves the output to the specified file path.
    """
    # Parse arguments from user
    parser = argparse.ArgumentParser(description='Combine sentences to a certain length')
    parser.add_argument('--file_path', type=str, help='Path to jsonl file', required=True)
    parser.add_argument('--output_file_path', type=str, help='Output filepath', required=True)
    parser.add_argument('--length_threshold', type=int, default=50, help='The minimum input length of phrases.')
    parser.add_argument('--threshold_type', type=str, choices=['char', 'word'], default='char', help='Type of length threshold: char for character count, word for word count')
    args = parser.parse_args()

    df = read_and_write_docs.read_jsonl_file(args.file_path)
    
    concat_df = concatenate_sentences(df, args.length_threshold, args.threshold_type)

    read_and_write_docs.save_as_jsonl(concat_df, args.output_file_path)

    print("Preprocessing Complete")

if __name__ == '__main__':
    main()
