import json
import re
import pandas as pd

class TextPreprocessor:
    def __init__(self):
        pass # Placeholder for any future initialisation

    def sample_equal_distribution(self, df, total_rows):
        """
        Samples an equal distribution of rows based on the 'same' column.

        Parameters:
        - df: DataFrame containing the data.
        - total_rows: Total number of rows to sample

        Returns:
        - A sampled DataFrame with equal number of 'same' True and False values.
        """
        if total_rows % 2 != 0:
        total_rows += 1
        print(f"Adjusted total rows to the next ever number: {total_rows}")
    
        # Filter rows where 'same' is True and sample
        same_true_sample = df[df['same'] == True].sample(n=total_rows // 2, replace=True)
    
        # Filter rows where 'same' is False and sample
        same_false_sample = df[df['same'] == False].sample(n=total_rows // 2, replace=True)
    
        # Concatenate and shuffle the samples
        sample_df = pd.concat([same_true_sample,same_false_sample]).sample(frac=1)
        sample_df = sample_df.sort_values(by='id').reset_index(drop=True)
    
        return sample_df

    def filter_ids_from_df(df, truth_sample):
        """
        Filters rows in a DataFrame based on the 'id' values in truth_sample.

        Parameters:
        - df: Original DataFrame
        - truth_sample: DataFrame containing the 'id' values to filter.

        Returns:
        - Filtered DataFrame.
        """

        filtered_df = df[df['id'].isin(truth_sample['id'])]
        filtered_df - filtered_df.sort_values(by='id').reset_index(drop=True)
        
        return filtered_df

    def split_sentences(text):
        """
        Splits a text string into sentences.
    
        Parameters:
        - text: The input text string.
    
        Returns:
        - A list of sentences.
        """
    
        # Remove all single and double quotation marks from the string.
        text = re.sub(r'[\'"]', '', text)
        
        # Split the string at sentence-ending punctuation marks
        pattern = r'(?<=[.!?]) +'
        sentences = re.split(pattern, text)
        
        # Strip leading and trailing whitespace from each sentence
        sentences = [sentence.strip() for sentence in sentences]
        
        return sentences

    def count_words(text):
        """Counts the words in a text string"""
        
        return len(text.split())

    def apply_sentence_split(df, input_col='text', output_col='text'):
        """
        Splits sentences in each row of the DataFrame and restructures the DataFrame.
    
        Parameters:
        - df: DataFrame with text data.
        - input_col: Column name with the original text.
        - output_col: Column name for the output split text.
    
        Returns:
        - Expanded DataFrame with split sentences.
        """
    
        # Apply the sentence split function across the user selected input column
        df['sentence'] = df[input_col].apply(split_sentences)
    
        # Explode the list of sentences creating a row per sentence then remove the column.
        df_expanded = df.explode('sentence').reset_index(drop=True)
        df_expanded = df_expanded.drop(columns=input_col)
        df_expanded.rename(columns={'sentence':output_col}, inplace=True)
    
        # TODO - Check if a better way to do this.
        df_expanded.insert(1, 'chunk_id', df_expanded.groupby('id').cumcount())
        df_expanded.insert(4, 'word_count', df_expanded[output_col].apply(count_words))
    
        # Remove any empty sentences.
        df_expanded = df_expanded[df_expanded['word_count'] >= 1]
        
        return df_expanded

    def split_rows_by_word_count(df, num_words):
        """
        Splits rows with excessive word counts in a sentence into smaller chunks.
    
        Parameters:
        - df: DataFrame with text data.
    
        Returns:
        - DataFrame with rows split into word count-complient chunks.
        """
        
        max_words = df['word_count'].max()
        
        if max_words >= num_words:
        
            new_rows = [] 
            drop_indices = []
            chunk_counter = {}  # Dictionary to keep track of chunk counts for each chunk_id
        
            # Iterate over the DataFrame
            for index, row in df.iterrows():
                if row['word_count'] > num_words:
                    words = row['text'].split()  # Split the text into words
                    num_chunks = len(words) // num_words  # Calculate the number of chunks
                    if len(words) % num_words != 0:
                        num_chunks += 1
                    
                    chunk_id = row['chunk_id']
                    if chunk_id not in chunk_counter:
                        chunk_counter[chunk_id] = 1
                    
                    for i in range(num_chunks):
                        start_idx = i * num_words
                        end_idx = min((i + 1) * num_words, len(words))
                        
                        new_row = row.copy()  # Create a copy of the original row
                        new_row['text'] = ' '.join(words[start_idx:end_idx])
                        new_row['subchunk_id'] = chunk_counter[chunk_id]
                        
                        new_rows.append(new_row)  # Append the new row to the list
                        chunk_counter[chunk_id] += 1
        
                    drop_indices.append(index)
        
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
    
        updated_df.reset_index(inplace=True)
        updated_df.sort_values(by=['index', 'chunk_id', 'subchunk_id'], inplace=True)
        updated_df.reset_index(drop=True, inplace=True)  # Reset the index to create a new one
    
        df_text = updated_df.pop('text')
        updated_df['text'] = df_text
        
    
        return updated_df