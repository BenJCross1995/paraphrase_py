import argparse
import read_and_write_docs
import random
import time

import pandas as pd

def filter_chunked_unknown(df):

    id_list = df['id'].unique()
    result_df = pd.DataFrame()
    
    for doc_id in id_list:

        # Filter the dataframe for the current id value, resetting the index
        filtered_df = df[df['id'] == doc_id].copy().reset_index(drop=True) 

        # Add a couple of columns to filter on. The idea is that if we're on row 180 for a 
        # particular doc which has 190 rows. Then the number of chunks must be less than 11
        # as 180 is inclusive. If not we say we won't keep it later on.
        filtered_df['row_number'] = filtered_df.index + 1
        filtered_df['num_rows'] = len(filtered_df)
        filtered_df['end_row'] = filtered_df['row_number'] + filtered_df['chunk_count'] - 1
        filtered_df['keep'] = filtered_df['end_row'] <= filtered_df['num_rows']

        result_df = pd.concat([result_df, filtered_df], ignore_index=True)

    result_df = result_df.drop(columns=['row_number', 'end_row', 'num_rows'])
        
    return result_df

def chunk_rephrased(unknown, rephrased, num_impostors=10):

    # We only want to apply the algorithm on only docs we have rephrased 
    rephrased_docs = rephrased['doc_id'].unique()
    unknown = unknown[unknown['id'].isin(rephrased_docs)]
    data = []

    # We want to loop across the rows in the unknown df
    for i in range(len(unknown)):

        # Keep the logged data and if the data is not 'kept' then skip the row
        # This is to make sure no rows are pulling sentences from other docs
        keep = unknown.iloc[i, unknown.columns.get_loc('keep')]
        if keep == False:
            continue

        # Keep the variables for each row that matter
        doc_id = unknown.iloc[i, unknown.columns.get_loc('id')]
        chunk_id = unknown.iloc[i, unknown.columns.get_loc('chunk_id')]
        subchunk_id = unknown.iloc[i, unknown.columns.get_loc('subchunk_id')]

        # Get the number of chunks in the new unknown text and filter the unknown
        # data to include those chunks.
        chunk_count = unknown.iloc[i, unknown.columns.get_loc('chunk_count')]
        filtered_unknown = unknown.iloc[i:i+chunk_count,]
        
        sentence_data = []

        # Loop however many times the user desires
        for _ in range(num_impostors):
            
            sentences = []

            # Want to loop through the rows in the filtered unknown dataframe
            for index, row in filtered_unknown.iterrows():

                # Get the variables to filter the rephrased df for current row of unknown df
                id_value = row['id']
                chunk_id_value = row['chunk_id']
                subchunk_id_value = row['subchunk_id']
                original_sentence = row['original_sentence']
            
                filtered_rephrased = rephrased[
                    (rephrased['doc_id'] == id_value) & 
                    (rephrased['chunk_id'] == chunk_id_value) & 
                    (rephrased['subchunk_id'] == subchunk_id_value)
                ]

                # Ensure rephrased_list contains only strings of paraphrases and add original sentence
                # We add the original sentence incase no rephrases were found we wont skip the chunk.
                rephrased_list = filtered_rephrased['rephrased'].tolist()
                rephrased_list = [str(item) for item in rephrased_list]
                rephrased_list.append(original_sentence)
            
                # Remove duplicates by converting to a set and back to a list
                distinct_list = list(set(rephrased_list))

                # select a random sentence and add to a list
                sample_sentence = random.choice(distinct_list)
                sentences.append(sample_sentence)

            # Convert to a paragraph by joining sentences together
            paragraph = " ".join(sentences)
            
            sentence_data.append({
                'doc_id': doc_id,
                'chunk_id': chunk_id,
                'subchunk_id': subchunk_id,
                'rephrased': paragraph
            })
    
        data.extend(sentence_data)
    
    result_df = pd.DataFrame(data)

    return result_df

def main():
    """Main function to parse arguments and process the input file.
    
    Parses command line arguments, reads the input file, processes the sentences,
    and saves the output to the specified file path.
    """

    start_time = time.time()
	
    # Parse arguments from user
    parser = argparse.ArgumentParser(description='Chunk sentences together into paragraphs from paraphrased sentences')
    parser.add_argument('--unknown_file_path', type=str, help='Path to unknown docs jsonl file', required=True)
    parser.add_argument('--rephrased_file_path', type=str, help='Path to rephrased docs jsonl file', required=True)
    parser.add_argument('--output_file_path', type=str, help='Output filepath', required=True)
    parser.add_argument('--num_impostors', type=int, default=10, help='The number of impostors for each sentence.')
    args = parser.parse_args()

    # Pull in the unknown and rephrased docs
    unknown = read_and_write_docs.read_jsonl_file(args.unknown_file_path)
    rephrased = read_and_write_docs.read_jsonl_file(args.rephrased_file_path)

    # Log each doc in the unknown whether to keep or remove
    print("Logging which sentences to keep based on chunks to threshold...")
    start_log_time = time.time()
    logged_unknown = filter_chunked_unknown(unknown)
    end_log_time = time.time()
    print(f"Time taken to log sentences: {end_log_time - start_log_time:.2f} seconds")

    # Run the function to chunk the rephrased docs
    print("Chunking the rephrased data...")
    start_chunk_time = time.time()
    result = chunk_rephrased(logged_unknown, rephrased, args.num_impostors)
    end_chunk_time = time.time()
    print(f"Time taken to chunk rephrased data: {end_chunk_time - start_chunk_time:.2f} seconds")

    read_and_write_docs.save_as_jsonl(result, args.output_file_path)

    print("Rephrasing complete!")
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()