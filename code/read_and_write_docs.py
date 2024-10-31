# -*- coding: utf-8 -*-
import json
import os
import uuid
import pandas as pd

from datetime import datetime

def read_jsonl(file_path):
    """
    Reads a JSONL file and converts it into a pandas DataFrame.

    Parameters:
    - file_path: Path to the JSON file to read.

    Returns:
    - A pandas Dataframe containing the data from the JSONL file.
    """
    
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
            
    data = pd.DataFrame(data)
    return data

def write_jsonl(data, output_file_path):
    """
    Writes a pandas DataFrame to a JSONL file.

    Parameters:
    - data: A pandas DataFrame to save.
    - output_file_path: Path to the output JSONL file.
    """
    
    with open(output_file_path, 'w') as file:
        for _, row in data.iterrows():
            json.dump(row.to_dict(), file)
            file.write('\n')

def save_error_as_txt(data, folder_path):
    """
    Saves error data to a folder path.
    
    Parameters:
    - data: The pandas DataFrame to save.
    - folder_path: The folder path where the file should be saved.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]  # Generate a unique ID
    filename = f"error_{timestamp}_{unique_id}.txt"
    file_path = os.path.join(folder_path, filename)
    
    with open(file_path, 'w') as file:
        file.write(data)