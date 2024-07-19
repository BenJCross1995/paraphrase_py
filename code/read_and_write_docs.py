# -*- coding: utf-8 -*-
import json
import os
import uuid
import pandas as pd

from datetime import datetime

def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
            
    data = pd.DataFrame(data)
    return data

def save_as_jsonl(data, output_file_path):
    with open(output_file_path, 'w') as file:
        for _, row in data.iterrows():
            json.dump(row.to_dict(), file)
            file.write('\n')

def save_error_as_txt(data, folder_path):
    """
    Saves the given content to a .txt file in the specified folder path.
    The filename is generated using a timestamp and UUID to ensure uniqueness.
    
    Parameters:
    - content: The content (string) to be saved.
    - folder_path: The folder path where the file should be saved.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]  # Generate a unique ID
    filename = f"error_{timestamp}_{unique_id}.txt"
    file_path = os.path.join(folder_path, filename)
    
    with open(file_path, 'w') as file:
        file.write(data)