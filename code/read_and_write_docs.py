# -*- coding: utf-8 -*-
import json
import pandas as pd

def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
            
    data = pd.DataFrame(data)
    return data

def save_as_jsonl(data, output_file_path):
    with open(file_path, 'w') as file:
        for _, row in data.iterrows():
            json.dump(row.to_dict(), file)
            file.write('\n')
