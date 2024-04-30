"""Code to Paraphrase Sentences in a Dataframe using an LLM"""

import argparse
import json
import os
import read_and_write_docs

import pandas as pd

from openai._client import OpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# Initialise OpenAI and create the function to get the results 
os.environ["OPENAI_API_KEY"] = json.load(open("../credentials.json"))['OPENAI_API_KEY']

# Define the chat GPT model
def gpt(prompt_input, model="gpt-3.5-turbo-1106"):

    client=OpenAI()

    messages = [
        {"role": "system", "content": prompt_input.messages[0].content},
        {"role": "user", "content": prompt_input.messages[1].content},
        {"role": "assistant", "content": prompt_input.messages[2].content},
        {"role": "user", "content": prompt_input.messages[-1].content}
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )

    answer = completion.choices[0].message.content

    return answer

system_prompt = """
Given the sentence, generate as many paraphrased sentences as possible while preserving the original semantic meaning. \
Return the rephrased sentences in a JSON format. Do not return the properties of the output.
Try to return at least 20 different paraphrases in the output.
"""

examples = [
    {
        "sentence": "Known for being very delicate, the skill could take a lifetime to master.",
        "rephrased": [
            "The skill is well known for its delicacy and could require a lifetime to perfect.",
            "The skill's reputation for delicateness suggests that it could take a whole lifetime to master.",
            "It may take a lifetime to master the skill, which is renowned for its delicacy.",
            "The delicacy of the skill means it could take a lifetime to master."
        ]
    }
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{sentence}"),
        ("ai", "{rephrased}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        few_shot_prompt,
        ("human", "{sentence}"),
    ]
)

chain = final_prompt | gpt

def paraphrase_sentences(sentence, n_runs, chain=chain):
    
    # Initialise the sentence list
    sentences = [sentence]
    new_sentence_amount = 1
    
    for i in range(1, n_runs + 1):
        
        print(f'  Iteration: {i}')
        # Break if no new sentences added in the previous loop
        if new_sentence_amount == 0:
            break
            
        while True:
            try:
                output_str = chain.invoke({"sentence":sentence})
                output_list = json.loads(output_str)
                break
            except:
                pass
        
        new_sentence_amount = 0
        
        for output in output_list:
        
            if output not in sentences:
                sentences.append(output)
                new_sentence_amount += 1
            
    return sentences  

def paraphrase_df(df, *args):
    doc_ids = []
    chunks = []
    rephrased_sentences = []
    
    n_rows = df['id'].count()

    for index, row in df.iterrows():
        row_num = index + 1
        print(f'Row {row_num} out of {n_rows}')
        doc_id = row['id']
        chunk = row['chunk_id']
        sentence = row['text']
        
        rephrased = paraphrase_sentences(sentence, *args)
        num_sent = len(rephrased)
        
        # Extend lists with repeated doc_id and chunk_id
        doc_ids.extend([doc_id] * num_sent)
        chunks.extend([chunk] * num_sent)
        
        rephrased_sentences.extend(rephrased)

    # Construct DataFrame
    result_df = pd.DataFrame({
        'doc_id': doc_ids,
        'chunk_id': chunks,
        'sentence': rephrased_sentences
    })

    return result_df

def main():

    # Parse arguments from user
    parser = argparse.ArgumentParser(description='Preprocessing steps for paraphrasing using LLMs')
    parser.add_argument('--file_path', type=str, help='Path to jsonl file', required=True)
    parser.add_argument('--output_file_path', type=str, help='Output filepath', required=True)
    parser.add_argument('--n_runs', type=int, default = 5, help='The number of calls to the LLM for each sentence.')
    args = parser.parse_args()

    df = read_and_write_docs.read_jsonl_file(args.file_path)
    
    paraphrase_data = paraphrase_df(df, args.n_runs)

    read_and_write_docs.save_as_jsonl(paraphrase_data, args.output_file_path)

if __name__ == '__main__':
    main()