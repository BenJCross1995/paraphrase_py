#!/usr/bin/env python
import argparse
import os
import json
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_hf_token(cli_token: str = None, creds_path: str = "credentials.json") -> str:
    """
    Retrieve the Hugging Face token from:
      1. Command-line argument (--token)
      2. Environment variable HF_TOKEN
      3. A JSON file (default: credentials.json) with key 'hf_token' or 'huggingface_token'
    """
    if cli_token:
        return cli_token

    env_token = os.getenv("HF_TOKEN")
    if env_token:
        return env_token

    if os.path.exists(creds_path):
        with open(creds_path, "r") as f:
            creds = json.load(f)
        file_token = creds.get("hf_token") or creds.get("huggingface_token")
        if file_token:
            return file_token

    raise ValueError("No Hugging Face token found. Provide one via --token, set the HF_TOKEN environment variable, or store it in credentials.json.")

def main():
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model locally after logging in using a token from an argument, environment variable, or a JSON file."
    )
    parser.add_argument("--model_id", type=str, required=True,
                        help="Hugging Face model identifier (e.g., 'meta-llama/Llama-3.1-8B').")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Local directory where the model and tokenizer will be saved.")
    parser.add_argument("--token", type=str, default=None,
                        help="Optional Hugging Face token. If not provided, the script checks the HF_TOKEN environment variable or credentials.json.")
    parser.add_argument("--creds_path", type=str, default="credentials.json",
                        help="Path to the credentials JSON file (default: credentials.json).")

    args = parser.parse_args()

    # Retrieve token and log in to Hugging Face Hub.
    hf_token = get_hf_token(args.token, args.creds_path)
    login(token=hf_token)
    print("Successfully logged in to Hugging Face Hub.")

    # Ensure the save directory exists.
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print(f"Downloading model and tokenizer for '{args.model_id}'...")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    print(f"Saving model and tokenizer to '{args.save_path}'...")
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print("Download and save complete.")

if __name__ == "__main__":
    main()
