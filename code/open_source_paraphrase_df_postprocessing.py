import argparse
import re
import json
from read_and_write_docs import read_jsonl, write_jsonl
import pandas as pd

def clean_markdown(raw: str) -> tuple:
    """
    Strip markdown code fences and attempt to load JSON.
    Returns (parsed_result or raw string, None) on success,
    or (None, error_message) on JSON decode failure.
    """
    clean = raw.strip()
    clean = re.sub(r"^```(?:json)?\s*\n?", "", clean)
    clean = re.sub(r"\n?```$", "", clean).strip()

    try:
        data = json.loads(clean)
        return data.get("new_document", clean), None
    except json.JSONDecodeError as e:
        return None, str(e)


    return data.get("new_document", clean)

def check_no_json(raw: str) -> tuple:
    """
    Detect plain text lacking the 'new_document' key and JSON structure.
    Strips markdown fences before checking. If the cleaned text does not
    contain the literal '"new_document"' and also lacks a JSON-like key:value
    pattern inside braces/brackets, returns (raw, None) to mark as plaintext.
    Otherwise returns (None, None) to continue parsing.
    """
    # Strip markdown code fences
    clean = raw.strip()
    clean = re.sub(r"^```(?:json)?\s*\n?", "", clean)
    clean = re.sub(r"\n?```$", "", clean).strip()

    # If it explicitly contains the new_document key, treat as JSON
    if '"new_document"' in clean:
        return None, None

    # Otherwise, if there's no {…:…} or […:…] pattern, treat as plaintext
    if not re.search(r"[{\[](?:[^:\[\]{}]*:[^\[\]{}]*)[}\]]", clean, re.DOTALL):
        return raw, None

    # Otherwise, fall through to the JSON parsing chain
    return None, None

def fix_trailing_commas(raw: str) -> tuple:
    """
    Remove trailing commas before closing braces/brackets to make JSON valid.
    Returns (parsed_data, None) or (None, error_message).
    """
    fixed = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        data = json.loads(fixed)
        return data, None
    except json.JSONDecodeError as e:
        return None, str(e)

def extract_new_document(raw: str) -> tuple:
    """
    Fallback: manually extract all quoted string segments and
    stitch them into one 'new_document' value.
    """
    clean = raw.strip()
    clean = re.sub(r"^```(?:json)?\s*\n?", "", clean)
    clean = re.sub(r"\n?```$", "", clean).strip()

    # find all double-quoted substrings
    parts = re.findall(r'"((?:\\.|[^"\\])*)"', clean)
    if parts:
        # skip the first part if it's the key name
        text_parts = parts[1::2] if parts and parts[0] == 'new_document' else parts
        combined = "\n\n".join(text_parts)
        return combined, None
    return None, "no quoted segments found"

def process_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process 'generated_text' through a chain of cleaners.
    Returns DataFrame with clean_text, text_cleaned, clean_stage, parsing_errors.
    """
    funcs = [
        ("check_no_json", check_no_json),
        ("clean_markdown", clean_markdown),
        ("extract_new_document", extract_new_document),
        ("fix_trailing_commas", fix_trailing_commas),
    ]

    output = []
    for rec in df.to_dict(orient='records'):
        raw = rec.get('generated_text', '')
        if rec.get('already_clean', False):
            rec.update({
                'clean_text': raw,
                'text_cleaned': 1,
                'clean_stage': 'already_clean',
                'parsing_errors': []
            })
            output.append(rec)
            continue

        cleaned = None
        stage = None
        errors = []

        for name, fn in funcs:
            result, err = fn(raw)
            if err:
                errors.append(f"{name}: {err}")
            if result is not None:
                cleaned = json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
                stage = name
                break

        # ensure text_cleaned matches stage
        if stage and stage != 'none':
            text_cleaned = 1
        else:
            cleaned = raw
            text_cleaned = 0
            stage = 'none'

        rec.update({
            'clean_text': cleaned,
            'text_cleaned': text_cleaned,
            'clean_stage': stage,
            'parsing_errors': errors
        })
        output.append(rec)

    return pd.DataFrame.from_records(output)


def main():
    parser = argparse.ArgumentParser(description="Postprocess LLM JSONL outputs to ensure valid JSON and extract clean_text.")
    parser.add_argument("--input_loc", required=True, help="Path to input JSONL file.")
    parser.add_argument("--output_loc", required=True, help="Path to output JSONL file.")
    
    args = parser.parse_args()
    
    df = read_jsonl(args.input_loc)
    processed_df = process_records(df)
    write_jsonl(processed_df, args.output_loc)


if __name__ == "__main__":
    main()