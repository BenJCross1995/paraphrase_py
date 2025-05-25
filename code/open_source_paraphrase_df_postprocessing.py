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

def process_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take in a DataFrame with a 'generated_text' column,
    apply cleaning functions in sequence, and return
    a new DataFrame enriched with clean_text, text_cleaned,
    clean_stage, and parsing_errors columns.
    """
    funcs = [
        ("clean_markdown", clean_markdown),
    ]

    records = []
    for record in df.to_dict(orient='records'):
        gen = record.get("generated_text", "")
        cleaned = None
        stage = None
        errors = []

        for name, func in funcs:
            result, err = func(gen)
            if result is not None:
                cleaned = json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
                stage = name
                break
            if err:
                errors.append(f"{name}: {err}")

        text_cleaned = 1 if cleaned is not None else 0
        if not cleaned:
            cleaned = gen
            stage = "none"

        record.update({
            "clean_text": cleaned,
            "text_cleaned": text_cleaned,
            "clean_stage": stage,
            "parsing_errors": errors,
        })
        records.append(record)

    return pd.DataFrame.from_records(records)


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