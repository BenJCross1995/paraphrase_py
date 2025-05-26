from read_and_write_docs import read_jsonl, write_jsonl
from typing import Tuple, Any

import argparse
import re
import json
import ast

import pandas as pd

def check_no_json(raw: str) -> Tuple[Any | None, str | None]:
    """
    Quick test for *already-clean* plain-text answers.

    Returns
    -------
    (raw, None)
        when the string looks like plain text (no JSON braces *with* colons).

    (None, None)
        when the string *might* be JSON – let the next cleaner decide.

    (None, error_message)
        never happens in this function, but keeps the common signature.
    """
    # 1)  Remove ``` fences if the LLM used a code block
    clean = raw.strip()
    clean = re.sub(r"^```(?:json)?\s*\n?", "", clean)
    clean = re.sub(r"\n?```$", "", clean).strip()

    # 2)  If the payload can be parsed by json.loads it is JSON – defer
    try:
        json.loads(clean)
        return None, None
    except json.JSONDecodeError:
        pass  # not valid JSON – may still *contain* braces, keep checking

    # 3)  Look for a *key : value* pattern inside braces or brackets
    #     e.g. {"foo": "bar"}  or  [ {"foo": 1}, ... ]
    json_like = re.search(r"[{\[][^{}\[\]]*:[^{}\[\]]*[}\]]", clean, re.S)

    if json_like:
        # Might be malformed JSON – leave it for the other cleaners
        return None, None

    # 4)  Otherwise we treat it as ready-to-use plain text
    return raw, None


def clean_markdown(raw: str) -> Tuple[Any, str | None]:
    """
    Remove ``` fences (optionally ```json) and try json.loads().
    """
    clean = raw.strip()
    clean = re.sub(r"^```(?:json)?\s*\n?", "", clean)
    clean = re.sub(r"\n?```$", "", clean).strip()
    try:
        data = json.loads(clean)
        return data.get("new_document", data), None
    except json.JSONDecodeError as e:
        return None, str(e)
    

def strip_json_fragment(raw: str) -> Tuple[Any, str | None]:
    """
    Find the *first* balanced {...} or [...] block and load it.
    Handles “Here is your JSON:” style prefixes/suffixes.
    """
    start_obj, start_arr = raw.find("{"), raw.find("[")
    if start_obj == start_arr == -1:
        return None, "No JSON brackets found"

    start = min(filter(lambda x: x != -1, [start_obj, start_arr]))
    stack = []
    for i, ch in enumerate(raw[start:], start=start):
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:                       # premature close
                return None, "Unbalanced brackets"
            opener = stack.pop()
            if (opener == "{" and ch != "}") or (opener == "[" and ch != "]"):
                return None, "Mismatched brackets"
            if not stack:                      # outermost closed
                frag = raw[start:i + 1]
                try:
                    data = json.loads(frag)
                    return data.get("new_document", data), None
                except json.JSONDecodeError as e:
                    return None, str(e)
    return None, "Unterminated JSON fragment"

def remove_json_comments(raw: str) -> Tuple[Any, str | None]:
    """
    Strip // line-comments and /* block-comments */ then load.
    """
    cleaned = re.sub(r"//.*?(?=\n|$)", "", raw)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.S)
    try:
        data = json.loads(cleaned)
        return data.get("new_document", data), None
    except json.JSONDecodeError as e:
        return None, str(e)
    
def remove_trailing_commas(raw: str) -> Tuple[Any, str | None]:
    """
    Delete trailing commas before } or ], then load.
    """
    fixed = re.sub(r",\s*(?=[}\]])", "", raw)
    try:
        data = json.loads(fixed)
        return data.get("new_document", data), None
    except json.JSONDecodeError as e:
        return None, str(e)
    
def patch_outer_brace(raw: str):
    """
    1 Trim whitespace + ```fences.
    2 If the *outermost* brace is missing its closer, append it.
    3 Try `json.loads` and return the result on success.
    """
    t = raw.strip()
    # Already balanced?   Nothing to patch
    if t.count("{") == t.count("}") and t.count("[") == t.count("]"):
        return None, None

    # { …   but no } —> add one
    if t.startswith("{") and not t.rstrip().endswith("}"):
        candidate = re.sub(r",\s*$", "", t) + "}"
    # [ …   but no ] —> add one
    elif t.startswith("[") and not t.rstrip().endswith("]"):
        candidate = re.sub(r",\s*$", "", t) + "]"
    else:
        return None, None             # either not JSON-like or too broken

    try:
        data = json.loads(candidate)
        return data.get("new_document", data), None
    except json.JSONDecodeError as e:
        return None, f"after patch_outer_brace: {e}"

    
def parse_pythonic_json(raw: str) -> Tuple[Any, str | None]:
    """
    Last-chance parser using `ast.literal_eval` so it tolerates:
        • single quotes
        • unquoted keys
        • Python literals (True/False/None)
        • trailing commas
    It then re-serialises via json.dumps for canonical JSON.
    SECURITY: *only* call this on trusted content.
    """
    try:
        py_obj = ast.literal_eval(raw)
        canonical = json.loads(json.dumps(py_obj))   # re-parse as true JSON
        return canonical.get("new_document", canonical), None
    except Exception as e:
        return None, str(e)
    
def salvage_quoted_strings(raw: str) -> Tuple[str | None, str | None]:
    """
    Last-chance fallback: concatenate all double-quoted substrings
    (excluding the literal key "new_document" if present).
    """
    clean = raw.strip()
    clean = re.sub(r"^```(?:json)?\s*\n?", "", clean)
    clean = re.sub(r"\n?```$", "", clean).strip()

    parts = re.findall(r'"((?:\\.|[^"\\])*)"', clean)
    if not parts:
        return None, "no quoted segments found"

    # Drop the first item if it's exactly the key name
    if parts[0] == "new_document":
        parts = parts[1:]

    combined = "\n\n".join(parts)
    return combined, None


CLEANERS = [
    ("check_no_json",          check_no_json),          # <- very strict
    ("clean_markdown",         clean_markdown),
    ("remove_json_comments",   remove_json_comments),
    ("remove_trailing_commas", remove_trailing_commas),
    ("patch_outer_brace", patch_outer_brace),
    ("strip_json_fragment",    strip_json_fragment),
    ("parse_pythonic_json",    parse_pythonic_json),    # <- most forgiving
    ("salvage_quoted_strings", salvage_quoted_strings), # <- HAIL MARY ATTEMPT - GPT
]


def process_records(df: pd.DataFrame, cleaners=CLEANERS) -> pd.DataFrame:
    """
    For every row:
        • walk through CLEANERS until one returns a non-None result
        • remember which stage succeeded and all errors along the way
    Adds/overwrites four columns:
        clean_text, text_cleaned, clean_stage, parsing_errors
    """
    processed_rows = []

    # convert DataFrame to a list of plain dicts (easiest to mutate)
    for rec in df.to_dict(orient="records"):

        original_text = rec.get("generated_text", "")
        already_ok    = rec.get("already_clean", False)

        # Default values (assume “nothing cleaned”)
        clean_text     = original_text
        text_cleaned   = 0
        clean_stage    = "none"
        parsing_errors = []

        # ----------------------------------------------------------
        # Fast-path: caller says the text is already fine
        # ----------------------------------------------------------
        if already_ok:
            clean_stage  = "already_clean"
            text_cleaned = 1

        # ----------------------------------------------------------
        # Otherwise: try each cleaner in turn
        # ----------------------------------------------------------
        else:
            for stage_name, cleaner in CLEANERS:
                try:
                    result, error = cleaner(original_text)

                    if error:                          # keep all errors
                        parsing_errors.append(f"{stage_name}: {error}")

                    if result is not None:             # success – stop here
                        clean_text = (
                            json.dumps(result, ensure_ascii=False)
                            if isinstance(result, (dict, list))
                            else str(result)
                        )
                        text_cleaned = int(clean_text != original_text)
                        clean_stage  = stage_name
                        break                           # leave the loop

                except Exception as ex:                # totally unexpected
                    parsing_errors.append(
                        f"{stage_name}: {type(ex).__name__}: {ex}"
                    )

        # ----------------------------------------------------------
        # Save the outcome back into this record
        # ----------------------------------------------------------
        rec.update({
            "clean_text"    : clean_text,
            "text_cleaned"  : text_cleaned,
            "clean_stage"   : clean_stage,
            "parsing_errors": parsing_errors,
        })
        processed_rows.append(rec)

    # Turn the list of dicts back into a DataFrame
    return pd.DataFrame(processed_rows)

def main() -> None:
    p = argparse.ArgumentParser(
        description="Clean LLM JSONL outputs and extract clean_text"
    )
    p.add_argument("--input_loc",  required=True)
    p.add_argument("--output_loc", required=True)
    args = p.parse_args()

    df_in   = read_jsonl(args.input_loc)
    df_out  = process_records(df_in)     # ← uses default CLEANERS
    write_jsonl(df_out, args.output_loc)

if __name__ == "__main__":
    main()