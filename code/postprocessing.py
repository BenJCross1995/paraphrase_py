import re, json, ast, argparse, pandas as pd
from typing import List, Tuple, Any
from read_and_write_docs import read_jsonl, write_jsonl

# ── 1.1  strip ```markdown fences ────────────────────────────────
def fix_markdown(text: str) -> Tuple[str, bool]:
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    return cleaned, cleaned != text

# ── 1.2  remove // and /* */ comments ────────────────────────────
def fix_comments(text: str) -> Tuple[str, bool]:
    cleaned = re.sub(r"//.*?(?=\n|$)", "", text)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.S)
    return cleaned, cleaned != text

# ── 1.3  drop trailing commas before } or ] ──────────────────────
def fix_trailing_commas(text: str) -> Tuple[str, bool]:
    cleaned = re.sub(r",\s*(?=[}\]])", "", text)
    return cleaned, cleaned != text

# ── 1.4  add one missing closing brace / bracket ────────────────
def fix_outer_brace(text: str) -> Tuple[str, bool]:
    t = text.strip()
    if t.startswith("{") and not t.rstrip().endswith("}"):
        return t + "}", True
    if t.startswith("[") and not t.rstrip().endswith("]"):
        return t + "]", True
    return text, False

# ── 1.5  pythonic → canonical JSON via ast.literal_eval ──────────
def fix_pythonic(text: str) -> Tuple[str, bool]:
    try:
        obj = ast.literal_eval(text)
        return json.dumps(obj, ensure_ascii=False), True
    except Exception:
        return text, False

# ── 1.6  wrap: wrap a string in valid JSON ───────────────
def wrap_plain_text(text: str) -> Tuple[str, bool]:
    """
    If the string contains NO curly/bracket braces with a colon (key:value),
    treat the whole thing as a plain answer and wrap it like
    {"new_document": "<text>"} so json.loads() will succeed.

    Returns (wrapped_text, True) only when it actually wrapped.
    """
    # quick sniff for any {...:...} or [...:...]  pattern
    if re.search(r"[{\[][^{}\[\]]*:[^{}\[\]]*[}\]]", text):
        return text, False                   # looks like JSON → skip

    wrapped = json.dumps({"new_document": text}, ensure_ascii=False)
    return wrapped, True

# -- POTENTIAL REPLACEMENT (STRICTER ON NEEDING new_document) ----
# -- NOT IMPLEMENTED ----

# compile once at module import
NEW_DOC_RE = re.compile(
    r"""            # opening {  or [
        [\{\[]\s*   #   optional whitespace
        ["']?       #   key may be quoted or bare
        new_document
        ["']?\s*:
    """,
    re.I | re.X,    # ignore case, verbose mode
)

def wrap_plain_text_stricter(text: str) -> tuple[str, bool]:
    """
    Treat as JSON-ish only when { or [ directly introduces 'new_document'.
    Otherwise wrap the whole thing as plain prose.
    """
    if NEW_DOC_RE.search(text):
        return text, False        # leave it for earlier fixers / json.loads

    wrapped = json.dumps({"new_document": text}, ensure_ascii=False)
    return wrapped, True

# ── 1.7  salvage: join all double-quoted strings ────────────────
def fix_salvage_quotes(text: str) -> Tuple[str, bool]:
    parts = re.findall(r'"((?:\\.|[^"\\])*)"', text)
    if not parts:
        return text, False
    if parts[0] == "new_document":
        parts = parts[1:]
    combined = "\n\n".join(parts)
    return json.dumps({"new_document": combined}, ensure_ascii=False), True

FIXERS: List[Tuple[str, Any]] = [
    ("fix_markdown",        fix_markdown),
    ("fix_comments",        fix_comments),
    ("fix_trailing_commas", fix_trailing_commas),
    ("fix_outer_brace",     fix_outer_brace),
    ("fix_pythonic",        fix_pythonic),
    ("wrap_plain_text", wrap_plain_text),
    ("fix_salvage_quotes",  fix_salvage_quotes),  # last-chance
]
    
# ── Code to print a summary ────────────────
def print_summary(df: pd.DataFrame,
                  fixers = FIXERS,
                  flag: str = "clean_stage") -> None:
    """
    Show overall success + per-stage counts in pipeline order.
    """
    total = len(df)
    cleaned = df[df[flag] != "none"]
    n_clean = len(cleaned)
    pct_clean = n_clean / total * 100 if total else 0.0

    print(f"\nOut of {total:,} total rows, {n_clean:,} are cleaned "
          f"({pct_clean:0.2f}%).\n")

    # Build the display order: original_ok → every fixer in FIXERS
    stage_order = ["original_ok"] + [name for name, _ in fixers]

    # Count rows per stage
    counts = cleaned[flag].value_counts()

    print(f"Out of {n_clean:,} cleaned rows, {flag} breakdown:")
    for stage in stage_order:
        if stage in counts:
            n = counts[stage]
            pct = n / n_clean * 100
            print(f"  Stage {stage}: {n:,} rows ({pct:0.2f}%)")
    print()
    
def process_records(df: pd.DataFrame,
                    fixers=FIXERS,
                    verbose: bool = False) -> pd.DataFrame:
    """
    After each single fix, attempt json.loads().
    Stop at the first success; otherwise keep errors.
    Adds columns: clean_text, text_cleaned, clean_stage, parsing_errors
    """
    rows_out: list[dict] = []
    skipped_rows: list[int] = []  

    for idx, rec in enumerate(df.to_dict(orient="records")):
        raw_text       = rec.get("generated_text", "")
        current_text   = raw_text
        parsing_errors = []
        clean_stage    = "none"
        text_cleaned   = 0

        if not isinstance(raw_text, str):
            skipped_rows.append(idx)
            if verbose:
                print(f"[{idx:>5}] ✘  skipped (non_string_input)")
            continue   # do NOT append this row to rows_out
            
        # -------------------------------------------------- 0) try untouched
        try:
            obj = json.loads(current_text)
            clean_stage  = "original_ok"
            text_cleaned = 1
        except json.JSONDecodeError as e:
            parsing_errors.append(f"original: {e}")
            obj = None

        # -------------------------------------------------- 1…N) fixes
        if obj is None:
            for stage, fixer in fixers:
                current_text, changed = fixer(current_text)

                if changed:
                    text_cleaned = 1  # we definitely modified something

                try:
                    obj = json.loads(current_text)
                    clean_stage = stage
                    break                      # success!
                except json.JSONDecodeError as e:
                    parsing_errors.append(f"{stage}: {e}")

        # -------------------------------------------------- final payload
        if obj is not None and isinstance(obj, dict):
            clean_text = obj.get("new_document", obj)
        else:
            # fall back to whatever text we ended with
            clean_text = current_text

        # ensure string for writing
        if not isinstance(clean_text, str):
            clean_text = json.dumps(clean_text, ensure_ascii=False)

        rec.update({
            "clean_text"    : clean_text,
            "text_cleaned"  : text_cleaned,
            "clean_stage"   : clean_stage,
            "parsing_errors": parsing_errors,
        })
        rows_out.append(rec)

        if verbose:
            tick = "✔︎" if clean_stage != "none" else "✘"
            print(f"[{idx:>5}] {tick}  stage={clean_stage}")
            
    if verbose and skipped_rows:
        print(f"\n␡  Skipped {len(skipped_rows)} non-string row(s). "
              f"{skipped_rows}")
        
    return pd.DataFrame(rows_out)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Iteratively clean LLM JSONL outputs"
    )
    p.add_argument("--input_loc",  required=True)
    p.add_argument("--output_loc", required=True)
    p.add_argument("--verbose",    action="store_true",
                   help="print per-row success info")
    args = p.parse_args()

    df_in  = read_jsonl(args.input_loc)
    df_out = process_records(df_in, verbose=args.verbose)
    write_jsonl(df_out, args.output_loc)

if __name__ == "__main__":
    main()
