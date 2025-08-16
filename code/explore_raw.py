# explore_raw.py
import pandas as pd, textwrap, ast, re, json, os
from collections import Counter, defaultdict
from pathlib import Path

CSV_PATH = "extraction_table_data.csv"

# Optional: provide one of these in the same folder to map file -> human title
TITLE_MAP_CSV = "paper_titles.csv"  # columns: file,title
TITLE_MAP_JSON = "paper_titles.json"  # {"file.pdf": "Title", ...}

cols = [
    "metric",
    "domain",
    "major_consumers",
    "evaluation_type",
    "energy_model",
    "techniques",
    "qa_tradeoff",
]

WS_RE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    s = str(s).strip().lower()
    s = WS_RE.sub(" ", s)
    if (s.startswith("'") and s.endswith("'")) or (
        s.startswith('"') and s.endswith('"')
    ):
        s = s[1:-1].strip()
    return s


def parse_values(cell):
    """Turn a cell into a list of atomic, normalized tokens.
    Handles python-list-like strings and comma-delimited fallbacks."""
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    # Try python list literal
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple, set)):
                return [_normalize(x) for x in parsed if str(x).strip()]
        except Exception:
            pass
    # Fallback: comma split (avoid URLs)
    if "," in s and not any(u in s for u in ("http://", "https://")):
        return [_normalize(x) for x in s.split(",") if x.strip()]
    # Single value
    return [_normalize(s)] if s else []


def load_title_map():
    """Load optional file->title mapping."""
    # JSON first
    pjson = Path(TITLE_MAP_JSON)
    if pjson.exists():
        try:
            m = json.loads(pjson.read_text())
            if isinstance(m, dict):
                return {
                    str(k).strip(): str(v).strip()
                    for k, v in m.items()
                    if str(k).strip()
                }
        except Exception:
            pass
    # CSV fallback
    pcsv = Path(TITLE_MAP_CSV)
    if pcsv.exists():
        try:
            dfm = pd.read_csv(pcsv)
            # be lenient about column names
            cols_lower = {c.lower(): c for c in dfm.columns}
            if "file" in cols_lower and "title" in cols_lower:
                return {
                    str(r[cols_lower["file"]])
                    .strip(): str(r[cols_lower["title"]])
                    .strip()
                    for _, r in dfm.iterrows()
                    if str(r[cols_lower["file"]]).strip()
                }
        except Exception:
            pass
    return {}


def prettify_from_filename(fname: str) -> str:
    """Readable fallback title from filename."""
    base = os.path.basename(fname)
    # strip extension
    if "." in base:
        base = ".".join(base.split(".")[:-1]) or base
    # replace underscores/dashes with space, collapse ws, title-case
    base = base.replace("_", " ").replace("-", " ")
    base = WS_RE.sub(" ", base).strip()
    # Keep acronyms; title() is fine for a fallback
    return base.title() if base else fname


def main():
    df = pd.read_csv(CSV_PATH)

    # Build title map (optional; fallback to filename prettifier)
    title_map = load_title_map()

    for c in cols:
        if c not in df.columns:
            continue

        # Tokenize the column
        tokens_per_row = df[c].apply(parse_values)
        # Count how many rows contain each token
        all_tokens = [tok for toks in tokens_per_row for tok in toks]
        value_counts = Counter(all_tokens)

        # Map token -> unique supporting files
        token_to_files = defaultdict(set)
        for i, toks in enumerate(tokens_per_row):
            f = str(df.at[i, "file"]).strip() if "file" in df.columns else ""
            for t in toks:
                if f:
                    token_to_files[t].add(f)

        # Header
        print(f"\n=== {c.upper()} ({len(value_counts)} unique) ===")

        # Sort by frequency desc, then value
        for val, freq in sorted(value_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            files = sorted(token_to_files.get(val, set()))
            # Build titles list (use mapping if present, else prettify filename)
            titles = [title_map.get(f, prettify_from_filename(f)) for f in files]

            # Line for the value
            print(f"- {val}  [rows: {freq}, papers: {len(files)}]")

            if titles:
                # Print titles (and show filename afterward for traceability)
                for t, f in zip(titles, files):
                    line = f"    • {t}  ({f})"
                    print(textwrap.fill(line, width=100, subsequent_indent="      "))
            else:
                print("    • (no supporting papers found)")


if __name__ == "__main__":
    main()
