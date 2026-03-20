#!/usr/bin/env python3
"""
guided_topic_discovery.py — Iterative residual c-TF-IDF per category.

For each category in the data, runs sequential passes of c-TF-IDF on bigrams.
Each pass removes questions containing the dominant bigrams from the previous
pass, revealing the next most coherent theme underneath.

Output format per category:
    Pass 1: top bigrams (+ % questions containing each)
    Pass 2: top bigrams after removing pass-1 questions
    Pass 3: top bigrams after removing pass-1 and pass-2 questions
    ...

Also writes an Excel file with one sheet per category showing the
columnar format: [Pass 1 bigrams | Pass 2 bigrams | Pass 3 bigrams ...]

Usage:
    python guided_topic_discovery.py --segment girls_13_15
    python guided_topic_discovery.py --segment girls_13_15 --category kropp
    python guided_topic_discovery.py --segment all
"""

import argparse
import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    print("WARNING: openpyxl not installed — Excel output will be skipped. pip install openpyxl")

try:
    from nltk.corpus import stopwords as _nltk_stop
    NLTK_STOP = set(_nltk_stop.words("norwegian"))
except Exception:
    NLTK_STOP = set()

SCRIPT_DIR     = Path(__file__).parent
DATA_DIR       = r"C:\Users\TorsteinDeBesche\NIFU\21571 Folkehelse og livsmestring - General\Ap2c Informasjonskanal for unge\Positron_project\data"
OUTPUT_DIR     = r"C:\Users\TorsteinDeBesche\NIFU\21571 Folkehelse og livsmestring - General\Ap2c Informasjonskanal for unge\Positron_project\outputs"
SEGMENT_SUBDIR = "segments"
SEGMENTS       = ["boys_13_15", "boys_16_20", "girls_13_15", "girls_16_20"]

# Discovery settings
N_PASSES           = 5     # number of iterative passes per category
TOP_BIGRAMS_SHOWN  = 10    # bigrams to display per pass
TOP_BIGRAMS_REMOVE = 3     # top bigrams from each pass to remove for next pass
MIN_BIGRAM_FREQ    = 0.02  # minimum fraction of questions a bigram must appear in
MIN_QUESTIONS      = 50    # minimum questions remaining to continue passes

# Matches R script's custom_stop
CUSTOM_STOP = {
    "hei", "hallo", "takk", "hilsen",
    "jeg", "du", "vi", "dere", "han", "hun", "de", "den", "det", "dette",
    "ikke", "bare", "litt", "veldig", "også", "så", "å",
    "må", "kan", "vil", "skal", "burde",
    "er", "var", "blir", "ble", "har", "hadde",
    "som", "til", "for", "på", "av", "i", "om", "fra", "med", "uten",
    "når", "hva", "hvordan", "hvor", "hvorfor",
}

# Extra stopwords for bigrams only — matches R script's glue_words
# Prevents meaningless bigrams like "følelser for", "snakke om"
GLUE_WORDS = {
    "for", "om", "til", "at", "som", "å", "med", "uten", "hos", "fra",
    "på", "av", "i", "den", "det", "de", "dette", "da", "nå", "så",
    "jeg", "du", "vi", "han", "hun",
    "er", "var", "blir", "ble", "har", "hadde",
    "ikke", "bare", "veldig", "litt",
}

# Bigrams to always exclude — matches R script's ban_bigrams
BAN_BIGRAMS = {
    "hele tiden", "hele tatt", "flere ganger", "lang tid", "svar fort",
}

STOP_WORDS = sorted(NLTK_STOP | CUSTOM_STOP)


# ── Topic column join ────────────────────────────────────────────────────────

def _norm_body(series: pd.Series) -> pd.Series:
    """Normalise body text for joining — must match bottom_up_clustering.py."""
    s = (
        series.astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    s = s.str.replace(
        r"^(hei|heisann|hallo|halloi)[\s,!:.\-]+", "", regex=True
    ).str.strip()
    return s


def build_topic_lookup(data_dir: Path) -> pd.DataFrame:
    """
    Scan DATA_DIR for *.csv files and build a (body_key, topic) lookup.
    Normalises body text the same way as the age join so keys match.
    Returns DataFrame with columns [body_key, topic].
    """
    csv_files = sorted(
        p for p in data_dir.glob("*.csv")
        if p.suffix.lower() == ".csv" and p.stat().st_size > 0
    )
    if not csv_files:
        warnings.warn(f"[topic] No CSV files found in {data_dir}")
        return pd.DataFrame(columns=["body_key", "topic"])

    # Detect text column from first file
    text_col = None
    try:
        header = pd.read_csv(csv_files[0], nrows=0, encoding="utf-8-sig",
                             sep=None, engine="python")
        for candidate in ["body", "question", "text", "body_norm", "sporsmal", "tekst"]:
            if candidate in header.columns:
                text_col = candidate
                break
    except Exception as exc:
        warnings.warn(f"[topic] Could not read header: {exc}")
        return pd.DataFrame(columns=["body_key", "topic"])

    if text_col is None:
        warnings.warn("[topic] Could not identify text column")
        return pd.DataFrame(columns=["body_key", "topic"])

    try:
        header2 = pd.read_csv(csv_files[0], nrows=0, encoding="utf-8-sig",
                              sep=None, engine="python")
        if "topic" not in header2.columns:
            warnings.warn(f"[topic] No 'topic' column found. Columns: {header2.columns.tolist()}")
            return pd.DataFrame(columns=["body_key", "topic"])
    except Exception:
        pass

    parts = []
    for p in csv_files:
        try:
            chunk = pd.read_csv(
                p, usecols=[text_col, "topic"],
                dtype={text_col: str, "topic": str},
                encoding="utf-8-sig",
                low_memory=True,
            )
            parts.append(chunk)
        except Exception as exc:
            warnings.warn(f"[topic] Could not read {p.name}: {exc}")

    if not parts:
        return pd.DataFrame(columns=["body_key", "topic"])

    raw = pd.concat(parts, ignore_index=True)
    raw["body_key"] = _norm_body(raw[text_col])

    # Split comma-separated topics into separate rows (matches R separate_rows)
    # e.g. "Sex, Kropp" becomes two rows: one for "Sex", one for "Kropp"
    raw["topic"] = raw["topic"].astype(str).str.split(",")
    raw = raw.explode("topic")
    raw["topic"] = raw["topic"].str.strip()
    raw = raw[raw["topic"].str.len() > 0]
    raw = raw[raw["topic"] != "-"]
    raw = raw[raw["topic"] != "nan"]

    # Keep all (body_key, topic) pairs — one question can appear in multiple categories
    lookup = (
        raw[["body_key", "topic"]]
        .dropna(subset=["body_key", "topic"])
        .drop_duplicates(subset=["body_key", "topic"])
    )
    print(f"[topic] lookup built: {len(lookup):,} rows from {len(parts)} files")
    print(f"[topic] categories found: {sorted(lookup['topic'].unique().tolist())}")
    return lookup


def join_topic_to_assignments(assign: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Join topic column onto assignments dataframe via normalised body text."""
    lookup = build_topic_lookup(data_dir)
    if lookup.empty:
        print("[topic] WARNING: empty lookup — topic column not joined")
        assign = assign.copy()
        assign["topic"] = "unknown"
        return assign

    body_keys = _norm_body(assign["body_norm"])
    assign = assign.copy()
    assign["body_key"] = body_keys.values

    # Merge — a question with multiple topics produces multiple rows
    assign_expanded = assign.merge(lookup, on="body_key", how="left")
    assign_expanded["topic"] = assign_expanded["topic"].fillna("unknown")

    coverage = (assign_expanded["topic"] != "unknown").mean()
    print(f"[topic] join coverage: {coverage:.1%}")
    print(f"[topic] questions after expansion: {len(assign_expanded):,} "
          f"(was {len(assign):,} — {len(assign_expanded)-len(assign):,} extra rows from multi-category questions)")
    print(f"[topic] category distribution:")
    for cat, cnt in assign_expanded["topic"].value_counts().items():
        print(f"         {cat:<35} {cnt:>6,}")

    assign_expanded = assign_expanded.drop(columns=["body_key"])
    return assign_expanded


# ── c-TF-IDF ────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Match R script clean_text: lowercase, strip URLs, numbers, punctuation."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[0-9]+', ' ', text)
    text = re.sub(r'[^\w\sæøå]', ' ', text)   # keep letters incl Norwegian
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _tokenize_bigrams(text: str) -> List[str]:
    """
    Tokenize into bigrams matching R script logic:
    - both words length >= 3
    - neither word in STOP_WORDS or GLUE_WORDS
    - result bigram not in BAN_BIGRAMS
    """
    words = text.split()
    words = [w for w in words if len(w) >= 3]
    bigrams = []
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        if (w1 not in STOP_WORDS and w2 not in STOP_WORDS
                and w1 not in GLUE_WORDS and w2 not in GLUE_WORDS):
            bg = f"{w1} {w2}"
            if bg not in BAN_BIGRAMS:
                bigrams.append(bg)
    return bigrams


def compute_ctfidf_bigrams(
    texts: List[str],
    top_k: int = TOP_BIGRAMS_SHOWN,
    min_freq: float = MIN_BIGRAM_FREQ,
) -> List[Tuple[str, float, float]]:
    """
    Compute c-TF-IDF for bigrams.
    Text cleaning matches R script: lowercase, strip URLs/numbers/punctuation.
    Bigram filtering matches R script: glue_words removed, ban_bigrams excluded.

    Returns list of (bigram, ctfidf_score, freq_pct).
    Only returns bigrams appearing in >= min_freq of questions.
    """
    n = len(texts)
    if n < 10:
        return []

    cleaned = [_clean_text(t) for t in texts]
    doc_bigrams = [_tokenize_bigrams(t) for t in cleaned]

    min_df = max(2, int(n * min_freq * 0.5))
    doc_freq: Counter = Counter()
    for bg_list in doc_bigrams:
        for bg in set(bg_list):
            doc_freq[bg] += 1

    vocab = [bg for bg, df in doc_freq.items() if df >= min_df]
    if not vocab:
        return []
    vocab_idx = {bg: i for i, bg in enumerate(vocab)}
    V = len(vocab)

    tf_agg   = np.zeros(V, dtype=np.float64)
    doc_flag = np.zeros(V, dtype=np.float64)
    for bg_list in doc_bigrams:
        seen: set = set()
        for bg in bg_list:
            if bg in vocab_idx:
                tf_agg[vocab_idx[bg]] += 1
                if bg not in seen:
                    doc_flag[vocab_idx[bg]] += 1
                    seen.add(bg)

    total = tf_agg.sum()
    if total == 0:
        return []
    tf_norm = tf_agg / total

    idf = np.log((n + 1) / (1 + doc_flag)) + 1.0
    scores = tf_norm * idf
    freq   = doc_flag / n

    mask = freq >= min_freq
    if not mask.any():
        return []

    filtered_scores = scores * mask
    top_idx = np.argsort(-filtered_scores)[:top_k * 2]
    top_idx = [i for i in top_idx if mask[i]][:top_k]

    return [(vocab[i], float(scores[i]), float(freq[i])) for i in top_idx]


def questions_containing_bigrams(texts: List[str], bigrams: List[str]) -> np.ndarray:
    """
    Returns boolean array: True for each question containing
    at least one of the given bigrams.
    """
    mask = np.zeros(len(texts), dtype=bool)
    for bigram in bigrams:
        words = bigram.split()
        if len(words) == 2:
            pattern = re.compile(
                r'\b' + re.escape(words[0]) + r'\s+' + re.escape(words[1]) + r'\b',
                re.IGNORECASE,
            )
            for i, t in enumerate(texts):
                if pattern.search(t):
                    mask[i] = True
    return mask


# ── Main discovery loop ──────────────────────────────────────────────────────

def discover_category(
    texts: List[str],
    category: str,
    n_passes: int = N_PASSES,
) -> List[Dict]:
    """
    Run n_passes of iterative residual c-TF-IDF on texts.

    Returns list of pass results, each with:
        pass_num        int
        n_questions     int   — questions in this pass
        pct_remaining   float — fraction of original questions remaining
        bigrams         list of (bigram, score, freq_pct)
        removed_bigrams list of bigrams used for removal (top TOP_BIGRAMS_REMOVE)
    """
    remaining = list(range(len(texts)))
    results = []

    for pass_num in range(1, n_passes + 1):
        n_remaining = len(remaining)
        pct_remaining = n_remaining / len(texts)

        if n_remaining < MIN_QUESTIONS:
            print(f"    Pass {pass_num}: stopping — only {n_remaining} questions left")
            break

        pass_texts = [texts[i] for i in remaining]
        bigrams = compute_ctfidf_bigrams(pass_texts, top_k=TOP_BIGRAMS_SHOWN)

        if not bigrams:
            print(f"    Pass {pass_num}: stopping — no bigrams above threshold")
            break

        remove_bigrams = [b for b, _, _ in bigrams[:TOP_BIGRAMS_REMOVE]]

        results.append({
            "pass_num": pass_num,
            "n_questions": n_remaining,
            "pct_remaining": round(pct_remaining, 3),
            "bigrams": bigrams,
            "removed_bigrams": remove_bigrams,
        })

        print(f"    Pass {pass_num} ({n_remaining:,} q, {pct_remaining:.0%} remaining):")
        for bigram, score, freq in bigrams[:TOP_BIGRAMS_SHOWN]:
            print(f"      {bigram:<35} {freq*100:>5.1f}% of questions")

        remove_mask = questions_containing_bigrams(pass_texts, remove_bigrams)
        n_removed = int(remove_mask.sum())
        keep_local = np.where(~remove_mask)[0]
        remaining = [remaining[i] for i in keep_local]

        print(f"      → removed {n_removed:,} questions containing: {', '.join(remove_bigrams)}")
        print()

    return results


def run_discovery(seg_name: str, target_category: str = None) -> None:
    seg_dir = Path(OUTPUT_DIR) / SEGMENT_SUBDIR / seg_name
    parquet  = seg_dir / "A_cluster_assignments.parquet"

    if not parquet.exists():
        print(f"ERROR: no assignments parquet at {parquet}")
        return

    print(f"\n{'='*60}")
    print(f"Segment: {seg_name}")
    print(f"{'='*60}")

    assign = pd.read_parquet(parquet)
    print(f"Loaded {len(assign):,} questions")

    assign = join_topic_to_assignments(assign, Path(DATA_DIR))

    categories = (
        [target_category] if target_category
        else sorted(assign["topic"].dropna().unique().tolist())
    )
    categories = [c for c in categories if c and c != "unknown"]

    all_results: Dict[str, List[Dict]] = {}

    for cat in categories:
        cat_df = assign[assign["topic"] == cat]
        texts  = cat_df["body_norm"].astype(str).tolist()
        print(f"\n{'─'*50}")
        print(f"Category: {cat}  ({len(texts):,} questions)")
        print(f"{'─'*50}")

        if len(texts) < MIN_QUESTIONS:
            print(f"  Too few questions, skipping.")
            continue

        results = discover_category(texts, cat)
        all_results[cat] = results

    _write_text_report(all_results, seg_name, seg_dir)
    if HAS_OPENPYXL:
        _write_excel_report(all_results, seg_name, seg_dir)
    _write_json(all_results, seg_name, seg_dir)

    print(f"\n[done] Discovery complete for {seg_name}")


# ── Output writers ───────────────────────────────────────────────────────────

def _write_text_report(
    all_results: Dict[str, List[Dict]],
    seg_name: str,
    seg_dir: Path,
) -> None:
    lines = [f"Guided topic discovery — {seg_name}", "=" * 80, ""]

    for cat, passes in all_results.items():
        if not passes:
            continue
        lines.append(f"{cat.upper()}")
        lines.append("─" * 60)

        header_parts = []
        for p in passes:
            label = f"Pass {p['pass_num']} ({p['n_questions']:,} q)"
            if p['pass_num'] > 1:
                prev = passes[p['pass_num'] - 2]
                label += f" [removed: {', '.join(prev['removed_bigrams'])}]"
            header_parts.append(f"{label:<40}")
        lines.append("  ".join(header_parts))
        lines.append("")

        max_rows = max(len(p["bigrams"]) for p in passes)
        for row_i in range(max_rows):
            row_parts = []
            for p in passes:
                if row_i < len(p["bigrams"]):
                    bigram, _, freq = p["bigrams"][row_i]
                    cell = f"{bigram}  ({freq*100:.1f}%)"
                else:
                    cell = ""
                row_parts.append(f"{cell:<40}")
            lines.append("  ".join(row_parts))

        lines.append("")
        lines.append("")

    out_path = seg_dir / f"topic_discovery_{seg_name}.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nText report written: {out_path.name}")


def _write_excel_report(
    all_results: Dict[str, List[Dict]],
    seg_name: str,
    seg_dir: Path,
) -> None:
    wb = Workbook()
    wb.remove(wb.active)  # remove default sheet

    HDR_FILL   = PatternFill("solid", fgColor="2C3E50")
    HDR_FONT   = Font(color="FFFFFF", bold=True, size=11)
    CAT_FILL   = PatternFill("solid", fgColor="ECF0F1")
    CAT_FONT   = Font(bold=True, size=12)
    PASS_FILLS = [
        PatternFill("solid", fgColor="D6EAF8"),
        PatternFill("solid", fgColor="D5F5E3"),
        PatternFill("solid", fgColor="FDEBD0"),
        PatternFill("solid", fgColor="F9EBEA"),
        PatternFill("solid", fgColor="EAF2FF"),
    ]
    thin   = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for cat, passes in all_results.items():
        if not passes:
            continue

        # Excel sheet names cannot contain: / \ ? * [ ] :
        _INVALID = r'[/\\?*\[\]:]'
        sheet_name = re.sub(_INVALID, '-', cat).strip()[:31]
        ws = wb.create_sheet(title=sheet_name)

        # Category title spanning all columns
        ws.merge_cells(start_row=1, start_column=1,
                       end_row=1, end_column=len(passes) * 2)
        cell = ws.cell(row=1, column=1, value=cat.upper())
        cell.font = CAT_FONT
        cell.fill = CAT_FILL
        cell.alignment = Alignment(horizontal="left", vertical="center")
        ws.row_dimensions[1].height = 22

        # Pass headers
        for p_idx, p in enumerate(passes):
            col = p_idx * 2 + 1
            pass_label = f"Pass {p['pass_num']}"
            sub_label  = f"{p['n_questions']:,} q  ({p['pct_remaining']*100:.0f}% remaining)"
            if p['pass_num'] > 1:
                prev_removed = passes[p['pass_num'] - 2]["removed_bigrams"]
                sub_label += f"\nRemoved: {', '.join(prev_removed)}"

            ws.merge_cells(start_row=2, start_column=col, end_row=2, end_column=col + 1)
            c = ws.cell(row=2, column=col, value=pass_label)
            c.font = HDR_FONT
            c.fill = HDR_FILL
            c.alignment = Alignment(horizontal="center")

            for cc, label in [(col, "Bigram"), (col + 1, "% questions")]:
                cell = ws.cell(row=3, column=cc, value=label)
                cell.font = Font(bold=True, size=10)
                cell.fill = PASS_FILLS[p_idx % len(PASS_FILLS)]
                cell.alignment = Alignment(horizontal="center")

        # Data rows
        max_rows = max(len(p["bigrams"]) for p in passes)
        for row_i in range(max_rows):
            excel_row = row_i + 4
            for p_idx, p in enumerate(passes):
                col  = p_idx * 2 + 1
                fill = PASS_FILLS[p_idx % len(PASS_FILLS)]
                if row_i < len(p["bigrams"]):
                    bigram, score, freq = p["bigrams"][row_i]
                    c1 = ws.cell(row=excel_row, column=col,     value=bigram)
                    c2 = ws.cell(row=excel_row, column=col + 1, value=round(freq * 100, 1))
                    c2.number_format = '0.0"%"'
                    for c in [c1, c2]:
                        c.fill = fill
                        c.border = border
                        c.alignment = Alignment(vertical="center")
                else:
                    for cc in [col, col + 1]:
                        ws.cell(row=excel_row, column=cc).fill = fill

        # Column widths
        for p_idx in range(len(passes)):
            col = p_idx * 2 + 1
            ws.column_dimensions[get_column_letter(col)].width     = 32
            ws.column_dimensions[get_column_letter(col + 1)].width = 14

        ws.freeze_panes = ws.cell(row=4, column=1)

    out_path = seg_dir / f"topic_discovery_{seg_name}.xlsx"
    wb.save(str(out_path))
    print(f"Excel report written: {out_path.name}")


def _write_json(
    all_results: Dict[str, List[Dict]],
    seg_name: str,
    seg_dir: Path,
) -> None:
    def _clean(obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, dict):        return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [_clean(v) for v in obj]
        return obj

    out_path = seg_dir / f"topic_discovery_{seg_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_clean(all_results), f, ensure_ascii=False, indent=2)
    print(f"JSON written: {out_path.name}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Iterative residual c-TF-IDF per category.")
    parser.add_argument("--segment",  default="girls_13_15",
                        help="Segment name or 'all'")
    parser.add_argument("--category", default=None,
                        help="Single category slug, e.g. 'kropp'. Default: all.")
    args = parser.parse_args()

    segs = SEGMENTS if args.segment == "all" else [args.segment]
    for seg in segs:
        try:
            run_discovery(seg, args.category)
        except Exception as e:
            print(f"ERROR in {seg}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
