#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Embed late-arriving questionnaire rows and append shards to an existing outputs folder.

This script is intended for the situation where the main embedding pipeline already ran,
but one additional CSV arrived later. It reuses the same text normalization + masking logic,
filters out texts that already exist in `C_candidates_index.parquet`, computes embeddings,
and writes additional shard files into the same output directory.

It does NOT rewrite existing shard files.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# --- Defaults aligned with your prior embedding job ---
DEFAULT_MODEL = "Alibaba-NLP/gte-multilingual-base"
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_SEQ_LEN = 256
DEFAULT_SHARD_PREFIX = "bodynorm"
DEFAULT_SHARD_SIZE = 10_000


def normalize_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def simple_pii_mask(s: str) -> str:
    if not s:
        return s
    s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[[MASKERT_EPOST]]", s)
    s = re.sub(r"(?:(?:\+?\d{1,3}[\s\-]?)?(?:\d[\s\-]?){7,})", "[[MASKERT_TLF]]", s)
    return s


def embed_texts(texts: List[str], model_name: str, batch_size: int, max_seq_len: int) -> np.ndarray:
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        provider = "DmlExecutionProvider"  # best effort for Windows/AMD
        tok = AutoTokenizer.from_pretrained(model_name)
        ort_model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True, provider=provider)

        out = []
        for i in range(0, len(texts), batch_size):
            enc = tok(
                texts[i : i + batch_size],
                padding=True,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="np",
            )
            o = ort_model(**enc)
            mask = enc["attention_mask"][..., None]
            s = (o.last_hidden_state * mask).sum(axis=1)
            d = mask.sum(axis=1).clip(min=1e-9)
            pooled = s / d
            pooled = pooled / (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12)
            out.append(pooled.astype("float32"))
        return np.vstack(out)
    except Exception:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, trust_remote_code=True)
        if hasattr(model, "max_seq_length"):
            model.max_seq_length = max_seq_len
        embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        return np.asarray(embs, dtype=np.float32)


def parse_existing_shard_endings(output_dir: Path, shard_prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(shard_prefix)}_embeddings_(\d+)_(\d+)\.npz$")
    max_end = 0
    for p in output_dir.glob(f"{shard_prefix}_embeddings_*_*.npz"):
        m = pattern.match(p.name)
        if m:
            max_end = max(max_end, int(m.group(2)))
    return max_end


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed late-arriving data and append shards.")
    parser.add_argument("--late-csv", required=True, help="Path to the late-arriving CSV file.")
    parser.add_argument("--output-dir", required=True, help="Path to existing outputs directory.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--shard-prefix", default=DEFAULT_SHARD_PREFIX)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    args = parser.parse_args()

    late_csv = Path(args.late_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not late_csv.exists():
        raise FileNotFoundError(f"Late CSV not found: {late_csv}")

    df = pd.read_csv(late_csv, encoding="utf-8", dtype=str, keep_default_na=False, na_values=["", "NA", "NaN"])
    df = df.rename(columns={c: c.strip() for c in df.columns})
    if "body" not in df.columns:
        raise ValueError("Late CSV is missing required column: body")

    df["body_norm"] = df["body"].map(normalize_text).map(simple_pii_mask).str.lower()
    df = df[~df["body_norm"].isna() & (df["body_norm"].str.len() > 3)].copy().reset_index(drop=True)

    # Remove duplicates inside late file
    df = df.drop_duplicates(subset=["body_norm"]).reset_index(drop=True)

    # Filter out texts already embedded in the previous run using candidate index.
    existing_idx_path = output_dir / "C_candidates_index.parquet"
    if existing_idx_path.exists():
        existing = pd.read_parquet(existing_idx_path, columns=["body_norm"])
        existing_set = set(existing["body_norm"].astype(str).tolist())
        df = df[~df["body_norm"].isin(existing_set)].copy().reset_index(drop=True)

    texts = df["body_norm"].tolist()
    if not texts:
        print("No new texts to embed after de-dup against existing outputs.")
        return

    start_idx = parse_existing_shard_endings(output_dir, args.shard_prefix)

    written_shards = []
    for i in range(0, len(texts), args.shard_size):
        chunk = texts[i : i + args.shard_size]
        embs = embed_texts(chunk, args.model, args.batch_size, args.max_seq_len)
        shard_start = start_idx + i
        shard_end = start_idx + i + len(chunk)
        shard_path = output_dir / f"{args.shard_prefix}_embeddings_{shard_start}_{shard_end}.npz"
        np.savez_compressed(shard_path, embs=np.asarray(embs, dtype=np.float32))
        written_shards.append(shard_path.name)

    map_path = output_dir / "late_arrivals_body_norm.csv"
    df[["body_norm"]].to_csv(map_path, index=False, encoding="utf-8")

    run_report = {
        "late_csv": str(late_csv),
        "n_input_rows_after_cleaning": int(len(df)),
        "n_shards_written": int(len(written_shards)),
        "written_shards": written_shards,
        "output_dir": str(output_dir),
        "model": args.model,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
    }

    report_path = output_dir / "late_arrivals_embedding_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(run_report, f, ensure_ascii=False, indent=2)

    print(json.dumps(run_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
