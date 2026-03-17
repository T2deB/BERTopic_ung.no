#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bottom-up clustering on all embeddings with optional guided seeding.

Now supports segmented runs:
- boys 13-15
- boys 16-20
- girls 13-15
- girls 16-20

Outputs are written per segment under: <OUTPUT_DIR>/segments/<segment_name>/
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import hdbscan
import numpy as np
import pandas as pd
import umap
from nltk.corpus import stopwords
from scipy.sparse import spdiags
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kwargs):  # type: ignore[misc]
        desc = kwargs.get("desc", "")
        if desc:
            print(f"[progress] {desc} ...")
        return it

# ---------------------------
# CONFIG
# ---------------------------
SCRIPT_DIR = Path(__file__).parent   # folder containing the .py files (tracked by git)

DATA_DIR = r"C:\Users\TorsteinDeBesche\NIFU\21571 Folkehelse og livsmestring - General\Ap2c Informasjonskanal for unge\Positron_project\data"
OUTPUT_DIR = r"C:\Users\TorsteinDeBesche\NIFU\21571 Folkehelse og livsmestring - General\Ap2c Informasjonskanal for unge\Positron_project\outputs"
SHARD_PREFIX = "bodynorm"
A_EMB_FILE = "A_embeddings.npz"
A_INDEX_FILE = "C_candidates_index.parquet"
AGE_LOOKUP_FILE = "C_index.parquet"   # optional; provides exact numeric age column
LATE_ROWS_FILE = "late_arrivals_body_norm.csv"

# Per-segment guided-topic files live in SCRIPT_DIR so they are tracked by git.
# The "_default" entry is used for unknown segment names and for the non-segmented run.
GUIDED_TOPICS_FILES = {
    "boys_13_15":  "guided_topics_boys_13_15.json",
    "boys_16_20":  "guided_topics_boys_16_20.json",
    "girls_13_15": "guided_topics_girls_13_15.json",
    "girls_16_20": "guided_topics_girls_16_20.json",
    "_default":    "guided_topics.json",
}
USE_BERTOPIC_GUIDED = True

# Fuzzy BERTopic (Nikbakht & Zojaji, 2026)
USE_FUZZY_BERTOPIC = True   # set True to activate fuzzy c-means mode
FUZZY_N_CLUSTERS = 50        # fallback cluster count for unknown segment names
FUZZY_N_CLUSTERS_BY_SEGMENT = {
    "boys_13_15":  80,
    "boys_16_20":  60,
    "girls_13_15": 80,
    "girls_16_20": 80,
}
FUZZY_M = 2.0                # fuzziness exponent (m=2 is standard)
FUZZY_BATCH_SIZE = 5000      # doc-batch size for TI accumulation and membership prediction

RUN_SEGMENTED = True
SEGMENT_OUTPUT_SUBDIR = "segments"

# NOTE: if exact age column is missing, 13-15 is approximated using age_group == "13-16"
SEGMENTS = [
    {"name": "boys_13_15", "gender": "m", "age_min": 13, "age_max": 15},
    {"name": "boys_16_20", "gender": "m", "age_min": 16, "age_max": 20},
    {"name": "girls_13_15", "gender": "k", "age_min": 13, "age_max": 15},
    {"name": "girls_16_20", "gender": "k", "age_min": 16, "age_max": 20},
]

# Sampling
SAMPLE_SIZE = 120_000
MIN_PER_STRATUM = 50
SAMPLE_RANDOM_STATE = 42

# UMAP
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.08
UMAP_N_COMPONENTS = 15
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42

# HDBSCAN
HDBSCAN_MIN_CLUSTER_SIZE = 100
HDBSCAN_MIN_SAMPLES = 30
HDBSCAN_CLUSTER_SELECTION_EPSILON = 0.0
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"
HDBSCAN_CORE_DIST_N_JOBS = 1

# Assignment thresholds
ASSIGN_MIN_SIM = 0.44
ASSIGN_MIN_MARGIN = 0.020

# c-TF-IDF
NGRAM_RANGE = (1, 2)
MIN_DF = 10
TOP_K_TERMS = 15

CUSTOM_STOP = {
    "hei", "heisann", "hallo", "halloi", "pls", "plis", "please", "takk", "påforhånd",
    "veldig", "ganske", "litt", "mye", "egentlig", "faktisk", "kanskje", "liksom", "bare", "helt",
    "må", "måtte", "kan", "kunne", "skal", "skulle", "vil", "ville", "bør", "burde",
    "gjør", "gjøre", "gjorde", "får", "få", "går", "gå", "kommer", "komme", "kom",
    "tar", "ta", "tok", "føler", "føle", "vet", "veta", "hatt", "har", "hadde", "blir", "bli", "ble", "vært", "er", "var",
    "ting", "andre", "mange", "noe", "noen", "gang", "år", "dag", "uke", "måte", "grunn", "type", "greie", "greit",
    "hva", "hvordan", "hvorfor", "hvem", "hvor", "når", "fordi", "men", "også", "derfor", "altså", "så", "om",
    "jeg", "meg", "min", "mitt", "mine", "vi", "oss", "vår", "vårt", "våre",
    "du", "deg", "din", "ditt", "dine", "dere", "deres", "han", "hun", "hen", "de", "dem", "man", "en", "den", "det", "dette", "disse", "slik", "sånn",
}
try:
    STOP_WORDS = set(stopwords.words("norwegian"))
except Exception:
    STOP_WORDS = set()
STOP_WORDS |= CUSTOM_STOP
STOP_WORDS = sorted(STOP_WORDS)


def l2norm(x: np.ndarray) -> np.ndarray:
    return normalize(x, norm="l2", axis=1, copy=False)


def strip_greeting(t: str) -> str:
    return re.sub(r"^\s*(hei|heisann|hallo|halloi)[\s,!:.-]+", " ", t, flags=re.IGNORECASE)


def load_all_embeddings(outdir: Path, shard_prefix: str) -> np.ndarray:
    a_npz = outdir / A_EMB_FILE
    if a_npz.exists():
        return np.load(a_npz)["embs"].astype("float32")

    pat = re.compile(rf"^{re.escape(shard_prefix)}_embeddings_(\d+)_(\d+)\.npz$")
    shards = []
    for p in outdir.glob(f"{shard_prefix}_embeddings_*_*.npz"):
        m = pat.match(p.name)
        if m:
            shards.append((int(m.group(1)), int(m.group(2)), p))
    if not shards:
        raise FileNotFoundError("No A_embeddings.npz and no shard files found.")
    shards.sort(key=lambda x: x[0])

    arrays = []
    for start, end, p in shards:
        arr = np.load(p)["embs"].astype("float32")
        if arr.shape[0] != (end - start):
            warnings.warn(f"Shard row count mismatch for {p.name}: expected {end-start}, got {arr.shape[0]}")
        arrays.append(arr)
    return np.vstack(arrays)


def load_aligned_index(outdir: Path) -> pd.DataFrame:
    idx_path = outdir / A_INDEX_FILE
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing {idx_path}")
    return pd.read_parquet(idx_path)


def _norm_ts(series: pd.Series) -> pd.Series:
    """Normalise a createdAt series to a canonical ISO-like string for joining.

    Tries pd.to_datetime first (handles formats like 'Jan 1, 2020 @ 02:26:42.000'
    and ISO strings equally).  Falls back to stripped raw strings when parsing fails.
    """
    parsed = pd.to_datetime(series.astype(str), errors="coerce", utc=True)
    if parsed.notna().mean() > 0.5:
        return parsed.dt.strftime("%Y-%m-%dT%H:%M:%S")
    return series.astype(str).str.strip()


def _norm_body(series: pd.Series) -> pd.Series:
    """Normalise question body text for joining.

    Applies: lowercase → strip → collapse multiple spaces → strip greeting prefix.
    Both the raw CSV body column and the index body_norm column are normalised
    identically so that the join key is consistent across sources.
    """
    s = (
        series.astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    # Strip leading greeting words (same pattern as strip_greeting())
    s = s.str.replace(
        r"^(hei|heisann|hallo|halloi)[\s,!:.\-]+", "", regex=True
    ).str.strip()
    return s


def build_age_lookup_from_raw(data_dir: Path) -> pd.DataFrame:
    """Scan DATA_DIR for *.csv files and build a (body_key, age_raw) lookup.

    Reads only the question body text and age columns to minimise memory.
    Auto-detects the text column name from the first file header.
    Returns DataFrame with columns [body_key, age_raw] deduplicated on body_key.
    Returns an empty DataFrame if no usable files are found.
    """
    csv_files = sorted(p for p in data_dir.glob("*.csv")
                       if p.suffix.lower() == ".csv" and p.stat().st_size > 0)
    if not csv_files:
        warnings.warn(f"[age_raw] No CSV files found in {data_dir}")
        return pd.DataFrame(columns=["body_key", "age_raw"])

    # Auto-detect the question body column name from the first file
    text_col = None
    try:
        header = pd.read_csv(csv_files[0], nrows=0, encoding="utf-8-sig", sep=None, engine="python")
        for candidate in ["body", "question", "text", "body_norm", "sporsmal", "tekst"]:
            if candidate in header.columns:
                text_col = candidate
                break
        if text_col is None:
            warnings.warn(
                f"[age_raw] Could not identify text column in {csv_files[0].name}; "
                f"columns found: {header.columns.tolist()}"
            )
            return pd.DataFrame(columns=["body_key", "age_raw"])
    except Exception as exc:
        warnings.warn(f"[age_raw] Could not read header of {csv_files[0].name}: {exc}")
        return pd.DataFrame(columns=["body_key", "age_raw"])

    parts = []
    for p in csv_files:
        try:
            chunk = pd.read_csv(
                p, usecols=[text_col, "age"],
                dtype={text_col: str, "age": str},  # read as str; coerce "-"/blanks to NaN after
                encoding="utf-8-sig",
                low_memory=True,
            )
            parts.append(chunk)
        except Exception as exc:
            warnings.warn(f"[age_raw] Could not read {p.name}: {exc}")

    if not parts:
        return pd.DataFrame(columns=["body_key", "age_raw"])

    raw = pd.concat(parts, ignore_index=True)
    raw["age_raw"] = pd.to_numeric(raw["age"], errors="coerce")
    raw["body_key"] = _norm_body(raw[text_col])

    lookup = (
        raw[["body_key", "age_raw"]]
        .dropna(subset=["body_key"])
        .drop_duplicates(subset=["body_key"])
    )
    print(f"Age lookup built: {len(lookup):,} rows from {len(parts)} files.")
    return lookup


def enrich_with_exact_age(idx: pd.DataFrame, outdir: Path) -> Tuple[pd.DataFrame, str, float]:
    """Join numeric age into idx.  Tries three sources in order:

    1. Raw CSV files in DATA_DIR  (joined on normalised createdAt)
    2. C_index.parquet in outdir  (joined on body_norm + createdAt)
    3. age_group string fallback  (existing approximate column)

    Returns (enriched_df, age_col_used, coverage_pct) where:
      age_col_used   = "age_numeric_joined" | "age_group_fallback"
      coverage_pct   = fraction of rows that received a numeric age (0.0–1.0)
    """
    _FILL_THRESHOLD = 0.80

    # If age is already present, validate and keep or drop
    if "age" in idx.columns:
        age_numeric = pd.to_numeric(idx["age"], errors="coerce")
        fill_rate = float(age_numeric.notna().mean())
        if fill_rate >= _FILL_THRESHOLD:
            idx = idx.copy()
            idx["age"] = age_numeric
            print(f"[age] existing 'age' column retained — numeric fill rate {fill_rate:.1%}")
            return idx, "age_numeric_joined", fill_rate
        print(
            f"[age] existing 'age' column has only {fill_rate:.1%} numeric values "
            f"(< {_FILL_THRESHOLD:.0%}); dropping and re-attempting join"
        )
        idx = idx.drop(columns=["age"])

    # Attempt 1 — raw CSV files in DATA_DIR (joined on normalised body text)
    data_dir = Path(DATA_DIR)
    if data_dir.exists():
        try:
            lookup = build_age_lookup_from_raw(data_dir)
            if not lookup.empty:
                idx_keys = _norm_body(idx["body_norm"])
                joined_age = (
                    pd.DataFrame({"body_key": idx_keys.values})
                    .merge(lookup, on="body_key", how="left")["age_raw"]
                )
                age_numeric = pd.to_numeric(joined_age, errors="coerce")
                fill_rate = float(age_numeric.notna().mean())
                n_matched = int(age_numeric.notna().sum())
                print(f"Numeric age join coverage: {fill_rate:.1%}")
                if fill_rate >= _FILL_THRESHOLD:
                    enriched = idx.copy()
                    enriched["age"] = age_numeric.values
                    print(
                        f"[age] raw CSV body-text join: age matched for {n_matched}/{len(enriched)} rows "
                        f"({fill_rate:.1%}) — using 'age_numeric_joined'"
                    )
                    return enriched, "age_numeric_joined", fill_rate
                print(
                    f"[age] raw CSV body-text join coverage {fill_rate:.1%} < {_FILL_THRESHOLD:.0%} "
                    f"— trying {AGE_LOOKUP_FILE} fallback"
                )
        except Exception as exc:
            warnings.warn(f"[age] raw CSV join failed: {exc}")

    # Attempt 2 — C_index.parquet
    lookup_path = outdir / AGE_LOOKUP_FILE
    if lookup_path.exists():
        try:
            lkp2 = pd.read_parquet(lookup_path, columns=["body_norm", "createdAt", "age"])
            lkp2 = lkp2.drop_duplicates(subset=["body_norm", "createdAt"])
            lkp2["createdAt"] = lkp2["createdAt"].astype(str)
            idx_tmp = idx.copy()
            idx_tmp["createdAt"] = idx_tmp["createdAt"].astype(str)
            enriched = idx_tmp.merge(lkp2, on=["body_norm", "createdAt"], how="left")
            age_numeric = pd.to_numeric(enriched["age"], errors="coerce")
            fill_rate = float(age_numeric.notna().mean())
            if fill_rate >= _FILL_THRESHOLD:
                enriched["age"] = age_numeric
                n_matched = int(age_numeric.notna().sum())
                print(
                    f"[age] {AGE_LOOKUP_FILE} join: age matched for {n_matched}/{len(enriched)} rows "
                    f"({fill_rate:.1%}) — using 'age_numeric_joined'"
                )
                return enriched, "age_numeric_joined", fill_rate
            enriched = enriched.drop(columns=["age"], errors="ignore")
            print(
                f"[age] {AGE_LOOKUP_FILE} coverage {fill_rate:.1%} < {_FILL_THRESHOLD:.0%} "
                f"— falling back to age_group"
            )
        except Exception as exc:
            warnings.warn(f"[age] {AGE_LOOKUP_FILE} join failed: {exc}")

    # Final fallback — age_group strings
    print(
        "WARNING: Numeric age join coverage too low or failed — "
        "using age_group string fallback. "
        "NOTE: The age_group string value '13-16' covers both 13–15 AND 16-year-olds. "
        "All 16-year-olds will be misclassified into the younger segments."
    )
    return idx, "age_group_fallback_BOUNDARY_WARNING", 0.0


def _empty_meta_frame(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame({
        "body_norm": [""] * n_rows,
        "createdAt": ["unknown"] * n_rows,
        "age_group": ["ukjent"] * n_rows,
        "gender_std": ["ikke oppgitt"] * n_rows,
        "source": ["auto_filled"] * n_rows,
    })


def reconcile_index_length(outdir: Path, idx: pd.DataFrame, n_embs: int) -> pd.DataFrame:
    n_idx = len(idx)
    if n_idx == n_embs:
        return idx
    if n_idx > n_embs:
        raise ValueError(f"Index longer than embeddings: index={n_idx}, embeddings={n_embs}")

    missing = n_embs - n_idx
    parts = [idx.copy()]

    late_path = outdir / LATE_ROWS_FILE
    if late_path.exists() and missing > 0:
        late = pd.read_csv(late_path, dtype=str).fillna("")
        if "body_norm" not in late.columns:
            raise ValueError(f"{LATE_ROWS_FILE} exists but has no body_norm column")
        take = late.iloc[:missing][["body_norm"]].copy()
        take["createdAt"] = "unknown"
        take["age_group"] = "ukjent"
        take["gender_std"] = "ikke oppgitt"
        take["source"] = "late_arrivals_body_norm"
        parts.append(take)
        missing -= len(take)

    if missing > 0:
        warnings.warn(f"Index short by {missing}; filling with placeholders")
        parts.append(_empty_meta_frame(missing))

    out = pd.concat(parts, ignore_index=True)
    if len(out) != n_embs:
        raise ValueError(f"Failed to align index length: got {len(out)} expected {n_embs}")
    return out


def normalize_export_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["createdAt", "age_group", "gender_std", "body_norm", "source"]:
        if col in out.columns:
            out[col] = out[col].astype(str)
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].astype(str)
    return out


def ensure_strat_cols(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for col, default in [("createdAt", "unknown"), ("age_group", "ukjent"), ("gender_std", "ikke oppgitt")]:
        if col not in x.columns:
            x[col] = default
        x[col] = x[col].astype(str)
    return x


def stratified_sample_index(df: pd.DataFrame, n_target: int, min_per_stratum: int, seed: int) -> np.ndarray:
    sdf = ensure_strat_cols(df)
    sdf["year"] = sdf["createdAt"].str[:4].where(sdf["createdAt"].str[:4].str.match(r"^\d{4}$"), "unknown")
    strata = sdf.groupby(["year", "age_group", "gender_std"]).size().reset_index(name="n")
    total = int(strata["n"].sum())
    if n_target >= total:
        return np.arange(len(df))

    strata["take"] = (strata["n"] / total * n_target).round().astype(int)
    strata["take"] = strata[["take", "n"]].apply(lambda r: min(max(int(r["take"]), min_per_stratum), int(r["n"])), axis=1)

    diff = int(n_target - strata["take"].sum())
    order = strata.sort_values("n", ascending=(diff < 0)).index.tolist()
    i = 0
    while diff != 0 and i < len(order):
        idx = order[i]
        cur = int(strata.at[idx, "take"])
        cap = int(strata.at[idx, "n"])
        if diff > 0 and cur < cap:
            strata.at[idx, "take"] = cur + 1
            diff -= 1
        elif diff < 0 and cur > 0:
            strata.at[idx, "take"] = cur - 1
            diff += 1
        i += 1

    rng = np.random.default_rng(seed)
    chosen = []
    keyed = sdf.reset_index()
    for _, row in strata.iterrows():
        m = (keyed["year"] == row["year"]) & (keyed["age_group"] == row["age_group"]) & (keyed["gender_std"] == row["gender_std"])
        idxs = keyed.loc[m, "index"].to_numpy()
        k = int(row["take"])
        if k > 0:
            chosen.append(idxs if len(idxs) <= k else rng.choice(idxs, size=k, replace=False))

    return np.unique(np.concatenate(chosen)) if chosen else np.arange(0)


def load_guided_topics(path: Path) -> Tuple[List[str], List[List[str]]]:
    if not path.exists():
        template = {"topics": [{"label": "example_sleep", "keywords": ["søvn", "sove", "trøtt", "døgnrytme"]}]}
        path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
        return [], []
    raw = json.loads(path.read_text(encoding="utf-8"))
    labels, seed_topic_list = [], []
    for t in raw.get("topics", []):
        label = str(t.get("label", "")).strip()
        kws = [str(x).strip().lower() for x in t.get("keywords", []) if str(x).strip()]
        if label and kws:
            labels.append(label)
            seed_topic_list.append(kws)
    return labels, seed_topic_list


def compute_centroids(embs: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    centroids = {}
    for lab in np.unique(labels[labels >= 0]):
        idx = np.where(labels == lab)[0]
        c = embs[idx].mean(axis=0, keepdims=True)
        centroids[int(lab)] = l2norm(c)[0]
    return centroids


def assign_all(embs: np.ndarray, centroids: Dict[int, np.ndarray], min_sim: float, min_margin: float):
    if not centroids:
        n = len(embs)
        return np.full(n, -1, dtype=int), np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    labs = sorted(centroids.keys())
    C = np.vstack([centroids[k] for k in labs])
    E = l2norm(embs.copy())
    sims = E @ C.T
    best_k = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)
    second_sim = np.partition(sims, -2, axis=1)[:, -2] if sims.shape[1] > 1 else np.zeros_like(best_sim)
    margin = best_sim - second_sim
    out_labels = np.array([labs[k] for k in best_k], dtype=int)
    out_labels[(best_sim < float(min_sim)) | (margin < float(min_margin))] = -1
    return out_labels, best_sim.astype(np.float32), margin.astype(np.float32)


def ctfi_df(texts_per_cluster: Dict[int, List[str]], ngram_range=(1, 2), min_df=3, stop_words=None, top_k=15) -> pd.DataFrame:
    clusters = sorted(texts_per_cluster.keys())
    docs = [" ".join(texts_per_cluster[cid]) for cid in clusters]
    vect = CountVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=0.6,
        stop_words=stop_words,
        lowercase=True,
        token_pattern=r"(?u)\b[a-zæøå0-9]{3,}\b",
        strip_accents=None,
    )
    X = vect.fit_transform(docs)
    vocab = vect.get_feature_names_out()
    print(f"[ctfidf] vocab={len(vocab)} min_df={min_df} ngram={ngram_range} stop_words={len(stop_words) if stop_words else 0}")
    tf = normalize(X, norm="l1", axis=1, copy=False)
    df = (X > 0).sum(axis=0).A1
    idf = np.log((len(clusters) + 1) / (1 + df)) + 1.0
    scores = tf @ spdiags(idf, 0, X.shape[1], X.shape[1])

    rows = []
    for i, cid in enumerate(clusters):
        row = scores.getrow(i).toarray().ravel()
        if row.size == 0 or (row <= 0).all():
            continue
        top_idx = np.argsort(-row)[: int(top_k)]
        for rank, j in enumerate(top_idx, start=1):
            rows.append((int(cid), vocab[j], float(row[j]), int(rank)))
    return pd.DataFrame(rows, columns=["cluster_id", "term", "score", "rank"])


def top_terms_per_cluster(kw_df: pd.DataFrame, top_k: int) -> Dict[int, str]:
    if kw_df is None or kw_df.empty:
        return {}
    return kw_df.sort_values(["cluster_id", "rank"]).groupby("cluster_id")["term"].apply(lambda s: ", ".join(s.tolist()[:top_k])).to_dict()


def cluster_with_hdbscan(sample_embs: np.ndarray) -> np.ndarray:
    um = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=UMAP_N_COMPONENTS,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
        verbose=True,
    )
    sample_umap = um.fit_transform(sample_embs)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_epsilon=HDBSCAN_CLUSTER_SELECTION_EPSILON,
        cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
        metric="euclidean",
        prediction_data=True,
        core_dist_n_jobs=HDBSCAN_CORE_DIST_N_JOBS,
    )
    return clusterer.fit_predict(sample_umap)


def cluster_with_bertopic_guided(sample_texts: List[str], sample_embs: np.ndarray, seed_topic_list: List[List[str]]) -> np.ndarray:
    try:
        from bertopic import BERTopic
    except Exception as e:
        warnings.warn(f"BERTopic unavailable ({e}); falling back to UMAP+HDBSCAN")
        return cluster_with_hdbscan(sample_embs)

    um = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=UMAP_N_COMPONENTS,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
        verbose=True,
    )
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_epsilon=HDBSCAN_CLUSTER_SELECTION_EPSILON,
        cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
        metric="euclidean",
        prediction_data=True,
        core_dist_n_jobs=HDBSCAN_CORE_DIST_N_JOBS,
    )
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=um,
        hdbscan_model=hdb,
        seed_topic_list=seed_topic_list,
        calculate_probabilities=False,
        verbose=True,
    )
    labels, _ = topic_model.fit_transform(sample_texts, embeddings=sample_embs)
    return np.array(labels, dtype=int)


def cluster_with_fuzzy_cmeans(
    sample_embs: np.ndarray,
    n_clusters: int,
    m: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, object, object]:
    """Fit Fuzzy c-means on UMAP-reduced sample embeddings.

    Returns
    -------
    hard_labels : (n_sample,) int ndarray   – argmax of membership per doc
    membership  : (n_sample, n_clusters) float32 ndarray
    umap_model  : fitted UMAP instance (needed to transform unseen docs), or None on fallback
    cntr        : (n_clusters, n_umap_components) cluster centres in UMAP space, or None on fallback
    """
    try:
        from skfuzzy import cluster as fuzz_cluster  # type: ignore  # noqa: F401
    except ImportError:
        warnings.warn(
            "scikit-fuzzy is not installed (pip install scikit-fuzzy). "
            "Falling back to UMAP + HDBSCAN."
        )
        hdb_labels = cluster_with_hdbscan(sample_embs)
        n_c = max(1, int(len(set(hdb_labels[hdb_labels >= 0]))))
        membership = np.zeros((len(hdb_labels), n_c), dtype=np.float32)
        for i, lab in enumerate(hdb_labels):
            if 0 <= lab < n_c:
                membership[i, lab] = 1.0
        return hdb_labels, membership, None, None

    from skfuzzy import cluster as fuzz_cluster  # type: ignore

    um = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=UMAP_N_COMPONENTS,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
        verbose=True,
    )
    print(f"[fuzzy_cmeans] UMAP fitting {len(sample_embs)} samples...")
    sample_umap = um.fit_transform(sample_embs)  # (n_sample, n_components)

    # skfuzzy expects shape (n_features, n_samples)
    data = sample_umap.T.astype(np.float64)
    print(f"[fuzzy_cmeans] fitting c={n_clusters} clusters, m={m}...")
    cntr, u, _u0, _d, _jm, _p, fpc = fuzz_cluster.cmeans(
        data, c=n_clusters, m=m, error=0.005, maxiter=1000, init=None
    )
    # u: (n_clusters, n_samples) → transpose to (n_samples, n_clusters)
    membership = u.T.astype(np.float32)
    hard_labels = np.argmax(membership, axis=1).astype(int)
    print(f"[fuzzy_cmeans] done – FPC={fpc:.4f}, unique hard labels={len(np.unique(hard_labels))}")
    return hard_labels, membership, um, cntr


def _predict_fuzzy_membership_batched(
    embs_all: np.ndarray,
    umap_model,
    cntr: np.ndarray,
    m: float = 2.0,
) -> np.ndarray:
    """Predict soft membership for the full segment in FUZZY_BATCH_SIZE chunks.

    Uses the fitted UMAP model to project embeddings into UMAP space, then
    runs cmeans_predict against the stored cluster centres.

    Returns
    -------
    membership : (n_docs, n_clusters) float32 ndarray
    """
    try:
        from skfuzzy import cluster as fuzz_cluster  # type: ignore
    except ImportError:
        return None  # caller handles None

    n_docs = len(embs_all)
    n_clusters = cntr.shape[0]
    membership_all = np.empty((n_docs, n_clusters), dtype=np.float32)

    batches = range(0, n_docs, FUZZY_BATCH_SIZE)
    for start in _tqdm(batches, desc="fuzzy predict (UMAP+cmeans)", unit="batch"):
        end = min(start + FUZZY_BATCH_SIZE, n_docs)
        umap_coords = umap_model.transform(embs_all[start:end])   # (batch, n_components)
        test_data = umap_coords.T.astype(np.float64)               # (n_components, batch)
        u, _u0, _d, _jm, _p, _fpc = fuzz_cluster.cmeans_predict(
            test_data, cntr, m, error=0.005, maxiter=1000
        )
        membership_all[start:end] = u.T.astype(np.float32)

    return membership_all


# Internal token batch size for fuzzy_term_importance memory management.
# (n_vocab × FUZZY_BATCH_SIZE) similarity matrices are computed per inner iteration;
# keeping token batches small bounds peak RAM.
_FUZZY_TOKEN_BATCH = 2000


def fuzzy_term_importance(
    texts: List[str],
    embs: np.ndarray,
    membership_matrix: np.ndarray,
    embedding_model=None,
) -> pd.DataFrame:
    """Compute fuzzy term importance scores (Algorithm 1, Nikbakht & Zojaji 2026).

    For each vocabulary token t and cluster c:
        TI(t, c)    = sum_{d: t in d} [ cosine_sim(emb_t, emb_d) × membership(d, c) ]
        rank_t,c    = rank of c among all clusters by TI(t, ·) descending (1 = best)
        TIadj(t, c) = TI(t, c) × (1 / rank_t,c)

    Tokens absent from a document contribute 0 (they are excluded from the sum),
    matching the paper's sentinel value of Sim=2 (outside valid cosine range).

    Documents are processed in batches of FUZZY_BATCH_SIZE; tokens are processed
    in internal batches of _FUZZY_TOKEN_BATCH to bound peak memory usage.

    Parameters
    ----------
    texts            : raw document texts (same order as rows of embs / membership_matrix)
    embs             : pre-computed document embeddings (n_docs, dim), L2-normalised
    membership_matrix: (n_docs, n_clusters) float – output of cluster_with_fuzzy_cmeans
    embedding_model  : optional pre-loaded SentenceTransformer; loaded if None

    Returns
    -------
    DataFrame with columns [cluster_id, term, score, rank]  – same format as ctfi_df()
    """
    n_docs, n_clusters = membership_matrix.shape
    print(f"[fuzzy_ti] input: n_docs={n_docs}, n_clusters={n_clusters}, embs.shape={embs.shape}")

    if n_docs > 10_000:
        warnings.warn(
            f"[fuzzy_term_importance] corpus has {n_docs} docs (>10 000); "
            "computation may be slow – consider sampling first."
        )

    # Cluster size diagnostics (hard assignment via argmax)
    hard_assign = np.argmax(membership_matrix, axis=1)
    cluster_sizes = np.bincount(hard_assign, minlength=n_clusters)
    min_cluster_size = int(cluster_sizes[cluster_sizes > 0].min()) if cluster_sizes.any() else 0
    print(f"[fuzzy_ti] cluster sizes: min={min_cluster_size}, max={int(cluster_sizes.max())}, "
          f"mean={cluster_sizes[cluster_sizes > 0].mean():.0f}, "
          f"empty clusters={int((cluster_sizes == 0).sum())}")

    # Load embedding model if not provided
    if embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            embedding_model = SentenceTransformer("Alibaba-NLP/gte-multilingual-base")
            print("[fuzzy_ti] loaded SentenceTransformer(gte-multilingual-base)")
        except Exception as exc:
            warnings.warn(
                f"[fuzzy_term_importance] cannot load embedding model: {exc}. "
                "Returning empty DataFrame."
            )
            return pd.DataFrame(columns=["cluster_id", "term", "score", "rank"])

    # Build vocabulary — lower min_df for small segments
    # If the smallest non-empty cluster has < 500 docs, min_df=3 avoids filtering everything out
    effective_min_df = 3 if min_cluster_size < 500 else MIN_DF
    print(f"[fuzzy_ti] building vocab: min_df={effective_min_df} (MIN_DF={MIN_DF}, "
          f"min_cluster_size={min_cluster_size}), stop_words={len(STOP_WORDS)}")

    vect = CountVectorizer(
        ngram_range=(1, 1),
        min_df=effective_min_df,
        max_df=0.6,
        stop_words=STOP_WORDS,
        lowercase=True,
        token_pattern=r"(?u)\b[a-zæøå0-9]{3,}\b",
        strip_accents=None,
    )
    X = vect.fit_transform(texts)        # (n_docs, n_vocab) sparse CSR
    vocab = vect.get_feature_names_out()
    n_vocab = len(vocab)
    total_tokens_raw = X.sum()
    print(f"[fuzzy_ti] vocab after CountVectorizer: {n_vocab} unique tokens, "
          f"{int(total_tokens_raw)} total token occurrences in corpus")

    if n_vocab == 0:
        print("[fuzzy_ti] WARNING: vocab is empty after filtering — all terms removed by "
              f"min_df={effective_min_df} + stopwords. Returning empty DataFrame.")
        return pd.DataFrame(columns=["cluster_id", "term", "score", "rank"])

    # Sanity check: show 10 sample tokens so we can confirm the tokeniser is working
    sample_tokens = vocab[:10].tolist()
    print(f"[fuzzy_ti] sample vocab (first 10): {sample_tokens}")

    # Embed all vocabulary tokens
    print(f"[fuzzy_ti] encoding {n_vocab} vocabulary tokens...")
    token_embs = np.array(
        embedding_model.encode(
            vocab.tolist(), batch_size=256, show_progress_bar=True, normalize_embeddings=True
        ),
        dtype=np.float32,
    )  # (n_vocab, dim)
    print(f"[fuzzy_ti] token_embs.shape={token_embs.shape}, "
          f"norm range=[{np.linalg.norm(token_embs, axis=1).min():.3f}, "
          f"{np.linalg.norm(token_embs, axis=1).max():.3f}]")

    # Ensure doc embeddings are L2-normalised
    embs_norm = l2norm(embs.astype(np.float32).copy())  # (n_docs, dim)
    print(f"[fuzzy_ti] embs_norm.shape={embs_norm.shape}")

    # Dimension compatibility check — token_embs come from gte-multilingual-base;
    # doc embeddings come from A_embeddings.npz.  If they were produced by different
    # models their vector spaces are incompatible and cosine similarity will be ~0.
    if token_embs.shape[1] != embs_norm.shape[1]:
        warnings.warn(
            f"[fuzzy_ti] DIMENSION MISMATCH: token_embs.dim={token_embs.shape[1]} "
            f"vs doc_embs.dim={embs_norm.shape[1]}. "
            f"A_embeddings.npz was built with a different model than "
            f"gte-multilingual-base — cosine similarity is undefined. "
            f"Falling back to c-TF-IDF."
        )
        texts_by_cluster_fb: Dict[int, List[str]] = {}
        for doc_i, c_i in enumerate(hard_assign):
            texts_by_cluster_fb.setdefault(int(c_i), []).append(texts[doc_i])
        return ctfi_df(
            texts_by_cluster_fb, ngram_range=(1, 1), min_df=effective_min_df,
            stop_words=STOP_WORDS, top_k=TOP_K_TERMS,
        )

    # CSC layout for efficient column (token) slicing inside the token batch loop
    X_csc = X.tocsc()
    membership_f64 = membership_matrix.astype(np.float64)

    # Accumulate TI(t, c) = sum_{d: t in d} [ sim(t, d) × membership(d, c) ]
    ti_matrix = np.zeros((n_vocab, n_clusters), dtype=np.float64)

    n_token_batches = -(-n_vocab // _FUZZY_TOKEN_BATCH)  # ceiling division
    print(f"[fuzzy_ti] TI accumulation: {n_token_batches} token batches × "
          f"{-(-n_docs // FUZZY_BATCH_SIZE)} doc batches")
    for t_start in _tqdm(
        range(0, n_vocab, _FUZZY_TOKEN_BATCH),
        desc="fuzzy TI (token batches)",
        unit="batch",
        total=n_token_batches,
    ):
        t_end = min(t_start + _FUZZY_TOKEN_BATCH, n_vocab)
        tok_embs_batch = token_embs[t_start:t_end]          # (t_batch, dim)
        tok_X_cols = X_csc[:, t_start:t_end]                # (n_docs, t_batch) CSC

        for d_start in range(0, n_docs, FUZZY_BATCH_SIZE):
            d_end = min(d_start + FUZZY_BATCH_SIZE, n_docs)
            d_embs = embs_norm[d_start:d_end]               # (d_batch, dim)
            d_mem = membership_f64[d_start:d_end]           # (d_batch, n_clusters)

            # Cosine similarities: (t_batch, d_batch)
            sims = (tok_embs_batch @ d_embs.T).astype(np.float64)

            # On the very first mini-batch, verify similarities are non-trivial
            if t_start == 0 and d_start == 0:
                max_sim_first = float(np.abs(sims).max())
                print(f"[fuzzy_ti] first-batch max |cosine_sim| = {max_sim_first:.4f} "
                      f"(expected >0.1 if doc and token embeddings share the same model)")
                if max_sim_first < 0.05:
                    print(
                        "[fuzzy_ti] WARNING: near-zero similarities — doc embeddings are likely "
                        "from a different model than gte-multilingual-base. "
                        "All TI scores will be ~0, and the c-TF-IDF fallback will run."
                    )

            # Binary presence matrix: token t present in doc d  →  (t_batch, d_batch)
            P = (tok_X_cols[d_start:d_end] > 0).T.toarray().astype(np.float64)

            # Mask: exclude token–doc pairs where token is absent
            masked = sims * P                               # (t_batch, d_batch)

            # Accumulate TI: (t_batch, d_batch) @ (d_batch, n_clusters) → (t_batch, n_clusters)
            ti_matrix[t_start:t_end] += masked @ d_mem

    # TI matrix diagnostics
    ti_nonzero_per_cluster = (ti_matrix > 0.0).sum(axis=0)  # (n_clusters,)
    print(f"[fuzzy_ti] TI matrix shape={ti_matrix.shape}, "
          f"global max={ti_matrix.max():.4f}, global min_nonzero="
          f"{ti_matrix[ti_matrix > 0].min():.6f if (ti_matrix > 0).any() else 'N/A'}")
    print(f"[fuzzy_ti] non-zero TI tokens per cluster: "
          f"min={int(ti_nonzero_per_cluster.min())}, "
          f"max={int(ti_nonzero_per_cluster.max())}, "
          f"mean={ti_nonzero_per_cluster.mean():.1f}, "
          f"clusters with 0 tokens={int((ti_nonzero_per_cluster == 0).sum())}")

    # Rank clusters per token descending by TI (rank 1 = highest TI)
    rank_matrix = np.argsort(np.argsort(-ti_matrix, axis=1), axis=1) + 1  # (n_vocab, n_clusters)

    # Adjusted TI: TIadj(t, c) = TI(t, c) × (1 / rank_t,c)
    tiadj_matrix = ti_matrix * (1.0 / rank_matrix)         # (n_vocab, n_clusters)
    tiadj_nonzero_per_cluster = (tiadj_matrix > 0.0).sum(axis=0)
    print(f"[fuzzy_ti] TIadj matrix shape={tiadj_matrix.shape}, "
          f"non-zero TIadj tokens per cluster: min={int(tiadj_nonzero_per_cluster.min())}, "
          f"max={int(tiadj_nonzero_per_cluster.max())}")

    # Build output DataFrame (same schema as ctfi_df)
    rows = []
    for c in range(n_clusters):
        nonzero = np.where(ti_matrix[:, c] > 0.0)[0]
        if len(nonzero) == 0:
            continue
        tiadj_col = tiadj_matrix[nonzero, c]
        order = np.argsort(-tiadj_col)[:TOP_K_TERMS]
        for rank_i, pos in enumerate(order, start=1):
            t_idx = nonzero[pos]
            rows.append((int(c), vocab[t_idx], float(tiadj_matrix[t_idx, c]), rank_i))

    result = pd.DataFrame(rows, columns=["cluster_id", "term", "score", "rank"])
    print(f"[fuzzy_ti] output: {len(result)} rows, {result['cluster_id'].nunique()} clusters with terms")

    # Safety net: if TIadj produced nothing, fall back to c-TF-IDF on hard cluster assignments
    if result.empty:
        warnings.warn(
            "[fuzzy_ti] TIadj output is empty — all TI scores are zero. "
            "This likely means the embedding model token similarities are all near-zero, "
            "or there is a dtype/shape mismatch. Falling back to c-TF-IDF on hard assignments."
        )
        texts_by_cluster: Dict[int, List[str]] = {}
        for doc_i, c in enumerate(hard_assign):
            texts_by_cluster.setdefault(int(c), []).append(texts[doc_i])
        result = ctfi_df(
            texts_by_cluster,
            ngram_range=(1, 1),
            min_df=effective_min_df,
            stop_words=STOP_WORDS,
            top_k=TOP_K_TERMS,
        )
        print(f"[fuzzy_ti] c-TF-IDF fallback produced {len(result)} rows")

    return result


def build_segment_mask(idx: pd.DataFrame, gender: str, age_min: int, age_max: int) -> np.ndarray:
    if "gender_std" not in idx.columns:
        return np.zeros(len(idx), dtype=bool)
    gmask = idx["gender_std"].astype(str).str.lower().eq(gender)

    if "age" in idx.columns:
        age = pd.to_numeric(idx["age"], errors="coerce")
        amask = (age >= age_min) & (age <= age_max)
        return (gmask & amask.fillna(False)).to_numpy()

    # fallback with age_group approximation
    if age_min == 16:
        warnings.warn(
            "WARNING: age_group string boundaries may misplace 16-year-olds — "
            "verify segment membership if results look off."
        )
    ag = idx.get("age_group", pd.Series(["ukjent"] * len(idx))).astype(str)
    if age_min == 13 and age_max == 15:
        amask = ag.eq("13-16")  # includes 16 when exact age is unavailable
    else:
        amask = ag.isin(["17-19", "20+"])  # approximates 16-20 when exact age is unavailable
    return (gmask & amask).to_numpy()


def run_one_segment(
    seg_name: str,
    outdir: Path,
    idx_seg: pd.DataFrame,
    embs_seg: np.ndarray,
    age_column_used: str = "age_group_fallback",
    age_coverage_pct: float = 0.0,
) -> None:
    # Load per-segment guided topics from SCRIPT_DIR (git-tracked)
    filename = GUIDED_TOPICS_FILES.get(seg_name, GUIDED_TOPICS_FILES["_default"])
    topics_path = SCRIPT_DIR / filename
    if not topics_path.exists() and filename != GUIDED_TOPICS_FILES["_default"]:
        print(f"[{seg_name}] No segment-specific file ({filename}), "
              f"falling back to {GUIDED_TOPICS_FILES['_default']}")
        topics_path = SCRIPT_DIR / GUIDED_TOPICS_FILES["_default"]
    seed_labels, seed_topic_list = load_guided_topics(topics_path)
    print(f"[{seg_name}] Loaded {len(seed_topic_list)} seed topics from {topics_path.name}")

    n_total = len(idx_seg)
    print(f"\n[segment] {seg_name} | N={n_total}")

    # Print actual age range found in this segment (Change 3)
    if "age" in idx_seg.columns:
        age_num = pd.to_numeric(idx_seg["age"], errors="coerce")
        valid_ages = age_num.dropna()
        if len(valid_ages):
            print(
                f"Segment {seg_name}: age range in data = "
                f"{int(valid_ages.min())}–{int(valid_ages.max())}, n={n_total}"
            )
        else:
            age_vals = sorted(idx_seg["age_group"].dropna().unique().tolist()) if "age_group" in idx_seg.columns else []
            print(f"Segment {seg_name}: age_group values in data = {age_vals}, n={n_total}")
    elif "age_group" in idx_seg.columns:
        age_vals = sorted(idx_seg["age_group"].dropna().unique().tolist())
        print(f"Segment {seg_name}: age_group values in data = {age_vals}, n={n_total}")

    if n_total < max(HDBSCAN_MIN_CLUSTER_SIZE * 3, 1000):
        warnings.warn(f"Segment {seg_name} too small; skipping")
        return

    sample_target = min(SAMPLE_SIZE, n_total)
    sample_idx = stratified_sample_index(idx_seg, sample_target, MIN_PER_STRATUM, SAMPLE_RANDOM_STATE)
    if len(sample_idx) == 0:
        warnings.warn(f"Segment {seg_name} returned empty sample; skipping")
        return

    sample_embs = embs_seg[sample_idx]
    sample_texts = idx_seg.iloc[sample_idx]["body_norm"].astype(str).tolist()

    full_membership = None   # (n_docs, n_clusters) – populated only in fuzzy mode
    centroid_embs = sample_embs   # overridden to embs_seg on successful full-corpus fit
    centroid_labels: np.ndarray   # assigned in all paths below
    sampling_method = "sample_fit"
    sampling_n_used = len(sample_idx)
    transform_n_used = 0
    n_fuzzy_clusters = FUZZY_N_CLUSTERS_BY_SEGMENT.get(seg_name, FUZZY_N_CLUSTERS)  # updated inside fuzzy block

    if USE_FUZZY_BERTOPIC:
        # Per-segment cluster count; falls back to global FUZZY_N_CLUSTERS for unknown names
        seg_n_clusters = FUZZY_N_CLUSTERS_BY_SEGMENT.get(seg_name, FUZZY_N_CLUSTERS)
        print(f"[segment:{seg_name}] fuzzy n_clusters={seg_n_clusters} "
              f"(from {'FUZZY_N_CLUSTERS_BY_SEGMENT' if seg_name in FUZZY_N_CLUSTERS_BY_SEGMENT else 'FUZZY_N_CLUSTERS fallback'})")

        if USE_BERTOPIC_GUIDED and seed_topic_list:
            # Discover cluster structure with guided BERTopic, then use that count for fuzzy c-means
            guided_labels = cluster_with_bertopic_guided(sample_texts, sample_embs, seed_topic_list)
            n_fuzzy_clusters = int(len(set(guided_labels[guided_labels >= 0])))
            if n_fuzzy_clusters < 2:
                warnings.warn(
                    f"Guided BERTopic found <2 clusters for {seg_name}; "
                    f"using per-segment default n_clusters={seg_n_clusters}"
                )
                n_fuzzy_clusters = seg_n_clusters
            mode = "fuzzy_bertopic_guided"
        else:
            n_fuzzy_clusters = seg_n_clusters
            mode = "fuzzy_bertopic"

        # Attempt A — full-corpus fit (only when segment exceeds sample cap)
        full_fit_succeeded = False
        if n_total > SAMPLE_SIZE:
            print(f"Segment {seg_name}: attempting full-corpus fit (n={n_total})...")
            try:
                hard_labels_all, full_membership_fit, umap_model, cntr = cluster_with_fuzzy_cmeans(
                    embs_seg, n_fuzzy_clusters, FUZZY_M
                )
                full_membership = full_membership_fit
                centroid_embs = embs_seg
                centroid_labels = hard_labels_all
                sampling_method = "full_fit"
                sampling_n_used = n_total
                full_fit_succeeded = True
                print(f"Segment {seg_name}: full-corpus fit succeeded (n={n_total}).")
            except Exception as _oom:
                if not (isinstance(_oom, MemoryError) or "MemoryError" in type(_oom).__name__):
                    raise
                print(
                    f"Segment {seg_name}: full-corpus fit failed with MemoryError. "
                    f"Falling back to fit-on-sample + umap.transform for remainder."
                )

        # Fallback B — fit on sample, transform remainder (also the normal path when n_total <= SAMPLE_SIZE)
        if not full_fit_succeeded:
            sample_labels, _sample_membership, umap_model, cntr = cluster_with_fuzzy_cmeans(
                sample_embs, n_fuzzy_clusters, FUZZY_M
            )
            centroid_labels = sample_labels
            centroid_embs = sample_embs

            if umap_model is not None and cntr is not None:
                print(f"[segment:{seg_name}] predicting fuzzy membership for all {n_total} docs...")
                full_membership = _predict_fuzzy_membership_batched(embs_seg, umap_model, cntr, FUZZY_M)

            if n_total > SAMPLE_SIZE:
                transform_n_used = n_total - len(sample_idx)
                sampling_method = "fit_sample_transform_remainder"
                sampling_n_used = len(sample_idx)
                print(
                    f"Segment {seg_name}: fit on {len(sample_idx)} sample, "
                    f"transformed {transform_n_used} additional docs via umap.transform."
                )

    elif USE_BERTOPIC_GUIDED and seed_topic_list:
        centroid_labels = cluster_with_bertopic_guided(sample_texts, sample_embs, seed_topic_list)
        mode = "bertopic_guided"
    else:
        centroid_labels = cluster_with_hdbscan(sample_embs)
        mode = "umap_hdbscan"

    n_clusters = int(len(set(centroid_labels[centroid_labels >= 0])))
    print(f"[segment:{seg_name}] mode={mode} clusters={n_clusters} noise={(centroid_labels == -1).sum()}")

    centroids = compute_centroids(centroid_embs, centroid_labels)
    labels_all, sim_all, margin_all = assign_all(embs_seg, centroids, ASSIGN_MIN_SIM, ASSIGN_MIN_MARGIN)
    print(f"[segment:{seg_name}] assigned={(labels_all >= 0).sum()} noise={(labels_all < 0).sum()}")

    out_assign = idx_seg.copy()
    out_assign["cluster_id"] = labels_all
    out_assign["cluster_sim"] = sim_all
    out_assign["cluster_margin"] = margin_all

    # In fuzzy mode, store per-document top-3 membership values as paired
    # (cluster_id, score) columns.  This is much more useful than the previous
    # approach of storing globally-dominant cluster columns, which gave near-zero
    # values for most documents whose primary cluster was not globally dominant.
    if full_membership is not None:
        top3_idx = np.argsort(-full_membership, axis=1)[:, :3]   # (n_docs, 3)
        for rank_i, prefix in enumerate(["fuzzy_top1", "fuzzy_top2", "fuzzy_top3"]):
            out_assign[f"{prefix}_cluster"] = top3_idx[:, rank_i].astype(np.int32)
            out_assign[f"{prefix}_score"] = full_membership[
                np.arange(len(full_membership)), top3_idx[:, rank_i]
            ].astype(np.float32)

    out_assign = normalize_export_dtypes(out_assign)

    try:
        out_assign.to_parquet(outdir / "A_cluster_assignments.parquet", index=False)
    except Exception as e:
        warnings.warn(f"Parquet export failed for {seg_name} ({e}); writing CSV fallback")
        out_assign.to_csv(outdir / "A_cluster_assignments.csv", index=False, encoding="utf-8")

    unassigned = out_assign[out_assign["cluster_id"] < 0].copy().sort_values(["cluster_sim", "cluster_margin"], ascending=[False, True])
    unassigned.to_csv(outdir / "A_unassigned_questions.csv", index=False, encoding="utf-8")

    valid = out_assign[out_assign["cluster_id"] >= 0].copy()
    texts_by_cluster = {
        int(cid): [strip_greeting(s) for s in grp["body_norm"].astype(str).tolist()]
        for cid, grp in valid.groupby("cluster_id")
    }

    if texts_by_cluster:
        if USE_FUZZY_BERTOPIC and full_membership is not None:
            # Use fuzzy term importance over the full segment (all docs, not just assigned).
            # kw / kw_uni both carry the fuzzy TI scores (unigrams);
            # kw_bi falls back to c-TF-IDF bigrams for supplementary catalog column.
            all_texts_stripped = [
                strip_greeting(s) for s in idx_seg["body_norm"].astype(str).tolist()
            ]
            kw = fuzzy_term_importance(all_texts_stripped, embs_seg, full_membership)
            if kw.empty:
                # fuzzy_term_importance already tried its own internal fallback;
                # this outer guard ensures we always have keyword output.
                warnings.warn(
                    f"[segment:{seg_name}] fuzzy_term_importance returned empty even "
                    f"after internal fallback — running c-TF-IDF as final safety net"
                )
                kw = ctfi_df(
                    texts_by_cluster, ngram_range=(1, 1), min_df=MIN_DF,
                    stop_words=STOP_WORDS, top_k=TOP_K_TERMS,
                )
            kw_uni = kw.copy()
            kw_bi = ctfi_df(
                texts_by_cluster, ngram_range=(2, 2), min_df=MIN_DF,
                stop_words=STOP_WORDS, top_k=TOP_K_TERMS,
            )
        else:
            kw = ctfi_df(texts_by_cluster, ngram_range=NGRAM_RANGE, min_df=MIN_DF, stop_words=STOP_WORDS, top_k=TOP_K_TERMS)
            kw_uni = ctfi_df(texts_by_cluster, ngram_range=(1, 1), min_df=MIN_DF, stop_words=STOP_WORDS, top_k=TOP_K_TERMS)
            kw_bi = ctfi_df(texts_by_cluster, ngram_range=(2, 2), min_df=MIN_DF, stop_words=STOP_WORDS, top_k=TOP_K_TERMS)
    else:
        kw = pd.DataFrame(columns=["cluster_id", "term", "score", "rank"])
        kw_uni = kw.copy()
        kw_bi = kw.copy()

    kw.to_csv(outdir / "clusters_keywords.csv", index=False, encoding="utf-8")
    kw_uni.to_csv(outdir / "clusters_keywords_unigrams.csv", index=False, encoding="utf-8")
    kw_bi.to_csv(outdir / "clusters_keywords_bigrams.csv", index=False, encoding="utf-8")

    sizes = valid.groupby("cluster_id").size().rename("size").reset_index()
    top_terms = top_terms_per_cluster(kw, TOP_K_TERMS)
    top_unigrams = top_terms_per_cluster(kw_uni, TOP_K_TERMS)
    top_bigrams = top_terms_per_cluster(kw_bi, TOP_K_TERMS)

    rows = []
    for _, r in sizes.iterrows():
        cid = int(r["cluster_id"])
        grp = valid[valid["cluster_id"] == cid]
        exemplar = grp.iloc[0]["body_norm"] if len(grp) else ""
        rows.append({
            "cluster_id": cid,
            "size": int(r["size"]),
            "top_terms": top_terms.get(cid, ""),
            "top_unigrams": top_unigrams.get(cid, ""),
            "top_bigrams": top_bigrams.get(cid, ""),
            "exemplar_text": str(exemplar)[:400].replace("\n", " "),
        })
    pd.DataFrame(rows).sort_values("size", ascending=False).to_csv(outdir / "clusters_catalog.csv", index=False, encoding="utf-8")

    sampling_entry: dict = {
        "size": SAMPLE_SIZE,
        "min_per_stratum": MIN_PER_STRATUM,
        "seed": SAMPLE_RANDOM_STATE,
        "method": sampling_method,
        "sampling_n": sampling_n_used,
    }
    if transform_n_used > 0:
        sampling_entry["transform_n"] = transform_n_used

    params = {
        "segment": seg_name,
        "mode": mode,
        "n_total": n_total,
        "n_sample": sampling_n_used,
        "n_clusters": n_clusters,
        "n_assigned": int((labels_all >= 0).sum()),
        "n_noise": int((labels_all < 0).sum()),
        "age_column_used": age_column_used,
        "age_coverage_pct": round(age_coverage_pct * 100, 1),
        "files": {
            "guided_topics_file": str(topics_path),
            "clusters_keywords": str(outdir / "clusters_keywords.csv"),
            "clusters_keywords_unigrams": str(outdir / "clusters_keywords_unigrams.csv"),
            "clusters_keywords_bigrams": str(outdir / "clusters_keywords_bigrams.csv"),
        },
        "sampling": sampling_entry,
        "assign": {"min_sim": ASSIGN_MIN_SIM, "min_margin": ASSIGN_MIN_MARGIN},
        "guided_topics_file": str(topics_path),
        "guided_topics_loaded": len(seed_topic_list),
        "guided_topic_labels": seed_labels,
        "age_split_note": "Exact 13-15/16-20 requires numeric age column. age_group fallback is approximate.",
        "fuzzy": {
            "enabled": USE_FUZZY_BERTOPIC,
            "n_clusters": n_fuzzy_clusters if USE_FUZZY_BERTOPIC else None,
            "m": FUZZY_M if USE_FUZZY_BERTOPIC else None,
            "membership_stored": full_membership is not None,
        },
    }
    with open(outdir / "clustering_run_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


def main() -> None:
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[load] embeddings + index")
    embs = load_all_embeddings(outdir, SHARD_PREFIX)
    idx = load_aligned_index(outdir)
    idx = reconcile_index_length(outdir, idx, len(embs))
    idx, age_col_used, age_coverage = enrich_with_exact_age(idx, outdir)
    if "body_norm" not in idx.columns:
        raise ValueError("Index is missing body_norm column")

    # --- Age join checkpoint ---
    print()
    print("=" * 60)
    print(f"  Age join result:  {age_col_used}")
    print(f"  Coverage:         {age_coverage:.1%}")
    if "age" in idx.columns:
        age_num = pd.to_numeric(idx["age"], errors="coerce")
        print(f"  Numeric age rows: {int(age_num.notna().sum()):,} / {len(idx):,}")
        print(f"  Age range:        {int(age_num.min())}–{int(age_num.max())}")
        for seg in SEGMENTS:
            mask = build_segment_mask(idx, seg["gender"], seg["age_min"], seg["age_max"])
            seg_ages = age_num[mask].dropna()
            n = int(mask.sum())
            age_range = f"{int(seg_ages.min())}–{int(seg_ages.max())}" if len(seg_ages) else "n/a"
            print(f"  {seg['name']:20s}  n={n:>7,}  age range={age_range}")
    else:
        print("  No numeric age column — string fallback will be used.")
    print("=" * 60)
    ans = input("  Continue with clustering? [y/N] ").strip().lower()
    if ans not in ("y", "yes"):
        print("[abort] Exiting before clustering. Fix the age join and re-run.")
        return
    print()

    if RUN_SEGMENTED:
        base = outdir / SEGMENT_OUTPUT_SUBDIR
        base.mkdir(parents=True, exist_ok=True)
        for seg in _tqdm(SEGMENTS, desc="segments", unit="seg"):
            mask = build_segment_mask(idx, seg["gender"], seg["age_min"], seg["age_max"])
            idx_seg = idx.loc[mask].reset_index(drop=True)
            embs_seg = embs[mask]
            seg_out = base / seg["name"]
            seg_out.mkdir(parents=True, exist_ok=True)
            run_one_segment(seg["name"], seg_out, idx_seg, embs_seg,
                            age_column_used=age_col_used, age_coverage_pct=age_coverage)
    else:
        run_one_segment("all", outdir, idx.reset_index(drop=True), embs,
                        age_column_used=age_col_used, age_coverage_pct=age_coverage)

    print("[done] clustering pipeline completed")


if __name__ == "__main__":
    main()
