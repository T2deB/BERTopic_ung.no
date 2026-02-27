#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bottom-up clustering on all embeddings with optional guided seeding.

Highlights:
- Loads full embedding set from A_embeddings.npz or stitches all shard files.
- Uses C_candidates_index.parquet as row-aligned index for all embeddings.
- Optional BERTopic guided fitting from external JSON seed file.
- Uses similarity + margin thresholds when assigning all rows to centroids.
- Exports explicit unassigned/noise rows for review.
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

# ---------------------------
# CONFIG (edit here)
# ---------------------------
OUTPUT_DIR = r"C:\Users\TorsteinDeBesche\NIFU\21571 Folkehelse og livsmestring - General\Ap2c Informasjonskanal for unge\Positron_project\outputs"
SHARD_PREFIX = "bodynorm"
A_EMB_FILE = "A_embeddings.npz"
A_INDEX_FILE = "C_candidates_index.parquet"  # row-aligned with A embeddings from original pipeline

GUIDED_TOPICS_JSON = "guided_topics.json"  # create/edit this file in OUTPUT_DIR
USE_BERTOPIC_GUIDED = True

# --- Sampling for clustering ---
SAMPLE_SIZE = 180_000
MIN_PER_STRATUM = 50
SAMPLE_RANDOM_STATE = 42

# --- UMAP params ---
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.08
UMAP_N_COMPONENTS = 15
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42

# --- HDBSCAN params ---
HDBSCAN_MIN_CLUSTER_SIZE = 100
HDBSCAN_MIN_SAMPLES = 30
HDBSCAN_CLUSTER_SELECTION_EPSILON = 0.0
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"

# --- Assignment thresholds ---
ASSIGN_MIN_SIM = 0.44
ASSIGN_MIN_MARGIN = 0.015

# --- c-TF-IDF params ---
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


# ---------------------------
# utils
# ---------------------------
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
        raise FileNotFoundError(f"Missing {idx_path}. This is required to map embeddings to text/meta.")
    return pd.read_parquet(idx_path)


def ensure_strat_cols(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "createdAt" not in x.columns:
        x["createdAt"] = "unknown"
    if "age_group" not in x.columns:
        x["age_group"] = "ukjent"
    if "gender_std" not in x.columns:
        x["gender_std"] = "ikke oppgitt"
    x["createdAt"] = x["createdAt"].astype(str)
    x["age_group"] = x["age_group"].astype(str)
    x["gender_std"] = x["gender_std"].astype(str)
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
        m = (
            (keyed["year"] == row["year"])
            & (keyed["age_group"] == row["age_group"])
            & (keyed["gender_std"] == row["gender_std"])
        )
        idxs = keyed.loc[m, "index"].to_numpy()
        k = int(row["take"])
        if k <= 0:
            continue
        chosen.append(idxs if len(idxs) <= k else rng.choice(idxs, size=k, replace=False))

    if not chosen:
        return np.arange(0)
    return np.unique(np.concatenate(chosen))


def load_guided_topics(path: Path) -> Tuple[List[str], List[List[str]]]:
    if not path.exists():
        template = {
            "topics": [
                {
                    "label": "example_sleep",
                    "keywords": ["søvn", "sove", "trøtt", "døgnrytme"],
                }
            ]
        }
        path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
        return [], []

    raw = json.loads(path.read_text(encoding="utf-8"))
    topics = raw.get("topics", [])
    labels, seed_topic_list = [], []
    for t in topics:
        label = str(t.get("label", "")).strip()
        kws = [str(x).strip().lower() for x in t.get("keywords", []) if str(x).strip()]
        if not label or not kws:
            continue
        labels.append(label)
        seed_topic_list.append(kws)
    return labels, seed_topic_list


def compute_centroids(embs: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    uniq = np.unique(labels[labels >= 0])
    centroids: Dict[int, np.ndarray] = {}
    for lab in uniq:
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

    if sims.shape[1] > 1:
        second_sim = np.partition(sims, -2, axis=1)[:, -2]
    else:
        second_sim = np.zeros_like(best_sim)
    margin = best_sim - second_sim

    out_labels = np.array([labs[k] for k in best_k], dtype=int)
    reject = (best_sim < float(min_sim)) | (margin < float(min_margin))
    out_labels[reject] = -1
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
    )
    return clusterer.fit_predict(sample_umap)


def cluster_with_bertopic_guided(sample_texts: List[str], sample_embs: np.ndarray, seed_topic_list: List[List[str]]) -> np.ndarray:
    try:
        from bertopic import BERTopic
    except Exception as e:
        warnings.warn(f"BERTopic unavailable ({e}); falling back to UMAP+HDBSCAN.")
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


def main() -> None:
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[load] embeddings + index")
    embs = load_all_embeddings(outdir, SHARD_PREFIX)
    idx = load_aligned_index(outdir)
    if len(embs) != len(idx):
        raise ValueError(f"Length mismatch: embeddings={len(embs)} index={len(idx)}. Rebuild index/embeddings alignment first.")
    if "body_norm" not in idx.columns:
        raise ValueError("Index is missing body_norm column; required for keywords and export.")
    print(f"[load] N={len(embs)} dim={embs.shape[1]}")

    print("[seeds] loading guided topic JSON")
    seed_labels, seed_topic_list = load_guided_topics(outdir / GUIDED_TOPICS_JSON)
    print(f"[seeds] loaded {len(seed_topic_list)} topics from {GUIDED_TOPICS_JSON}")

    print("[sample] stratified sample")
    sample_idx = stratified_sample_index(idx, SAMPLE_SIZE, MIN_PER_STRATUM, SAMPLE_RANDOM_STATE)
    if len(sample_idx) == 0:
        raise RuntimeError("Sampling returned 0 rows.")
    sample_embs = embs[sample_idx]
    sample_texts = idx.iloc[sample_idx]["body_norm"].astype(str).tolist()
    print(f"[sample] {len(sample_idx)} / {len(idx)}")

    print("[cluster] fitting on sample")
    if USE_BERTOPIC_GUIDED and seed_topic_list:
        sample_labels = cluster_with_bertopic_guided(sample_texts, sample_embs, seed_topic_list)
        print("[cluster] mode=bertopic_guided")
    else:
        sample_labels = cluster_with_hdbscan(sample_embs)
        print("[cluster] mode=umap_hdbscan")

    n_clusters = int(len(set(sample_labels[sample_labels >= 0])))
    print(f"[cluster] discovered={n_clusters} noise={(sample_labels == -1).sum()}")

    centroids = compute_centroids(sample_embs, sample_labels)
    print(f"[centroids] {len(centroids)}")

    print("[assign] assigning all embeddings")
    labels_all, sim_all, margin_all = assign_all(embs, centroids, ASSIGN_MIN_SIM, ASSIGN_MIN_MARGIN)
    print(f"[assign] assigned={(labels_all >= 0).sum()} noise={(labels_all < 0).sum()}")

    out_assign = idx.copy()
    out_assign["cluster_id"] = labels_all
    out_assign["cluster_sim"] = sim_all
    out_assign["cluster_margin"] = margin_all
    out_assign.to_parquet(outdir / "A_cluster_assignments.parquet", index=False)

    unassigned = out_assign[out_assign["cluster_id"] < 0].copy()
    unassigned = unassigned.sort_values(["cluster_sim", "cluster_margin"], ascending=[False, True])
    unassigned.to_csv(outdir / "A_unassigned_questions.csv", index=False, encoding="utf-8")
    print(f"[export] A_unassigned_questions.csv rows={len(unassigned)}")

    valid = out_assign[out_assign["cluster_id"] >= 0].copy()
    texts_by_cluster = {
        int(cid): [strip_greeting(s) for s in grp["body_norm"].astype(str).tolist()]
        for cid, grp in valid.groupby("cluster_id")
    }

    if texts_by_cluster:
        kw = ctfi_df(texts_by_cluster, ngram_range=NGRAM_RANGE, min_df=MIN_DF, stop_words=STOP_WORDS, top_k=TOP_K_TERMS)
    else:
        kw = pd.DataFrame(columns=["cluster_id", "term", "score", "rank"])
    kw.to_csv(outdir / "clusters_keywords.csv", index=False, encoding="utf-8")

    sizes = valid.groupby("cluster_id").size().rename("size").reset_index()
    top_terms = (
        kw.sort_values(["cluster_id", "rank"]).groupby("cluster_id")["term"].apply(lambda s: ", ".join(s.tolist()[:TOP_K_TERMS])).to_dict()
        if len(kw)
        else {}
    )
    catalog_rows = []
    for _, r in sizes.iterrows():
        cid = int(r["cluster_id"])
        grp = valid[valid["cluster_id"] == cid]
        exemplar = grp.iloc[0]["body_norm"] if len(grp) else ""
        catalog_rows.append({
            "cluster_id": cid,
            "size": int(r["size"]),
            "top_terms": top_terms.get(cid, ""),
            "exemplar_text": str(exemplar)[:400].replace("\n", " "),
        })
    pd.DataFrame(catalog_rows).sort_values("size", ascending=False).to_csv(outdir / "clusters_catalog.csv", index=False, encoding="utf-8")

    params = {
        "files": {
            "embeddings": str(outdir / A_EMB_FILE),
            "index": str(outdir / A_INDEX_FILE),
            "guided_topics_json": str(outdir / GUIDED_TOPICS_JSON),
        },
        "sampling": {"size": SAMPLE_SIZE, "min_per_stratum": MIN_PER_STRATUM, "seed": SAMPLE_RANDOM_STATE},
        "umap": {
            "n_neighbors": UMAP_N_NEIGHBORS,
            "min_dist": UMAP_MIN_DIST,
            "n_components": UMAP_N_COMPONENTS,
            "metric": UMAP_METRIC,
            "random_state": UMAP_RANDOM_STATE,
        },
        "hdbscan": {
            "min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
            "min_samples": HDBSCAN_MIN_SAMPLES,
            "cluster_selection_epsilon": HDBSCAN_CLUSTER_SELECTION_EPSILON,
            "cluster_selection_method": HDBSCAN_CLUSTER_SELECTION_METHOD,
        },
        "assign": {"min_sim": ASSIGN_MIN_SIM, "min_margin": ASSIGN_MIN_MARGIN},
        "guided_topics_loaded": len(seed_topic_list),
        "guided_topic_labels": seed_labels,
    }
    with open(outdir / "clustering_run_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    print("[done] completed clustering pipeline")


if __name__ == "__main__":
    main()
