#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
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
# PATHS
# ---------------------------
OUTPUT_DIR = r"C:\Users\TorsteinDeBesche\NIFU\21571 Folkehelse og livsmestring - General\Ap2c Informasjonskanal for unge\Positron_project\outputs"
SEGMENT_NAME = "girls_13_15"

EMB_FILE = r"C:\Users\TorsteinDeBesche\NIFU\21571 Folkehelse og livsmestring - General\Ap2c Informasjonskanal for unge\Positron_project\outputs\embeddings\nb-bert\A_embeddings.npz"
ASSIGN_FILE = rf"{OUTPUT_DIR}\segments\{SEGMENT_NAME}\A_cluster_assignments.parquet"

# ---------------------------
# GLOBAL NESTED SETTINGS
# ---------------------------
RANDOM_STATE = 42

# Skip tiny parent clusters
MIN_PARENT_SIZE = 80

# Only process parent clusters with cluster_id >= 0
SKIP_NOISE_PARENT = True

# Keyword extraction
TOP_K_TERMS = 15
MIN_DF_DEFAULT = 5

CUSTOM_STOP = {
    "hei", "heisann", "hallo", "halloi", "pls", "plis", "please", "takk",
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


def get_nested_params(n_docs: int) -> dict:
    """
    Adaptive nested-clustering parameters based on parent-cluster size.

    These are starting values, not universal truths.
    """
    if n_docs < 150:
        return {
            "umap_n_neighbors": 8,
            "umap_min_dist": 0.01,
            "umap_n_components": 6,
            "hdbscan_min_cluster_size": 10,
            "hdbscan_min_samples": 4,
        }
    elif n_docs < 300:
        return {
            "umap_n_neighbors": 10,
            "umap_min_dist": 0.02,
            "umap_n_components": 6,
            "hdbscan_min_cluster_size": 15,
            "hdbscan_min_samples": 5,
        }
    elif n_docs < 700:
        return {
            "umap_n_neighbors": 12,
            "umap_min_dist": 0.02,
            "umap_n_components": 8,
            "hdbscan_min_cluster_size": 20,
            "hdbscan_min_samples": 6,
        }
    elif n_docs < 1500:
        return {
            "umap_n_neighbors": 12,
            "umap_min_dist": 0.02,
            "umap_n_components": 8,
            "hdbscan_min_cluster_size": 25,
            "hdbscan_min_samples": 8,
        }
    else:
        return {
            "umap_n_neighbors": 15,
            "umap_min_dist": 0.03,
            "umap_n_components": 10,
            "hdbscan_min_cluster_size": 35,
            "hdbscan_min_samples": 10,
        }


def ctfi_df(
    texts_per_cluster: Dict[int, List[str]],
    ngram_range=(1, 2),
    min_df=5,
    stop_words=None,
    top_k=15,
) -> pd.DataFrame:
    clusters = sorted(texts_per_cluster.keys())
    docs = [" ".join(texts_per_cluster[cid]) for cid in clusters]

    n_docs = len(docs)
    if n_docs == 0:
        return pd.DataFrame(columns=["subcluster_num", "term", "score", "rank"])

    effective_min_df = min(min_df, max(1, n_docs - 1))
    effective_max_df = 1.0 if n_docs < 5 else 0.7

    vect = CountVectorizer(
        ngram_range=ngram_range,
        min_df=effective_min_df,
        max_df=effective_max_df,
        stop_words=stop_words,
        lowercase=True,
        token_pattern=r"(?u)\b[a-zæøå0-9]{3,}\b",
        strip_accents=None,
    )

    try:
        X = vect.fit_transform(docs)
    except ValueError:
        return pd.DataFrame(columns=["subcluster_num", "term", "score", "rank"])

    vocab = vect.get_feature_names_out()
    if len(vocab) == 0:
        return pd.DataFrame(columns=["subcluster_num", "term", "score", "rank"])

    tf = normalize(X, norm="l1", axis=1, copy=False)
    df = (X > 0).sum(axis=0).A1
    idf = np.log((len(clusters) + 1) / (1 + df)) + 1.0
    scores = tf @ spdiags(idf, 0, X.shape[1], X.shape[1])

    rows = []
    for i, cid in enumerate(clusters):
        row = scores.getrow(i).toarray().ravel()
        if row.size == 0 or (row <= 0).all():
            continue
        top_idx = np.argsort(-row)[:top_k]
        for rank, j in enumerate(top_idx, start=1):
            rows.append((int(cid), vocab[j], float(row[j]), int(rank)))

    return pd.DataFrame(rows, columns=["subcluster_num", "term", "score", "rank"])


def top_terms_per_cluster(kw_df: pd.DataFrame, top_k: int) -> Dict[int, str]:
    if kw_df.empty:
        return {}
    return (
        kw_df.sort_values(["subcluster_num", "rank"])
        .groupby("subcluster_num")["term"]
        .apply(lambda s: ", ".join(s.tolist()[:top_k]))
        .to_dict()
    )


def build_parent_cluster_list(df: pd.DataFrame) -> List[Tuple[int, int]]:
    sizes = df.groupby("cluster_id").size().rename("size").reset_index()
    if SKIP_NOISE_PARENT:
        sizes = sizes[sizes["cluster_id"] >= 0]
    sizes = sizes[sizes["size"] >= MIN_PARENT_SIZE]
    sizes = sizes.sort_values("size", ascending=False)
    return [(int(r.cluster_id), int(r.size)) for r in sizes.itertuples(index=False)]


def run_subclustering_for_parent(
    parent_cluster: int,
    df: pd.DataFrame,
    embs: np.ndarray,
    out_base: Path,
) -> dict:
    sub = df[df["cluster_id"] == parent_cluster].copy()
    if sub.empty:
        print(f"[parent {parent_cluster}] no rows found, skipping")
        return {"parent_cluster": parent_cluster, "status": "empty"}

    sub_idx = sub.index.to_numpy()
    sub_embs = embs[sub_idx].astype("float32")
    sub_embs = l2norm(sub_embs.copy())

    n_docs = len(sub)
    params = get_nested_params(n_docs)

    print(
        f"\n[parent {parent_cluster}] n_docs={n_docs} | "
        f"UMAP(n_neighbors={params['umap_n_neighbors']}, min_dist={params['umap_min_dist']}, n_components={params['umap_n_components']}) | "
        f"HDBSCAN(min_cluster_size={params['hdbscan_min_cluster_size']}, min_samples={params['hdbscan_min_samples']})"
    )

    um = umap.UMAP(
        n_neighbors=params["umap_n_neighbors"],
        min_dist=params["umap_min_dist"],
        n_components=params["umap_n_components"],
        metric="cosine",
        random_state=RANDOM_STATE,
        verbose=True,
    )
    X_um = um.fit_transform(sub_embs)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=params["hdbscan_min_cluster_size"],
        min_samples=params["hdbscan_min_samples"],
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    sub_labels = clusterer.fit_predict(X_um)

    n_subclusters = len(set(sub_labels[sub_labels >= 0]))
    n_noise = int((sub_labels < 0).sum())

    print(f"[parent {parent_cluster}] subclusters={n_subclusters}, noise={n_noise}")

    sub["parent_cluster"] = parent_cluster
    sub["subcluster_num"] = sub_labels
    sub["subcluster_id"] = sub["subcluster_num"].apply(
        lambda x: f"{parent_cluster}_{x}" if x >= 0 else f"{parent_cluster}_noise"
    )

    out_dir = out_base / f"parent_{parent_cluster}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sub.to_parquet(out_dir / "subcluster_assignments.parquet", index=False)

    valid = sub[sub["subcluster_num"] >= 0].copy()
    texts_by_cluster = {
        int(cid): grp["body_norm"].astype(str).tolist()
        for cid, grp in valid.groupby("subcluster_num")
    }

    if texts_by_cluster:
        kw = ctfi_df(
            texts_by_cluster,
            ngram_range=(1, 2),
            min_df=MIN_DF_DEFAULT,
            stop_words=STOP_WORDS,
            top_k=TOP_K_TERMS,
        )
        kw_uni = ctfi_df(
            texts_by_cluster,
            ngram_range=(1, 1),
            min_df=MIN_DF_DEFAULT,
            stop_words=STOP_WORDS,
            top_k=TOP_K_TERMS,
        )
        kw_bi = ctfi_df(
            texts_by_cluster,
            ngram_range=(2, 2),
            min_df=MIN_DF_DEFAULT,
            stop_words=STOP_WORDS,
            top_k=TOP_K_TERMS,
        )
    else:
        kw = pd.DataFrame(columns=["subcluster_num", "term", "score", "rank"])
        kw_uni = kw.copy()
        kw_bi = kw.copy()

    kw.to_csv(out_dir / "subcluster_keywords.csv", index=False, encoding="utf-8")
    kw_uni.to_csv(out_dir / "subcluster_keywords_unigrams.csv", index=False, encoding="utf-8")
    kw_bi.to_csv(out_dir / "subcluster_keywords_bigrams.csv", index=False, encoding="utf-8")

    sizes = valid.groupby("subcluster_num").size().rename("size").reset_index()
    top_terms = top_terms_per_cluster(kw, TOP_K_TERMS)
    top_uni = top_terms_per_cluster(kw_uni, TOP_K_TERMS)
    top_bi = top_terms_per_cluster(kw_bi, TOP_K_TERMS)

    rows = []
    for _, r in sizes.iterrows():
        cid = int(r["subcluster_num"])
        grp = valid[valid["subcluster_num"] == cid]
        exemplar = grp.iloc[0]["body_norm"] if len(grp) else ""
        rows.append({
            "parent_cluster": parent_cluster,
            "subcluster_num": cid,
            "subcluster_id": f"{parent_cluster}_{cid}",
            "size": int(r["size"]),
            "top_terms": top_terms.get(cid, ""),
            "top_unigrams": top_uni.get(cid, ""),
            "top_bigrams": top_bi.get(cid, ""),
            "exemplar_text": str(exemplar)[:500].replace("\n", " "),
        })

    catalog = pd.DataFrame(rows).sort_values("size", ascending=False)
    catalog.to_csv(out_dir / "subclusters_catalog.csv", index=False, encoding="utf-8")

    meta = {
        "parent_cluster": parent_cluster,
        "n_docs": n_docs,
        "n_subclusters": n_subclusters,
        "n_noise": n_noise,
        "params": params,
    }
    with open(out_dir / "subclustering_params.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "parent_cluster": parent_cluster,
        "n_docs": n_docs,
        "n_subclusters": n_subclusters,
        "n_noise": n_noise,
        "status": "ok",
        **params,
    }


def main():
    out_base = Path(OUTPUT_DIR) / "segments" / SEGMENT_NAME / "nested_topics"
    out_base.mkdir(parents=True, exist_ok=True)

    print("[load] assignments + embeddings")
    df = pd.read_parquet(ASSIGN_FILE)
    embs = np.load(EMB_FILE)["embs"].astype("float32")

    parent_clusters = build_parent_cluster_list(df)
    print(f"[info] parent clusters to process: {len(parent_clusters)}")

    summary_rows = []
    for parent_cluster, size in parent_clusters:
        result = run_subclustering_for_parent(parent_cluster, df, embs, out_base)
        summary_rows.append(result)

    summary_df = pd.DataFrame(summary_rows).sort_values("n_docs", ascending=False)
    summary_df.to_csv(out_base / "nested_topics_summary.csv", index=False, encoding="utf-8")

    print("\n[done] nested topic analysis complete")


if __name__ == "__main__":
    main()