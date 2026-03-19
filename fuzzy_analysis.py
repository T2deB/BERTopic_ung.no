#!/usr/bin/env python3
"""
fuzzy_analysis.py — Fuzzy membership analysis for Positron pipeline.

Produces three outputs per segment:
  1. fuzzy_theme_prevalence.csv   — weighted prevalence of each topic
  2. fuzzy_cooccurrence.csv       — pairwise topic co-occurrence strengths
  3. fuzzy_topic_profiles.csv     — per-topic summary for reporting

Usage:
  python fuzzy_analysis.py --segment girls_13_15
  python fuzzy_analysis.py --segment all   (runs all four)
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = r"C:\Users\TorsteinDeBesche\NIFU\21571 Folkehelse og livsmestring - General\Ap2c Informasjonskanal for unge\Positron_project\outputs"
SEGMENT_SUBDIR = "segments"
SEGMENTS = ["boys_13_15", "boys_16_20", "girls_13_15", "girls_16_20"]


def load_segment(seg_name: str, base: Path):
    seg_dir = base / SEGMENT_SUBDIR / seg_name

    # Load assignments (contains fuzzy top-5 columns)
    parquet_path = seg_dir / "A_cluster_assignments.parquet"
    assign = pd.read_parquet(parquet_path)

    # Load params to get guided topic labels and cluster count
    params_path = seg_dir / "clustering_run_params.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    guided_labels = params.get("guided_topic_labels", [])
    n_clusters = params["n_clusters"]

    # Load full membership matrix if available
    matrix_path = seg_dir / "fuzzy_membership_matrix.npz"
    if matrix_path.exists():
        membership = np.load(matrix_path)["membership"]  # (n_docs, n_clusters)
        print(f"  Loaded full membership matrix: {membership.shape}")
    else:
        # Reconstruct from top-5 columns as fallback
        print(f"  No full matrix found — reconstructing from top-5 columns")
        n_docs = len(assign)
        membership = np.zeros((n_docs, n_clusters), dtype=np.float32)
        for rank in range(1, 6):
            cid_col = f"fuzzy_top{rank}_cluster"
            score_col = f"fuzzy_top{rank}_score"
            if cid_col in assign.columns and score_col in assign.columns:
                for i, (cid, score) in enumerate(zip(
                    pd.to_numeric(assign[cid_col], errors="coerce"),
                    pd.to_numeric(assign[score_col], errors="coerce"),
                )):
                    if pd.notna(cid) and pd.notna(score) and 0 <= int(cid) < n_clusters:
                        membership[i, int(cid)] = float(score)

    return assign, membership, guided_labels, n_clusters, params


def compute_prevalence(membership: np.ndarray, n_clusters: int, guided_labels: list) -> pd.DataFrame:
    """
    Weighted prevalence: for each cluster, sum all membership scores and
    divide by total number of questions. This gives the fractional
    'share of attention' that cluster commands across the corpus.
    """
    n_docs = membership.shape[0]
    weighted_sum = membership.sum(axis=0)  # (n_clusters,)
    prevalence = weighted_sum / n_docs

    rows = []
    for cid in range(min(n_clusters, membership.shape[1])):
        label = guided_labels[cid] if cid < len(guided_labels) else f"cluster_{cid}"
        rows.append({
            "cluster_id": cid,
            "label": label,
            "is_guided_topic": cid < len(guided_labels),
            "weighted_prevalence": round(float(prevalence[cid]), 6),
            "weighted_question_count": round(float(weighted_sum[cid]), 1),
            "n_questions_total": n_docs,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("weighted_prevalence", ascending=False)
        .reset_index(drop=True)
    )


def compute_cooccurrence(membership: np.ndarray, n_clusters: int, guided_labels: list) -> pd.DataFrame:
    """
    Pairwise co-occurrence: for each pair of clusters (i, j), compute the
    mean of min(membership_i, membership_j) across all questions.

    This measures how often a question has strong membership in BOTH topics
    simultaneously — the 'joint attention' between themes.

    Only reports pairs where at least one cluster is a guided topic.
    Sorted by co-occurrence strength descending.
    """
    n_c = min(n_clusters, membership.shape[1])
    rows = []

    for i in range(n_c):
        for j in range(i + 1, n_c):
            # Only include pairs involving at least one guided topic
            if i >= len(guided_labels) and j >= len(guided_labels):
                continue

            co = float(np.minimum(membership[:, i], membership[:, j]).mean())
            if co < 0.001:
                continue

            label_i = guided_labels[i] if i < len(guided_labels) else f"cluster_{i}"
            label_j = guided_labels[j] if j < len(guided_labels) else f"cluster_{j}"
            rows.append({
                "cluster_a": i,
                "label_a": label_i,
                "cluster_b": j,
                "label_b": label_j,
                "cooccurrence_score": round(co, 6),
                "both_guided": i < len(guided_labels) and j < len(guided_labels),
            })

    return (
        pd.DataFrame(rows)
        .sort_values("cooccurrence_score", ascending=False)
        .reset_index(drop=True)
    )


def compute_topic_profiles(
    membership: np.ndarray,
    assign: pd.DataFrame,
    guided_labels: list,
    n_clusters: int,
) -> pd.DataFrame:
    """
    Per-topic summary combining prevalence, concentration, and multi-topic rate.

    multi_topic_rate: fraction of questions where this cluster scores > 0.1
    AND at least one other cluster also scores > 0.1. Measures how often
    this theme co-occurs with other themes rather than appearing alone.
    """
    n_docs = membership.shape[0]
    rows = []

    for cid in range(min(n_clusters, membership.shape[1])):
        label = guided_labels[cid] if cid < len(guided_labels) else f"cluster_{cid}"
        col = membership[:, cid]

        # Questions where this topic has meaningful membership (> 0.1)
        active_mask = col > 0.1
        n_active = int(active_mask.sum())

        # Among active questions, how many also have another topic > 0.1?
        if n_active > 0:
            other_max = np.delete(membership[active_mask], cid, axis=1).max(axis=1)
            multi_topic_rate = float((other_max > 0.1).mean())
        else:
            multi_topic_rate = float("nan")

        n_hard_assigned = int((assign["cluster_id"] == cid).sum()) if "cluster_id" in assign.columns else 0

        rows.append({
            "cluster_id": cid,
            "label": label,
            "is_guided_topic": cid < len(guided_labels),
            "weighted_prevalence": round(float(col.sum() / n_docs), 6),
            "n_active_questions": n_active,
            "pct_active": round(n_active / n_docs * 100, 2),
            "mean_membership_when_active": round(float(col[active_mask].mean()), 4) if n_active > 0 else float("nan"),
            "multi_topic_rate": round(multi_topic_rate, 4),
            "n_hard_assigned": n_hard_assigned,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("weighted_prevalence", ascending=False)
        .reset_index(drop=True)
    )


def run_segment(seg_name: str, base: Path):
    print(f"\n{'='*60}")
    print(f"Fuzzy analysis: {seg_name}")
    print(f"{'='*60}")

    seg_dir = base / SEGMENT_SUBDIR / seg_name
    assign, membership, guided_labels, n_clusters, params = load_segment(seg_name, base)

    n_docs, n_c = membership.shape
    print(f"  n_docs={n_docs:,}, n_clusters={n_c}, n_guided={len(guided_labels)}")

    # 1. Weighted prevalence
    prevalence_df = compute_prevalence(membership, n_clusters, guided_labels)
    out1 = seg_dir / "fuzzy_theme_prevalence.csv"
    prevalence_df.to_csv(out1, index=False, encoding="utf-8")
    print(f"  Prevalence written: {out1.name} ({len(prevalence_df)} topics)")
    print(f"  Top 5 topics by weighted prevalence:")
    for _, row in prevalence_df.head(5).iterrows():
        print(f"    {row['label'][:45]:<45} {row['weighted_prevalence']:.4f}")

    # 2. Co-occurrence
    cooc_df = compute_cooccurrence(membership, n_clusters, guided_labels)
    out2 = seg_dir / "fuzzy_cooccurrence.csv"
    cooc_df.to_csv(out2, index=False, encoding="utf-8")
    print(f"  Co-occurrence written: {out2.name} ({len(cooc_df)} pairs)")
    if not cooc_df.empty:
        print(f"  Top 5 co-occurring topic pairs:")
        for _, row in cooc_df.head(5).iterrows():
            print(f"    {row['label_a'][:30]:<30} ↔ {row['label_b'][:30]:<30} {row['cooccurrence_score']:.4f}")

    # 3. Topic profiles
    profiles_df = compute_topic_profiles(membership, assign, guided_labels, n_clusters)
    out3 = seg_dir / "fuzzy_topic_profiles.csv"
    profiles_df.to_csv(out3, index=False, encoding="utf-8")
    print(f"  Topic profiles written: {out3.name}")


def main():
    parser = argparse.ArgumentParser(description="Fuzzy membership analysis per segment.")
    parser.add_argument("--segment", default="all", help="Segment name or 'all'")
    args = parser.parse_args()

    base = Path(OUTPUT_DIR)
    segs = SEGMENTS if args.segment == "all" else [args.segment]

    for seg in segs:
        try:
            run_segment(seg, base)
        except Exception as e:
            print(f"ERROR in {seg}: {e}")
            import traceback
            traceback.print_exc()

    print("\n[done] fuzzy_analysis.py completed")


if __name__ == "__main__":
    main()
