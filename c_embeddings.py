# make_C_embeddings.py
from pathlib import Path
import numpy as np, pandas as pd, json, re

OUTPUT_DIR = r"C:\Users\TorsteinDeBesche\NIFU\21571 Folkehelse og livsmestring - General\Ap2c Informasjonskanal for unge\Positron_project\outputs"
SHARD_PREFIX = "bodynorm"

outs = Path(OUTPUT_DIR)

# 1) Load A embeddings (prefer stitched; else stitch shards now)
a_npz = outs / "A_embeddings.npz"
if a_npz.exists():
    embs = np.load(a_npz)["embs"]
else:
    shards = sorted(outs.glob(f"{SHARD_PREFIX}_embeddings_*_*.npz"),
                    key=lambda p: int(re.search(r"_(\d+)_\d+\.npz$", p.name).group(1)))
    assert shards, "No A_embeddings.npz and no shard files found."
    arrays = [np.load(p)["embs"] for p in shards]
    embs = np.vstack(arrays).astype("float32")

# 2) Load candidate index (order used when computing embs)
idx_path = outs / "C_candidates_index.parquet"
assert idx_path.exists(), "Missing C_candidates_index.parquet (written by 01-script)."
idx_df = pd.read_parquet(idx_path)  # columns: body_norm, createdAt, age_group, gender_std
# Build a fast lookup: (body_norm, createdAt) -> row_idx
# createdAt format should match what 01 wrote (string or ISO-like)
idx_df = idx_df.reset_index().rename(columns={"index":"row_idx"})
key_idx = idx_df[["row_idx","body_norm","createdAt"]].copy()

# 3) Load representatives (C)
c_csv = outs / "C_dedup_representatives.csv"
assert c_csv.exists(), "Missing C_dedup_representatives.csv (written by 01-script)."
C = pd.read_csv(c_csv, dtype=str)

# 4) Join reps to row indices by (body_norm, createdAt).
#    If createdAt types differ, normalize to strings.
key_idx["createdAt"] = key_idx["createdAt"].astype(str)
C["createdAt"] = C["createdAt"].astype(str)
C_keyed = C.merge(key_idx, on=["body_norm","createdAt"], how="left", validate="m:1")

# Fallback: if some row_idx are NaN (rare), try matching only on body_norm (first match)
if C_keyed["row_idx"].isna().any():
    miss = C_keyed[C_keyed["row_idx"].isna()][["body_norm"]].drop_duplicates()
    bm = miss.merge(idx_df[["row_idx","body_norm"]], on="body_norm", how="left")
    C_keyed = C_keyed.drop(columns=["row_idx"]).merge(bm, on="body_norm", how="left")

assert not C_keyed["row_idx"].isna().any(), "Could not map some representatives to row indices."

rep_idxs = C_keyed["row_idx"].astype(int).tolist()
rep_idxs_sorted = sorted(rep_idxs)  # order doesn't matter for slicing, but is tidy

# 5) Slice and save C embeddings
C_embs = embs[rep_idxs_sorted]
np.savez_compressed(outs / "C_embeddings.npz", embs=C_embs.astype("float32"))

# 6) Save a tiny manifest
manifest = {
    "n_A": int(embs.shape[0]),
    "n_C": int(C_embs.shape[0]),
    "rep_idxs": [int(i) for i in rep_idxs_sorted]
}
(outs / "C_embeddings_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print(f"OK: wrote C_embeddings.npz with shape {C_embs.shape} and C_embeddings_manifest.json")
