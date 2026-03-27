#!/usr/bin/env python3
"""
generate_embeddings.py — Generate sentence embeddings for the Positron corpus
using a configurable sentence-transformers model. Saves embeddings in a
model-specific subfolder so different models can be compared.

Recommended default:
  nb-bert — NbAiLab/nb-sbert-base (Norwegian sentence embedding model)

Usage:
  python generate_embeddings.py
  python generate_embeddings.py --model nb-bert
  python generate_embeddings.py --model gte
  python generate_embeddings.py --batch-size 64
  python generate_embeddings.py --force

Outputs:
  <OUTPUT_DIR>/embeddings/<model_name>/A_embeddings.npz
  <OUTPUT_DIR>/embeddings/<model_name>/embedding_run_params.json

Optional temporary chunk files:
  <OUTPUT_DIR>/embeddings/<model_name>/tmp_chunks/chunk_00000.npy
  ...

To use with bottom_up_clustering.py, set:
  A_EMB_FILE to the model-specific A_embeddings.npz file.
"""

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

OUTPUT_DIR = r"C:\Users\TorsteinDeBesche\NIFU\21571 Folkehelse og livsmestring - General\Ap2c Informasjonskanal for unge\Positron_project\outputs"
INDEX_FILE = r"C:\Users\TorsteinDeBesche\NIFU\21571 Folkehelse og livsmestring - General\Ap2c Informasjonskanal for unge\Positron_project\outputs\C_candidates_index.parquet"
BATCH_SIZE = 64

MODEL_CONFIGS = {
    "gte": {
        "model_id": "Alibaba-NLP/gte-multilingual-base",
        "description": "Strong multilingual embedding model",
        "normalize": True,
        "max_length": 512,
    },
    "nb-bert": {
        "model_id": "NbAiLab/nb-sbert-base",
        "description": "Norwegian sentence-BERT from National Library",
        "normalize": True,
        "max_length": 512,
    },
    "norbert": {
        "model_id": "ltg/norbert3-large",
        "description": "NorBERT3 encoder; not ideal unless you explicitly want pooled base-model embeddings",
        "normalize": True,
        "max_length": 512,
    },
}


def load_index(index_path: str) -> pd.DataFrame:
    idx = pd.read_parquet(index_path)
    if "body_norm" not in idx.columns:
        raise ValueError(f"Index missing body_norm column. Columns: {idx.columns.tolist()}")
    print(f"Loaded index: {len(idx):,} rows")
    return idx


def save_params(params_path: Path, params: dict) -> None:
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)


def cleanup_temp_chunks(tmp_dir: Path) -> None:
    if tmp_dir.exists():
        for fp in tmp_dir.glob("chunk_*.npy"):
            fp.unlink()


def embed_texts(
    texts: list[str],
    model_id: str,
    normalize: bool,
    max_length: int,
    batch_size: int,
    save_chunks: bool,
    tmp_dir: Path,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {model_id}")
    model = SentenceTransformer(model_id, trust_remote_code=True)
    model.max_seq_length = max_length

    print(f"Model loaded.")
    print(f"Texts        : {len(texts):,}")
    print(f"Batch size   : {batch_size}")
    print(f"Max length   : {max_length}")
    print(f"Normalize    : {normalize}")

    n_batches = (len(texts) + batch_size - 1) // batch_size
    all_embs = []

    if save_chunks:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Temporary chunks will be saved in: {tmp_dir}")

    for batch_idx, start in enumerate(tqdm(range(0, len(texts), batch_size), total=n_batches, desc="embedding")):
        batch = texts[start:start + batch_size]

        try:
            embs = model.encode(
                batch,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)
        except Exception:
            print(f"\nEmbedding failed at batch {batch_idx} (rows {start} to {start + len(batch) - 1})")
            if batch:
                print(f"First text char length in failed batch: {len(batch[0])}")
            traceback.print_exc()
            raise

        if batch_idx == 0 and len(batch) > 0:
            print(f"First batch embedded successfully.")
            print(f"Embedding dim: {embs.shape[1]}")
            print(f"First text char length: {len(batch[0])}")

        if save_chunks:
            chunk_path = tmp_dir / f"chunk_{batch_idx:05d}.npy"
            np.save(chunk_path, embs)
        else:
            all_embs.append(embs)

    if save_chunks:
        print("Re-loading chunk files and combining into one array...")
        chunk_files = sorted(tmp_dir.glob("chunk_*.npy"))
        if not chunk_files:
            raise RuntimeError("No temporary chunk files found after embedding.")
        all_embs = [np.load(fp) for fp in tqdm(chunk_files, desc="loading chunks")]

    return np.vstack(all_embs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="nb-bert",
        choices=list(MODEL_CONFIGS.keys()),
        help="Which embedding model to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Encoding batch size (default {BATCH_SIZE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing embedding file without prompting",
    )
    parser.add_argument(
        "--save-chunks",
        action="store_true",
        help="Save batch embeddings as temporary .npy chunk files during the run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: only embed the first N texts for testing",
    )
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]

    print(f"\nEmbedding model : {args.model}")
    print(f"Model ID        : {cfg['model_id']}")
    print(f"Description     : {cfg['description']}")

    out_dir = Path(OUTPUT_DIR) / "embeddings" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "A_embeddings.npz"
    params_path = out_dir / "embedding_run_params.json"
    tmp_dir = out_dir / "tmp_chunks"

    if out_path.exists() and not args.force:
        print(f"\nEmbeddings already exist at:\n{out_path}")
        print("Use --force to overwrite.")
        return

    idx = load_index(INDEX_FILE)
    texts = idx["body_norm"].fillna("").astype(str).tolist()

    if args.limit is not None:
        texts = texts[:args.limit]
        print(f"Using only first {len(texts):,} texts because --limit was set.")

    if len(texts) == 0:
        raise ValueError("No texts found to embed.")

    t0 = time.time()

    try:
        embs = embed_texts(
            texts=texts,
            model_id=cfg["model_id"],
            normalize=cfg["normalize"],
            max_length=cfg["max_length"],
            batch_size=args.batch_size,
            save_chunks=args.save_chunks,
            tmp_dir=tmp_dir,
        )
    except Exception:
        elapsed = time.time() - t0
        fail_params = {
            "status": "failed",
            "model_key": args.model,
            "model_id": cfg["model_id"],
            "description": cfg["description"],
            "n_texts_attempted": len(texts),
            "normalize": cfg["normalize"],
            "batch_size": args.batch_size,
            "max_length": cfg["max_length"],
            "elapsed_min": round(elapsed / 60, 2),
        }
        save_params(params_path, fail_params)
        print(f"\nFailure details saved to: {params_path}")
        raise

    elapsed = time.time() - t0

    print(f"\nEmbedding complete: shape={embs.shape}, time={elapsed / 60:.1f} min")
    print(f"Saving compressed embeddings to:\n{out_path}")
    np.savez_compressed(str(out_path), embs=embs)

    params = {
        "status": "success",
        "model_key": args.model,
        "model_id": cfg["model_id"],
        "description": cfg["description"],
        "n_texts": len(texts),
        "emb_dim": int(embs.shape[1]),
        "normalize": cfg["normalize"],
        "batch_size": args.batch_size,
        "max_length": cfg["max_length"],
        "elapsed_min": round(elapsed / 60, 2),
        "save_chunks": args.save_chunks,
    }
    save_params(params_path, params)

    if args.save_chunks:
        cleanup_temp_chunks(tmp_dir)
        print("Temporary chunk files deleted after successful final save.")

    print(f"Params saved: {params_path}")
    print("\nTo use these embeddings in bottom_up_clustering.py:")
    print(f'  Set A_EMB_FILE = r"{out_path}"')


if __name__ == "__main__":
    main()