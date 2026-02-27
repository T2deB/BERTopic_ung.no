# -*- coding: utf-8 -*-
"""
bottom_up_clustering.py
[...see docstring in previous cell for full description...]
"""
import os, json, math, re, warnings
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import umap, hdbscan
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import spdiags
from nltk.corpus import stopwords

# ---------------------------
# CONFIG (edit here)
# ---------------------------
OUTPUT_DIR   = r"C:\Users\TorsteinDeBesche\NIFU\21571 Folkehelse og livsmestring - General\Ap2c Informasjonskanal for unge\Positron_project\outputs"
C_EMB_FILE   = "C_embeddings.npz"
C_INDEX_FILE = "C_index.parquet"

# --- Sampling for clustering ---
SAMPLE_SIZE           = 150_000
MIN_PER_STRATUM       = 50
SAMPLE_RANDOM_STATE   = 42

# --- UMAP params ---
UMAP_N_NEIGHBORS      = 30
UMAP_MIN_DIST         = 0.1
UMAP_N_COMPONENTS     = 10
UMAP_METRIC           = "cosine"
UMAP_RANDOM_STATE     = 42

# --- HDBSCAN params ---
HDBSCAN_MIN_CLUSTER_SIZE = 120
HDBSCAN_MIN_SAMPLES      = 40
HDBSCAN_CLUSTER_SELECTION_EPSILON = 0.0
HDBSCAN_CLUSTER_SELECTION_METHOD  = "eom"

# --- Assignment threshold ---
ASSIGN_MIN_SIM       = 0.35

# --- c-TF-IDF params ---
NGRAM_RANGE          = (1, 2)
MIN_DF               = 10
TOP_K_TERMS          = 15
STOP_WORDS           = stopwords.words('norwegian')

# Ekstra norske stoppord vi ser ofte i ung.no-data
CUSTOM_STOP = {
    # hilsener/fyll
    "hei","heisann","hallo","halloi","pls","plis","please","takk","påforhånd",
    # vanlige fyllord/gradord
    "veldig","ganske","litt","mye","egentlig","faktisk","kanskje","liksom","bare","helt",
    # hjelpe-/modalverb og generelle verb
    "må","måtte","kan","kunne","skal","skulle","vil","ville","bør","burde",
    "gjør","gjøre","gjorde","får","få","går","gå","kommer","komme","kom",
    "tar","ta","tok","føler","føle","vet","veta","hatt","har","hadde","blir","bli","ble","vært","er","var",
    # svært generelle substantiv/ord
    "ting","andre","mange","noe","noen","gang","år","dag","uke","måte","grunn","type","greie","greit",
    # spørreord/lenkeord (ofte lite temabærende i nøkkelord)
    "hva","hvordan","hvorfor","hvem","hvor","når","fordi","men","også","derfor","altså","så","om",
    # pronomen (vanligvis lite informative som nøkkelord)
    "jeg","meg","min","mitt","mine","vi","oss","vår","vårt","våre",
    "du","deg","din","ditt","dine","dere","deres",
    "han","hun","hen","de","dem","deres","man","en","den","det","dette","disse","slik","sånn",
}

import unicodedata
def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).lower().strip()

STOP_WORDS = sorted({_norm(w) for w in STOP_WORDS})  # sikre lower+NFKC


# Kombiner NLTK + custom (og fjern duplikater)
try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("norwegian"))
except Exception:
    STOP_WORDS = set()
STOP_WORDS |= CUSTOM_STOP
STOP_WORDS = sorted(STOP_WORDS)



# --- Misc ---
VERBOSE_PRINT        = True

def vprint(*args, **kwargs):
    if VERBOSE_PRINT:
        print(*args, **kwargs)

def l2norm(x: np.ndarray) -> np.ndarray:
    return normalize(x, norm="l2", axis=1, copy=False)

def stratified_sample_index(df: pd.DataFrame, n_target: int, min_per_stratum: int = 0, seed: int = 42) -> np.ndarray:
    sdf = df.copy()
    sdf["year"] = sdf["createdAt"].astype(str).str[:4]
    strata = sdf.groupby(["year","age_group","gender_std"]).size().reset_index(name="n")
    total = strata["n"].sum()
    if n_target >= total:
        return np.arange(len(df))

    strata["take"] = (strata["n"] / total * n_target).round().astype(int)
    if min_per_stratum and min_per_stratum > 0:
        strata["take"] = strata[["take","n"]].apply(lambda r: min(max(r["take"], min_per_stratum), r["n"]), axis=1)

    diff = n_target - strata["take"].sum()
    if diff != 0:
        order = strata.sort_values("n", ascending=(diff<0)).index.tolist()
        i = 0
        while diff != 0 and i < len(order):
            idx = order[i]
            cur = strata.at[idx, "take"]
            cap = strata.at[idx, "n"]
            if diff > 0 and cur < cap:
                strata.at[idx, "take"] = cur + 1; diff -= 1
            elif diff < 0 and cur > 0:
                strata.at[idx, "take"] = cur - 1; diff += 1
            i += 1

    rng = np.random.default_rng(seed)
    chosen = []
    keyed = sdf.reset_index()
    for _, row in strata.iterrows():
        m = (keyed["year"]==row["year"]) & (keyed["age_group"]==row["age_group"]) & (keyed["gender_std"]==row["gender_std"])
        idxs = keyed.loc[m, "index"].to_numpy()
        k = int(row["take"])
        if k <= 0:
            continue
        if len(idxs) <= k:
            chosen.append(idxs)
        else:
            chosen.append(rng.choice(idxs, size=k, replace=False))
    if not chosen:
        return np.arange(0)
    return np.unique(np.concatenate(chosen))


def strip_greeting(t: str) -> str:
    # fjerner "hei", "heisann", "hallo" + ev. tegnsetting i starten
    return re.sub(r'^\s*(hei|heisann|hallo|halloi)[\s,!:.-]+', ' ', t, flags=re.IGNORECASE)

def compute_centroids(embs: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    uniq = np.unique(labels[labels >= 0])
    centroids = {}
    for lab in uniq:
        idx = np.where(labels == lab)[0]
        c = embs[idx].mean(axis=0, keepdims=True)
        centroids[int(lab)] = l2norm(c)[0]
    return centroids

def assign_all(embs: np.ndarray, centroids: Dict[int, np.ndarray], min_sim: float):
    if not centroids:
        return np.full(len(embs), -1, dtype=int), np.zeros(len(embs), dtype=np.float32)
    labs = sorted(centroids.keys())
    C = np.vstack([centroids[k] for k in labs])
    E = l2norm(embs.copy())
    sims = E @ C.T
    best_k = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)
    out_labels = np.array([labs[k] for k in best_k], dtype=int)
    out_labels[best_sim < float(min_sim)] = -1
    return out_labels, best_sim.astype(np.float32)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import spdiags

def ctfi_df(texts_per_cluster, ngram_range=(1,2), min_df=3, stop_words=None, top_k=15):
    """
    c-TF-IDF på sparsmatrise. Returnerer DF: ['cluster_id','term','score','rank'].
    Bruker *funksjonsparametrene* (ikke globale) og logger vokabular/stoppord.
    """
    clusters = sorted(texts_per_cluster.keys())
    docs = [" ".join(texts_per_cluster[cid]) for cid in clusters]

    vect = CountVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=0.6,  # filtrer ekstremt vanlige termer på tvers av klynger
        stop_words=stop_words,
        lowercase=True,
        token_pattern=r"(?u)\b[a-zæøå0-9]{3,}\b",  # minst 3 tegn, inkluder æøå
        strip_accents=None,
    )

    X = vect.fit_transform(docs)            # [n_clusters, vocab]
    vocab = vect.get_feature_names_out()
    print(f"[ctfidf] vocab size: {len(vocab)} | min_df={min_df} ngram={ngram_range} stop_words={len(stop_words) if stop_words else 0}")

    # sanity-check: lekkasje av kjente stoppord
    probe = {"hei","veldig","gjøre","litt","mye","får","få","går","ta","gjør","år","vet","føler"}
    leaked = sorted(probe.intersection(set(vocab)))
    if leaked:
        print(f"[ctfidf][WARN] stoppord i vokabular: {leaked[:20]} (viser opptil 20). Sjekk token_pattern/stop_words.")

    # c-TF
    tf = normalize(X, norm="l1", axis=1, copy=False)

    # c-IDF
    df = (X > 0).sum(axis=0).A1
    idf = np.log((len(clusters) + 1) / (1 + df)) + 1.0

    # c-TF-IDF
    idf_diag = spdiags(idf, 0, X.shape[1], X.shape[1])
    scores = tf @ idf_diag

    rows = []
    for i, cid in enumerate(clusters):
        row = scores.getrow(i).toarray().ravel()
        if row.size == 0 or (row <= 0).all():
            continue
        top_idx = np.argsort(-row)[:int(top_k)]
        for rank, j in enumerate(top_idx, start=1):
            rows.append((int(cid), vocab[j], float(row[j]), int(rank)))

    return pd.DataFrame(rows, columns=["cluster_id","term","score","rank"])


def pick_exemplar(embs: np.ndarray, labels: np.ndarray, centroids: Dict[int, np.ndarray], texts: List[str]):
    exemplars = {}
    E = l2norm(embs.copy())
    for cid, c in centroids.items():
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue
        sims = (E[idx] @ c.reshape(-1,1)).ravel()
        j = idx[int(np.argmax(sims))]
        exemplars[cid] = (int(j), texts[j])
    return exemplars

def main():
    outdir = Path(OUTPUT_DIR); outdir.mkdir(parents=True, exist_ok=True)
    print("[load] reading embeddings and index…")
    C_embs = np.load(outdir / C_EMB_FILE)["embs"]
    C_idx  = pd.read_parquet(outdir / C_INDEX_FILE)
    assert len(C_embs) == len(C_idx), "Embeddings and index row counts do not match."
    print(f"[load] C size: {len(C_embs)}  dim: {C_embs.shape[1]}")

    print("[sample] building stratified sample…")
    sample_idx = stratified_sample_index(C_idx, n_target=SAMPLE_SIZE, min_per_stratum=MIN_PER_STRATUM, seed=SAMPLE_RANDOM_STATE)
    if len(sample_idx) == 0:
        raise RuntimeError("Sampling returned 0 items; check your parameters.")
    print(f"[sample] sample size: {len(sample_idx)} / {len(C_idx)}")

    print("[umap] fitting UMAP on sample…")
    um = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=UMAP_N_COMPONENTS,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
        verbose=True
    )
    sample_embs = C_embs[sample_idx]
    sample_umap = um.fit_transform(sample_embs)
    print(f"[umap] UMAP sample shape: {sample_umap.shape}")

    print("[hdbscan] clustering UMAP-reduced sample…")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_epsilon=HDBSCAN_CLUSTER_SELECTION_EPSILON,
        cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
        metric="euclidean",
        prediction_data=True
    )
    sample_labels = clusterer.fit_predict(sample_umap)
    n_clusters = int(len(set(sample_labels[sample_labels >= 0])))
    print(f"[hdbscan] clusters discovered on sample: {n_clusters} (noise: {(sample_labels==-1).sum()} items)")

    print("[centroids] computing centroids in original space…")
    centroids = compute_centroids(C_embs[sample_idx], sample_labels)
    print(f"[centroids] {len(centroids)} centroids computed.")

    print("[assign] assigning all items to nearest centroid…")
    C_labels, C_sim = assign_all(C_embs, centroids, min_sim=ASSIGN_MIN_SIM)
    print(f"[assign] assigned {int((C_labels>=0).sum())} items to clusters; {int((C_labels<0).sum())} labeled as noise (-1).")

    print("[export] writing C_cluster_assignments.parquet…")
    out_assign = C_idx.copy()
    out_assign["cluster_id"] = C_labels
    out_assign["cluster_sim"] = C_sim
    out_assign.to_parquet(outdir / "C_cluster_assignments.parquet", index=False)

    print("[keywords] computing c-TF-IDF keywords…")
    valid = out_assign["cluster_id"] >= 0
    texts_by_cluster = {
    int(cid): [strip_greeting(s) for s in grp["body_norm"].astype(str).tolist()]
    for cid, grp in out_assign.loc[valid].groupby("cluster_id")
    }

    if len(texts_by_cluster) == 0:
        warnings.warn("No valid clusters to compute keywords on. Check HDBSCAN params and ASSIGN_MIN_SIM.")
        kw_df = pd.DataFrame(columns=["cluster_id","term","score","rank"])
    else:
        kw_df = ctfi_df(texts_by_cluster, ngram_range=NGRAM_RANGE, min_df=MIN_DF,
                        stop_words=STOP_WORDS, top_k=TOP_K_TERMS)
    kw_path = outdir / "clusters_keywords.csv"
    kw_df.to_csv(kw_path, index=False, encoding="utf-8")
    print(f"[keywords] wrote: {kw_path.name}")

    print("[catalog] building cluster catalog…")
    sizes = out_assign.loc[valid].groupby("cluster_id").size().rename("size").reset_index()
    exemplars = pick_exemplar(C_embs, C_labels, centroids, C_idx["body_norm"].astype(str).tolist())
    if len(kw_df):
        top_terms = kw_df.sort_values(["cluster_id","rank"]).groupby("cluster_id")["term"].apply(lambda s: ", ".join(s.tolist()[:TOP_K_TERMS])).to_dict()
    else:
        top_terms = {}

    rows = []
    for _, r in sizes.iterrows():
        cid = int(r["cluster_id"]); size = int(r["size"])
        ex_idx, ex_text = exemplars.get(cid, (-1, ""))
        rows.append({"cluster_id": cid, "size": size, "top_terms": top_terms.get(cid, ""), "exemplar_text": ex_text[:400].replace("\n"," ")})
    catalog = pd.DataFrame(rows).sort_values("size", ascending=False)
    cat_path = outdir / "clusters_catalog.csv"
    catalog.to_csv(cat_path, index=False, encoding="utf-8")
    print(f"[catalog] wrote: {cat_path.name}  (clusters: {len(catalog)})")

    params = {
        "SAMPLE_SIZE": SAMPLE_SIZE, "MIN_PER_STRATUM": MIN_PER_STRATUM, "SAMPLE_RANDOM_STATE": SAMPLE_RANDOM_STATE,
        "UMAP": {"n_neighbors": UMAP_N_NEIGHBORS, "min_dist": UMAP_MIN_DIST, "n_components": UMAP_N_COMPONENTS, "metric": UMAP_METRIC, "random_state": UMAP_RANDOM_STATE},
        "HDBSCAN": {"min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE, "min_samples": HDBSCAN_MIN_SAMPLES, "cluster_selection_epsilon": HDBSCAN_CLUSTER_SELECTION_EPSILON, "cluster_selection_method": HDBSCAN_CLUSTER_SELECTION_METHOD},
        "ASSIGN_MIN_SIM": ASSIGN_MIN_SIM,
        "cTFIDF": {"ngram_range": NGRAM_RANGE, "min_df": MIN_DF, "top_k_terms": TOP_K_TERMS, "stop_words": STOP_WORDS},
        "files": {"C_embeddings": str(outdir / C_EMB_FILE), "C_index": str(outdir / C_INDEX_FILE)}
    }
    with open(outdir / "umap_hdbscan_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print("[done] completed bottom-up clustering pipeline.")

if __name__ == "__main__":
    main()
