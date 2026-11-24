"""Duplicate detection and recurring-issue clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize

from .config import RANDOM_SEED


def similarity_vectorizer(texts: pd.Series) -> TfidfVectorizer:
    """TF-IDF tuned for "is this the same problem?" rather than "how severe is it?".

    Terms found in more than 5% of tickets (impact phrases such as "users are
    affected", greetings, service names) are dropped so that matches are driven
    by symptom words like "certificate", "vpn" or "outofmemoryerror".
    """
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.05, max_features=5000, sublinear_tf=True,
                          stop_words="english", token_pattern=r"(?u)\b[a-z][a-z0-9_]+\b")
    return vec.fit(texts)


def find_duplicate_candidates(df: pd.DataFrame, vectorizer: TfidfVectorizer, window_hours: float = 6.0) -> pd.DataFrame:
    """For each ticket, find the most similar earlier ticket on the same service within the window."""
    X = normalize(vectorizer.transform(df["text"]))
    opened = df["opened_at"].to_numpy()
    services = df["service"].to_numpy()
    ids = df["incident_id"].to_numpy()
    order = np.argsort(opened)
    best_sim = np.zeros(len(df))
    best_id = np.array([None] * len(df), dtype=object)
    window = np.timedelta64(int(window_hours * 3600), "s")
    start = 0
    for pos, i in enumerate(order):
        while opened[order[start]] < opened[i] - window:
            start += 1
        cand = [j for j in order[start:pos] if services[j] == services[i]]
        if not cand:
            continue
        sims = (X[cand] @ X[i].T).toarray().ravel()
        k = int(sims.argmax())
        best_sim[i], best_id[i] = sims[k], ids[cand[k]]
    return pd.DataFrame({"incident_id": ids, "match_id": best_id, "similarity": best_sim})


def score_duplicates(cands: pd.DataFrame, truth: pd.DataFrame, thresholds=None) -> list[dict]:
    t = truth.set_index("incident_id")["duplicate_of"]
    is_dup = cands["incident_id"].map(t).notna().to_numpy()
    true_parent = cands["incident_id"].map(t).to_numpy()
    rows = []
    for th in thresholds if thresholds is not None else np.round(np.arange(0.10, 0.91, 0.05), 2):
        flag = cands["similarity"].to_numpy() >= th
        tp = int((flag & is_dup).sum())
        exact = int((flag & is_dup & (cands["match_id"].to_numpy() == true_parent)).sum())
        fp, fn = int((flag & ~is_dup).sum()), int((~flag & is_dup).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"threshold": float(th), "precision": round(p, 4), "recall": round(r, 4),
                     "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0, "flagged": int(flag.sum()),
                     "linked_to_correct_parent": round(exact / tp, 4) if tp else 0.0})
    return rows


def cluster_incidents(df: pd.DataFrame, k_range=range(12, 33, 2), sample: int = 4000) -> dict:
    """Cluster ticket text to surface recurring root causes."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=8000, sublinear_tf=True, stop_words="english")
    X = vec.fit_transform(df["text"])
    svd = TruncatedSVD(n_components=80, random_state=RANDOM_SEED)
    Z = normalize(svd.fit_transform(X))
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(df), size=min(sample, len(df)), replace=False)
    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=4, random_state=RANDOM_SEED).fit(Z)
        scores.append({"k": k, "silhouette": round(float(silhouette_score(Z[idx], km.labels_[idx])), 4)})
    best_k = max(scores, key=lambda s: s["silhouette"])["k"]
    km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_SEED).fit(Z)
    labels = km.labels_
    # Top terms per cluster, read from cluster means in TF-IDF space.
    terms = np.array(vec.get_feature_names_out())
    top_terms = {}
    for c in range(best_k):
        centroid = np.asarray(X[labels == c].mean(axis=0)).ravel()
        top_terms[c] = [t for t in terms[centroid.argsort()[::-1]] if t not in {"num", "changeid"}][:6]
    return {"labels": labels, "k_search": scores, "best_k": best_k, "top_terms": top_terms}


def cluster_quality(labels, root_causes) -> dict:
    return {
        "adjusted_rand_index": round(float(adjusted_rand_score(root_causes, labels)), 4),
        "normalized_mutual_info": round(float(normalized_mutual_info_score(root_causes, labels)), 4),
    }


def repeat_rate(df: pd.DataFrame, days: int = 7) -> pd.Series:
    """Share of incidents in each cluster that repeat on the same service within `days`."""
    d = df.sort_values("opened_at")
    prev = d.groupby(["cluster", "service"])["opened_at"].shift()
    rep = (d["opened_at"] - prev) <= pd.Timedelta(days=days)
    return rep.groupby(d["cluster"]).mean()
