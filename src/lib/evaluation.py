"""Protocolo de evaluación común para los Modelos A y B (se evalúan SOBRE EMBEDDINGS).

Todas las funciones esperan embeddings normalizados L2, de modo que el coseno es un producto punto.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    auc,
    confusion_matrix,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
    roc_curve,
    silhouette_score,
)

UNKNOWN = -1


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


# ----------------------------------------------------------------------------- verificación
def make_pairs(labels: np.ndarray, n_pairs: int = 10000, seed: int = 42):
    """Pares balanceados: n_pairs positivos (misma identidad) y n_pairs negativos."""
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    by_id = {c: np.flatnonzero(labels == c) for c in np.unique(labels)}
    ids = [c for c, idx in by_id.items() if len(idx) >= 2]

    pos_a, pos_b = [], []
    for c in rng.choice(ids, size=n_pairs):
        a, b = rng.choice(by_id[c], size=2, replace=False)
        pos_a.append(a)
        pos_b.append(b)

    n = len(labels)
    i, j = rng.randint(0, n, size=n_pairs * 3), rng.randint(0, n, size=n_pairs * 3)
    keep = labels[i] != labels[j]
    neg_a, neg_b = i[keep][:n_pairs], j[keep][:n_pairs]
    assert len(neg_a) == n_pairs, "no se pudieron generar suficientes pares negativos"

    a = np.concatenate([pos_a, neg_a]).astype(int)
    b = np.concatenate([pos_b, neg_b]).astype(int)
    y = np.concatenate([np.ones(n_pairs), np.zeros(n_pairs)]).astype(int)
    return a, b, y


def pair_scores(emb: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(emb[a] * emb[b], axis=1)


def verification_metrics(scores: np.ndarray, y: np.ndarray, far_targets=(1e-2, 1e-3)) -> dict:
    """ROC, AUC, EER (con su umbral) y TAR @ FAR."""
    fpr, tpr, thr = roc_curve(y, scores)
    fnr = 1.0 - tpr
    k = int(np.nanargmin(np.abs(fnr - fpr)))
    return {
        "auc": float(auc(fpr, tpr)),
        "eer": float((fpr[k] + fnr[k]) / 2.0),
        "eer_threshold": float(thr[k]),
        "tar_at_far": {far: float(np.interp(far, fpr, tpr)) for far in far_targets},
        "fpr": fpr, "tpr": tpr, "thresholds": thr,
    }


def rates_at_threshold(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    """FAR (negativos aceptados) y FRR (positivos rechazados) con un umbral dado."""
    far = float(np.mean(scores[y == 0] >= threshold))
    frr = float(np.mean(scores[y == 1] < threshold))
    return {"far": far, "frr": frr, "tar": 1.0 - frr}


def far_frr_curve(scores: np.ndarray, y: np.ndarray, grid: np.ndarray):
    far = np.array([np.mean(scores[y == 0] >= t) for t in grid])
    frr = np.array([np.mean(scores[y == 1] < t) for t in grid])
    return far, frr


# ----------------------------------------------------------------------------- identificación
def split_gallery_probes(labels: np.ndarray, gallery_ids, k_gallery: int, seed: int):
    """Galería = k imágenes por identidad registrada (imita /insert). Probes = el resto de esas
    identidades (conocidas) + TODAS las imágenes de las identidades no registradas (desconocidas)."""
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    gallery_ids = set(gallery_ids)
    g_idx, p_idx, p_true = [], [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if c in gallery_ids:
            idx = rng.permutation(idx)
            g_idx += idx[:k_gallery].tolist()
            p_idx += idx[k_gallery:].tolist()
            p_true += [c] * len(idx[k_gallery:])
        else:
            p_idx += idx.tolist()
            p_true += [UNKNOWN] * len(idx)
    return np.array(g_idx), np.array(p_idx), np.array(p_true)


def identification_eval(emb, labels, gallery_ids, k_gallery, threshold, seed=0, topk=5) -> dict:
    """Identificación open-set: top-1 entre las identidades de la galería (similitud máxima por
    identidad, igual que `FaceService.identify`) y rechazo como `unknown` si score < umbral."""
    labels = np.asarray(labels)
    g_idx, p_idx, p_true = split_gallery_probes(labels, gallery_ids, k_gallery, seed)
    G, g_lab, P = emb[g_idx], labels[g_idx], emb[p_idx]
    sims = P @ G.T
    ids = np.array(sorted(set(g_lab.tolist())))
    best = np.stack([sims[:, g_lab == c].max(axis=1) for c in ids], axis=1)   # (n_probes, n_ids)
    order = np.argsort(-best, axis=1)
    top1 = ids[order[:, 0]]
    top1_score = best[np.arange(len(P)), order[:, 0]]
    pred = np.where(top1_score >= threshold, top1, UNKNOWN)

    known = p_true != UNKNOWN
    k = min(topk, len(ids))
    topk_hit = np.array([p_true[n] in ids[order[n, :k]] for n in np.flatnonzero(known)])
    classes = list(ids) + [UNKNOWN]
    prec, rec, f1, _ = precision_recall_fscore_support(
        p_true, pred, labels=classes, average="macro", zero_division=0)
    return {
        "top1": float(np.mean(top1[known] == p_true[known])),
        f"top{topk}": float(topk_hit.mean()),
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "unknown_rejected": float(np.mean(pred[~known] == UNKNOWN)) if (~known).any() else float("nan"),
        "known_rejected": float(np.mean(pred[known] == UNKNOWN)),
        "known_misidentified": float(np.mean((pred[known] != UNKNOWN) & (pred[known] != p_true[known]))),
        "confusion": confusion_matrix(p_true, pred, labels=classes),
        "classes": classes, "p_idx": p_idx, "p_true": p_true, "pred": pred, "score": top1_score,
    }


# ----------------------------------------------------------------------------- clustering
def clustering_metrics(emb: np.ndarray, labels: np.ndarray, seed: int = 42) -> dict:
    labels = np.asarray(labels)
    km = KMeans(n_clusters=len(np.unique(labels)), n_init=10, random_state=seed).fit(emb)
    return {
        "ari": float(adjusted_rand_score(labels, km.labels_)),
        "nmi": float(normalized_mutual_info_score(labels, km.labels_)),
        "silhouette": float(silhouette_score(emb, labels, metric="cosine")),
    }
