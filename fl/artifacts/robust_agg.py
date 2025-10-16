# Asilo_1/fl/artifacts/robust_agg.py
from __future__ import annotations
import numpy as np
from typing import List

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))

def trimmed_mean(arrays: List[np.ndarray], trim_p: float = 0.1) -> np.ndarray:
    """
    Robust mean that trims extremes along axis 0.
    Safe for small N: if 2*k >= N, falls back to simple mean.
    """
    A = np.stack(arrays)
    n = A.shape[0]
    if n == 1:
        return A[0]
    k = int(trim_p * n)
    if k <= 0 or (2 * k) >= n:
        return A.mean(axis=0)
    A_sorted = np.sort(A, axis=0)
    return A_sorted[k:-k].mean(axis=0)

def geometric_median(arrays: List[np.ndarray], iters: int = 50, eps: float = 1e-6) -> np.ndarray:
    """
    Weiszfeld iteration with small epsilon to avoid division by zero.
    """
    X = np.stack(arrays)
    if X.shape[0] == 1:
        return X[0]
    m = X.mean(axis=0)
    for _ in range(iters):
        d = np.linalg.norm(X - m, axis=1) + eps
        w = 1.0 / d
        m_new = (X * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(m_new - m) < eps:
            break
        m = m_new
    return m

def coord_median(arrays: List[np.ndarray]) -> np.ndarray:
    """Coordinate-wise median."""
    return np.median(np.stack(arrays), axis=0)

def cosine_gate(arrays: List[np.ndarray], thresh: float = 0.30) -> List[np.ndarray]:
    """
    Drop vectors whose cosine to the cohort mean is below 'thresh'.
    If all drop, return original list to avoid empty set.
    """
    A = np.stack(arrays)
    m = A.mean(axis=0)
    keep = [v for v in arrays if cosine_similarity(v, m) >= thresh]
    return keep if keep else arrays

def aggregate_with_gate(arrays: List[np.ndarray],
                        mode: str = "trimmed",
                        trim_p: float = 0.10,
                        cos_thresh: float = 0.30) -> np.ndarray:
    """
    Cosine-gate first, then aggregate.
    Special-case n==3: use coordinate-wise median (or trimmed-one-off).
    """
    arrays = cosine_gate(arrays, thresh=cos_thresh)
    n = len(arrays)
    if n == 1:
        return arrays[0]
    if n == 3:
        # With three updates, coord-wise median is the “trim-one-each-side” analogue.
        return coord_median(arrays)
    if mode == "median":
        return geometric_median(arrays)
    if mode == "trimmed":
        return trimmed_mean(arrays, trim_p=trim_p)
    return np.stack(arrays).mean(axis=0)
