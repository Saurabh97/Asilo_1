# Asilo_1/fl/artifacts/robust_agg.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple

# Simple robust aggregators for small message sets.

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))

def trimmed_mean(arrays: List[np.ndarray], trim_p: float = 0.1) -> np.ndarray:
    A = np.stack(arrays)
    k = int(trim_p * A.shape[0])
    if k == 0: return A.mean(axis=0)
    A_sorted = np.sort(A, axis=0)
    return A_sorted[k:-k].mean(axis=0)

def geometric_median(arrays: List[np.ndarray], iters: int = 50, eps: float = 1e-6) -> np.ndarray:
    X = np.stack(arrays)
    m = X.mean(axis=0)
    for _ in range(iters):
        d = np.linalg.norm(X - m, axis=1) + eps
        w = 1.0 / d
        m_new = (X * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(m_new - m) < eps:
            break
        m = m_new
    return m
