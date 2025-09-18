# asilo/fl/privacy/sketches.py
import hashlib, math
from typing import Iterable

def tiny_bitset(vec: Iterable[float], bits: int = 64) -> int:
    # Coarse, privacy-light sketch of a feature vector
    h = hashlib.sha1((",".join(f"{v:.3f}" for v in list(vec)[:16])).encode()).hexdigest()
    return int(h, 16) & ((1 << bits) - 1)

def jaccard_bits(a: int, b: int) -> float:
    inter = (a & b).bit_count()
    union = (a | b).bit_count()
    return inter / union if union else 0.0
