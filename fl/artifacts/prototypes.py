# Asilo_1/fl/artifacts/prototypes.py
from __future__ import annotations
import numpy as np
import io
from typing import Dict, Any, Tuple

# Optional torch support (no hard dependency)
try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


# For tabular demo, treat raw features as embeddings; for NN replace with penultimate-layer features.

def _to_numpy(a):
    """Convert torch.Tensor -> np.ndarray (CPU), else np.asarray."""
    if torch is not None and isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def build_prototypes(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Returns:
      {"kind": "proto", "prototypes": { "<cls>": {"mean": <list float32>, "count": int }, ... }}
    Works with NumPy arrays OR Torch tensors for X and y (signature unchanged).
    """
    X = _to_numpy(X)
    y = _to_numpy(y)

    # Ensure proper shapes
    if y.ndim != 1:
        y = y.reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N,D), got shape={X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"N mismatch: X={X.shape[0]} vs y={y.shape[0]}")

    # Robust dtype/casting
    try:
        classes = np.unique(y)
    except TypeError:
        # In case of mixed/object labels, coerce to int if possible
        classes = np.unique(y.astype(np.int64))

    protos: Dict[str, Any] = {}
    for c in classes:
        c_int = int(c)  # ensure consistent key type
        idx = (y == c)
        if np.any(idx):
            mean_vec = np.mean(X[idx], axis=0).astype(np.float32, copy=False)
            protos[str(c_int)] = {
                "mean": mean_vec.tolist(),          # keep as list (backward compatible)
                "count": int(np.sum(idx)),
            }

    return {"kind": "proto", "prototypes": protos}


def merge_prototypes(local: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two prototype payloads. Keeps the same schema and types.
    """
    if not local:
        return incoming
    out = {"kind": "proto", "prototypes": {}}
    A = local.get("prototypes", {})
    B = incoming.get("prototypes", {})
    keys = set(A.keys()) | set(B.keys())
    for k in keys:
        if k in A and k in B:
            ma = np.array(A[k].get("mean", []), dtype=np.float32)
            mb = np.array(B[k].get("mean", []), dtype=np.float32)
            ca = int(A[k].get("count", 0))
            cb = int(B[k].get("count", 0))
            denom = max(1, ca + cb)
            # If shapes differ, fall back to the larger-count mean
            if ma.shape != mb.shape:
                if ca >= cb:
                    out["prototypes"][k] = {"mean": ma.tolist(), "count": ca}
                else:
                    out["prototypes"][k] = {"mean": mb.tolist(), "count": cb}
                continue
            m = (ma * ca + mb * cb) / denom
            out["prototypes"][k] = {"mean": m.tolist(), "count": int(ca + cb)}
        else:
            out["prototypes"][k] = A.get(k, B.get(k))
    return out


def _find_linear_head_torch(model):
    """Heuristics to find a final nn.Linear head, if available."""
    if nn is None:
        return None

    def _get(root, path: str):
        cur = root
        for part in path.split("."):
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
        return cur

    candidate_paths = [
        "fc", "classifier", "classifier.6", "classifier.3",
        "head", "linear", "model.head", "model.classifier",
        # add your own custom paths here if needed, e.g. "clf.linear"
    ]
    for p in candidate_paths:
        m = _get(model, p)
        if m is None:
            continue
        if isinstance(m, nn.Linear):
            return m
        if isinstance(m, nn.Sequential):
            last_lin = None
            for sub in m.modules():
                if isinstance(sub, nn.Linear):
                    last_lin = sub
            if last_lin is not None:
                return last_lin

    # fallback: last Linear anywhere
    last = None
    for m in getattr(model, "modules", lambda: [])():
        if isinstance(m, nn.Linear):
            last = m
    return last


def apply_proto_pull(model, proto_payload: Dict[str, Any]):
    """
    Nudge model using class prototypes.
    - sklearn LR: tiny fit on class means (kept as before).
    - torch nn.Module: if a linear head exists, blend head weights toward class means (simple heuristic).
    Signature and return type unchanged.
    """
    # Extract class means (as np arrays)
    try:
        P = proto_payload["prototypes"]
        Xs, ys = [], []
        for cls, obj in P.items():
            if "mean" not in obj:
                continue
            Xs.append(np.array(obj["mean"], dtype=np.float32))
            ys.append(int(cls))
        if not Xs:
            return
        Xs = np.stack(Xs)  # [C, D]
        ys = np.array(ys)  # [C]
    except Exception:
        return

    # --- sklearn path (original behavior) ---
    try:
        # If model has .fit, try a tiny fit on class means (as before)
        # This is a no-op for torch models without .fit
        model.fit(Xs, ys)
        return
    except Exception:
        pass

    # --- torch path (safe heuristic) ---
    if torch is None or not isinstance(model, nn.Module):
        return

    head = _find_linear_head_torch(model)
    if head is None or getattr(head, "weight", None) is None:
        return

    # Expect head.weight: [num_classes, D]
    num_classes, feat_dim = head.weight.shape
    C, D = Xs.shape
    if D != feat_dim:
        # Feature dimension mismatch; cannot apply safely
        return
    if C != num_classes:
        # If prototype classes do not cover all classes, skip to avoid misalignment
        return

    # Convert to torch and alpha-blend toward proto means
    with torch.no_grad():
        device = head.weight.device
        dtype = head.weight.dtype
        proto_w = torch.from_numpy(Xs).to(device=device, dtype=dtype)  # [C, D]
        # Optional: normalize rows (comment out if you prefer raw means)
        # proto_w = torch.nn.functional.normalize(proto_w, p=2, dim=1)

        alpha = 0.5  # gentle nudge
        head.weight.copy_((1.0 - alpha) * head.weight + alpha * proto_w)

        # Bias: set to zero (or keep as-is). Adjust if you have class priors.
        if getattr(head, "bias", None) is not None and head.bias.shape[0] == C:
            # leave bias unchanged or move slightly toward zero for stability
            head.bias.copy_((1.0 - alpha) * head.bias + alpha * torch.zeros_like(head.bias))

def pack_prototypes(proto_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """
    Serialize the prototypes dict into a compressed NPZ.
    The receiver can unpack and use however it wants.
    """
    buf = io.BytesIO()
    arrays = {}
    for c, v in proto_dict.get("prototypes", {}).items():
        mean = v.get("mean")
        if mean is None:
            continue
        arrays[f"c{c}"] = np.asarray(mean, dtype=np.float32)

    if not arrays:
        return {"kind": "proto", "empty": True, "reason": "no_classes"}, 0

    np.savez_compressed(buf, kind="proto", **arrays)
    data = buf.getvalue()
    return {"kind": "proto", "npz": data}, len(data)
