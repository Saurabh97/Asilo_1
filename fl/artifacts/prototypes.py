# Asilo_1/fl/artifacts/prototypes.py
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple

# For tabular demo, treat raw features as embeddings; for NN replace with penultimate-layer features.

def build_prototypes(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    protos: Dict[str, Any] = {}
    classes = np.unique(y)
    for c in classes:
        idx = (y == c)
        if idx.any():
            protos[str(int(c))] = {
                "mean": np.mean(X[idx], axis=0).astype(np.float32).tolist(),
                "count": int(idx.sum()),
            }
    return {"kind": "proto", "prototypes": protos}


def merge_prototypes(local: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    if not local: return incoming
    out = {"kind": "proto", "prototypes": {}}
    A = local.get("prototypes", {})
    B = incoming.get("prototypes", {})
    keys = set(A.keys()) | set(B.keys())
    for k in keys:
        if k in A and k in B:
            ma, ca = np.array(A[k]["mean"], dtype=np.float32), A[k]["count"]
            mb, cb = np.array(B[k]["mean"], dtype=np.float32), B[k]["count"]
            m = (ma * ca + mb * cb) / max(1, (ca + cb))
            out["prototypes"][k] = {"mean": m.tolist(), "count": int(ca + cb)}
        else:
            out["prototypes"][k] = A.get(k, B.get(k))
    return out


def apply_proto_pull(model, proto_payload: Dict[str, Any]):
    # For sklearn LR we can do a tiny fit on class means as a nudge.
    try:
        P = proto_payload["prototypes"]
        Xs, ys = [], []
        for cls, obj in P.items():
            Xs.append(np.array(obj["mean"], dtype=np.float32))
            ys.append(int(cls))
        Xs = np.stack(Xs); ys = np.array(ys)
        model.fit(Xs, ys)
    except Exception:
        pass
