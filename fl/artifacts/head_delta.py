# Asilo_1/fl/artifacts/head_delta.py
import io, numpy as np
from typing import Dict, Any, Tuple

def pack_head(model) -> Tuple[Dict[str, Any], int]:
    # For sklearn LR: pack coef_ and intercept_
    coef = getattr(model, "coef_", None)
    inter = getattr(model, "intercept_", None)
    if coef is None or inter is None:
        return {"kind": "head", "empty": True}, 0
    buf = io.BytesIO(); np.savez_compressed(buf, coef=coef, intercept=inter)
    payload = {"kind": "head", "npz": buf.getvalue()}
    return payload, len(payload["npz"])

def apply_head(model, payload: Dict[str, Any]):
    if payload.get("empty"): return
    arr = np.load(io.BytesIO(payload["npz"]))
    coef_r = arr["coef"]; inter_r = arr["intercept"]
    if hasattr(model, "coef_"):
        model.coef_ = (model.coef_ + coef_r) / 2.0
        model.intercept_ = (model.intercept_ + inter_r) / 2.0
