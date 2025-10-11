# Asilo_1/fl/artifacts/head_delta.py
"""
Head (classifier) artifact pack/apply for both scikit-learn and PyTorch models.

- sklearn: packs coef_ and intercept_
- torch: finds the final nn.Linear head and packs weight (+ bias if present)
- includes alpha for receiver-side blending: new = (1-alpha)*old + alpha*recv

Returns:
  payload: {"kind":"head", "npz": <bytes>} or {"kind":"head","empty":True,"reason":...}
  nbytes:  length of the "npz" buffer (0 if empty)
"""

from typing import Dict, Any, Tuple, Optional
import io
import numpy as np

# --- optional deps (safe import) ---
try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


# ============== sklearn helpers ==============

def _unwrap_sklearn_estimator(model):
    """
    If model is a Pipeline or has nested steps, return the final estimator
    that actually has coef_/intercept_. Otherwise return model unchanged.
    """
    for attr in ("steps", "named_steps"):
        if hasattr(model, attr):
            steps = getattr(model, attr)
            if isinstance(steps, dict):
                candidates = list(steps.values())
            elif isinstance(steps, list):
                # Pipeline([...("clf", Estimator)]) -> last item is tuple(name, est)
                candidates = [s[-1] if isinstance(s, tuple) else s for s in steps]
            else:
                candidates = []

            for est in reversed(candidates):
                if hasattr(est, "coef_") and hasattr(est, "intercept_"):
                    return est
    return model


# ============== torch helpers ==============

_DEFAULT_HEAD_PATHS = [
    # common torchvision-ish locations
    "fc",
    "classifier",
    "classifier.6",
    "classifier.3",
    "head",
    "linear",
    "model.head",
    "model.classifier",
    # add your custom ones here if needed, e.g.:
    # "clf.linear",
]

def set_torch_head_paths(paths):
    """
    Allow caller to override/extend the attribute paths we probe to find the head.
    """
    global _DEFAULT_HEAD_PATHS
    _DEFAULT_HEAD_PATHS = list(paths)


def _get_by_attr_path(root, path: str):
    cur = root
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _find_linear_head_torch(model) -> Optional["nn.Linear"]:
    """
    Heuristics to find the classifier head (nn.Linear) in common Torch models.
    - Try common attribute paths
    - If attribute is a Sequential, use its last Linear
    - Fallback: last Linear anywhere in the model
    """
    if nn is None:
        return None

    # 1) try known attribute paths first
    for p in _DEFAULT_HEAD_PATHS:
        mod = _get_by_attr_path(model, p)
        if mod is None:
            continue
        if isinstance(mod, nn.Linear):
            return mod
        if isinstance(mod, nn.Sequential):
            last_lin = None
            for m in mod.modules():
                if isinstance(m, nn.Linear):
                    last_lin = m
            if last_lin is not None:
                return last_lin

    # 2) fallback: walk entire module tree and return the last Linear
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    return last_linear


# ============== public API ==============

def pack_head(model, *, alpha: float = 0.5) -> Tuple[Dict[str, Any], int]:
    """
    Build a compact 'head' payload from either sklearn (coef_/intercept_) or PyTorch (nn.Linear).
    Returns (payload, nbytes). If nbytes == 0, payload['empty']=True with 'reason'.
    """
    # --- sklearn path ---
    mdl = _unwrap_sklearn_estimator(model)
    coef = getattr(mdl, "coef_", None)
    inter = getattr(mdl, "intercept_", None)
    if coef is not None and inter is not None:
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            kind="sk",
            alpha=np.array([alpha], dtype=np.float32),
            coef=np.asarray(coef),
            intercept=np.asarray(inter),
        )
        data = buf.getvalue()
        return {"kind": "head", "npz": data}, len(data)

    # --- torch path ---
    try:
        if torch is None or nn is None:
            return {"kind": "head", "empty": True, "reason": "no_torch"}, 0

        head = _find_linear_head_torch(model)
        if head is None or not hasattr(head, "weight") or head.weight is None:
            return {"kind": "head", "empty": True, "reason": "no_torch_linear"}, 0

        with torch.no_grad():
            w = head.weight.detach().cpu().numpy()
            b = head.bias.detach().cpu().numpy() if getattr(head, "bias", None) is not None else None

        buf = io.BytesIO()
        if b is None:
            np.savez_compressed(
                buf,
                kind="torch",
                alpha=np.array([alpha], dtype=np.float32),
                weight=w,
            )
        else:
            np.savez_compressed(
                buf,
                kind="torch",
                alpha=np.array([alpha], dtype=np.float32),
                weight=w,
                bias=b,
            )
        data = buf.getvalue()
        return {"kind": "head", "npz": data}, len(data)

    except Exception as e:  # keep the reason handy for logs
        return {"kind": "head", "empty": True, "reason": f"torch_pack_error:{e}"}, 0


def apply_head(model, payload: Dict[str, Any]) -> None:
    """
    Apply a received head payload onto the local model.
    - sklearn: alpha-blend coef_/intercept_
    - torch: alpha-blend nn.Linear weight (+ bias if present)
    """
    if payload.get("empty"):
        return
    raw = payload.get("npz")
    if raw is None:
        return

    arr = np.load(io.BytesIO(raw))
    kind = str(arr["kind"])
    alpha = float(arr["alpha"][0]) if "alpha" in arr else 0.5

    if kind == "sk":
        coef_r = arr["coef"]
        inter_r = arr["intercept"]
        mdl = _unwrap_sklearn_estimator(model)
        if hasattr(mdl, "coef_") and hasattr(mdl, "intercept_"):
            mdl.coef_ = (1.0 - alpha) * mdl.coef_ + alpha * coef_r
            mdl.intercept_ = (1.0 - alpha) * mdl.intercept_ + alpha * inter_r
        return

    if kind == "torch":
        if torch is None or nn is None:
            return
        head = _find_linear_head_torch(model)
        if head is None or head.weight is None:
            return

        w_r = arr["weight"]
        if head.weight.shape != w_r.shape:
            # class count or feature width differs; skip to avoid corruption
            return

        device = head.weight.device
        dtype = head.weight.dtype

        w_r_t = torch.from_numpy(w_r).to(device=device, dtype=dtype)
        with torch.no_grad():
            head.weight.copy_((1.0 - alpha) * head.weight + alpha * w_r_t)

            if "bias" in arr and getattr(head, "bias", None) is not None:
                b_r = arr["bias"]
                if head.bias.shape == b_r.shape:
                    b_r_t = torch.from_numpy(b_r).to(device=device, dtype=dtype)
                    head.bias.copy_((1.0 - alpha) * head.bias + alpha * b_r_t)
        return
