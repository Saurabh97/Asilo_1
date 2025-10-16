"""
Head (classifier) artifact pack/apply for both scikit-learn and PyTorch models.

- sklearn: packs coef_ and intercept_
- torch: finds the final nn.Linear head and packs weight (+ bias if present)
- includes alpha for receiver-side blending: new = (1-alpha)*old + alpha*recv
"""

from typing import Dict, Any, Tuple, Optional
import io
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


# ============== sklearn helpers ==============
def _unwrap_sklearn_estimator(model):
    for attr in ("steps", "named_steps"):
        if hasattr(model, attr):
            steps = getattr(model, attr)
            if isinstance(steps, dict):
                candidates = list(steps.values())
            elif isinstance(steps, list):
                candidates = [s[-1] if isinstance(s, tuple) else s for s in steps]
            else:
                candidates = []
            for est in reversed(candidates):
                if hasattr(est, "coef_") and hasattr(est, "intercept_"):
                    return est
    return model


# ============== torch helpers ==============
_DEFAULT_HEAD_PATHS = [
    "fc",
    "classifier",
    "classifier.6",
    "classifier.3",
    "head",
    "linear",
    "model.head",
    "model.classifier",
]

def set_torch_head_paths(paths):
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
    if nn is None:
        return None
    # Try known attribute paths first
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
    # Fallback: last Linear anywhere in model
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
    mdl = _unwrap_sklearn_estimator(model)
    coef = getattr(mdl, "coef_", None)
    inter = getattr(mdl, "intercept_", None)

    # --- sklearn path ---
    if coef is not None and inter is not None:
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            _kind=np.array(["sk"], dtype="<U10"),
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
                _kind=np.array(["torch"], dtype="<U10"),
                alpha=np.array([alpha], dtype=np.float32),
                weight=w,
            )
        else:
            np.savez_compressed(
                buf,
                _kind=np.array(["torch"], dtype="<U10"),
                alpha=np.array([alpha], dtype=np.float32),
                weight=w,
                bias=b,
            )
        data = buf.getvalue()
        return {"kind": "head", "npz": data}, len(data)

    except Exception as e:
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
    # safer extraction: handle old 'kind' or new '_kind'
    kind = str(arr["_kind"][0]) if "_kind" in arr else str(arr["kind"][0])
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
            return  # skip incompatible shapes

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
