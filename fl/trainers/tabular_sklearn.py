# Asilo_1/fl/trainers/tabular_sklearn.py
import os, numpy as np, pandas as pd, io
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score
from Asilo_1.fl.trainers.base import LocalTrainer
from Asilo_1.fl.artifacts.prototypes import build_prototypes, apply_proto_pull
from Asilo_1.fl.artifacts.head_delta import pack_head, apply_head
from sklearn.exceptions import NotFittedError

def _load_csv(path: str):
    df = pd.read_csv(path)
    X = df[[c for c in df.columns if c not in ("label","ts")]].values.astype(float)
    y = df["label"].values.astype(int)
    return X, y

def make_trainer_for_subject(data_dir: str, subject_id: str, test_size=0.2):
    path = os.path.join(data_dir, f"{subject_id}.csv")
    if not os.path.exists(path):
        # synth fallback
        n, d = 2000, 16
        X = np.random.randn(n, d)
        w = np.random.randn(d)
        logits = X @ w
        p = 1/(1+np.exp(-logits))
        y = (p > 0.7).astype(int)
    else:
        X, y = _load_csv(path)
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=test_size, stratify=y)
    return SkTabularTrainer(Xtr, ytr, Xv, yv)

class SkTabularTrainer(LocalTrainer):
    def __init__(self, Xtr, ytr, Xv, yv):
        self.X_train, self.y_train = Xtr, ytr
        self.X_val, self.y_val = Xv, yv
        self.model = LogisticRegression(max_iter=200)
        self._cursor = 0

    def fit_local(self, n_batches: int) -> None:
        n = len(self.X_train)
        bs = max(32, n // 20)
        for _ in range(n_batches):
            lo = self._cursor; hi = min(n, lo + bs)
            if lo >= hi:
                self._cursor = 0; lo, hi = 0, min(n, bs)
            Xb, yb = self.X_train[lo:hi], self.y_train[lo:hi]
            self.model.fit(Xb, yb)
            self._cursor = hi

    def eval(self) -> Dict[str, float]:
        try:
            probs = self.model.predict_proba(self.X_val)[:, 1]
        except NotFittedError:
            # First round before any fit: neutral metrics (no utility spike)
            return {"f1_val": 0.0, "auprc": 0.0, "psi": 0.0}
        preds = (probs >= 0.5).astype(int)
        f1 = f1_score(self.y_val, preds)
        auprc = average_precision_score(self.y_val, probs)
        return {"f1_val": float(f1), "auprc": float(auprc), "psi": float(min(1.0, auprc*1.2))}

    def compute_utility(self, prev: Dict[str, float], curr: Dict[str, float]) -> float:
        print(f"  prev: {prev}, curr: {curr}")
        return float(max(0.0, curr.get("auprc", 0.0) - prev.get("auprc", 0.0)) * 5.0)

    # NEW: satisfy ABC even if base_agent uses its own _build_payload
    def make_delta(self, strategy: str) -> Tuple[Dict[str, Any], int]:
        if strategy == "proto":
            payload = build_prototypes(self.X_train, self.y_train)
            size = sum(len(v["mean"]) for v in payload["prototypes"].values()) * 4 + 16
            return payload, size
        # fallback to head delta
        return pack_head(self.model)

    def apply_delta(self, payload: Dict[str, Any], strategy: str) -> None:
        kind = payload.get("kind")
        if kind == "proto":
            apply_proto_pull(self.model, payload)
        elif kind == "head":
            apply_head(self.model, payload)