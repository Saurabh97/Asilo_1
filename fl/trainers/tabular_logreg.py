import numpy as np
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from .base import LocalTrainer

class TabularLogRegTrainer(LocalTrainer):
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, random_state: int = 42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model = LogisticRegression(max_iter=200, random_state=random_state)
        # Start warm with a small fit to initialize coef_
        if X_train.shape[0] > 0:
            self.model.fit(X_train[:min(64, len(X_train))], y_train[:min(64, len(y_train))])

    def fit_local(self, batches: int) -> Dict[str, float]:
        # For v0.1, use repeated partial fits by sampling mini-batches
        n = len(self.X_train)
        if n == 0:
            return {"loss": 0.0}
        idx = np.random.randint(0, n, size=min(256 * batches, n))
        Xb, yb = self.X_train[idx], self.y_train[idx]
        self.model.fit(Xb, yb)
        return {"loss": 0.0}

    def eval(self) -> Dict[str, float]:
        if len(self.X_val) == 0:
            return {"f1_val": 0.0}
        yhat = self.model.predict(self.X_val)
        f1 = f1_score(self.y_val, yhat, zero_division=0)
        return {"f1_val": float(f1)}

    def compute_utility(self, prev_metrics: Dict[str, float], curr_metrics: Dict[str, float]) -> float:
        f_prev = prev_metrics.get("f1_val", 0.0)
        f_curr = curr_metrics.get("f1_val", 0.0)
        delta = max(0.0, f_curr - f_prev)
        return float(min(1.0, delta))

    def make_delta(self, strategy: str) -> Tuple[Dict[str, Any], int]:
        if strategy == 'head':
            coef = self.model.coef_.astype(np.float32)
            intercept = self.model.intercept_.astype(np.float32)
            payload = {"coef": coef.tolist(), "intercept": intercept.tolist()}
            size = coef.nbytes + intercept.nbytes
            return payload, int(size)
        elif strategy == 'proto':
            # In tabular w/o embeddings, approximate prototypes as class means
            X0 = self.X_train[self.y_train == 0]
            X1 = self.X_train[self.y_train == 1]
            m0 = (X0.mean(axis=0) if len(X0) else np.zeros(self.X_train.shape[1])).astype(np.float32)
            m1 = (X1.mean(axis=0) if len(X1) else np.zeros(self.X_train.shape[1])).astype(np.float32)
            payload = {"proto0": m0.tolist(), "proto1": m1.tolist(), "d": int(self.X_train.shape[1])}
            size = m0.nbytes + m1.nbytes
            return payload, int(size)
        else:
            return {}, 0

    def apply_delta(self, payload: Dict[str, Any], strategy: str) -> None:
        if strategy == 'head':
            if "coef" in payload and "intercept" in payload:
                peer_coef = np.array(payload["coef"], dtype=np.float32)
                peer_inter = np.array(payload["intercept"], dtype=np.float32)
                # simple moving average blend (tiny step to avoid divergence)
                if hasattr(self.model, 'coef_'):
                    self.model.coef_ = 0.9 * self.model.coef_ + 0.1 * peer_coef
                    self.model.intercept_ = 0.9 * self.model.intercept_ + 0.1 * peer_inter
        elif strategy == 'proto':
            # Optionally use prototypes to adjust bias (toy example)
            pass
