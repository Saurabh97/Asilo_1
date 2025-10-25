"""
Aligned LSTM Trainer for WESAD Wrist Dataset — Streamed / Round‑wise Training
Author: Saurabh Singh (ASILO Project) 
Version: Chronological Split (Train/Val) + persistent streaming iterator per FL round

Key additions in this patch:
- Persistent round-wise training via a streaming DataLoader iterator (no full-epoch loop).
- steps_per_round + batch_size control exact local compute budget per round.
- Iterator seamlessly reinitializes on exhaustion; optional reshuffle at each wrap.
- Optional replay buffer growth via ingest_stream() to append new windows online.
- Gradient clipping + eval-safe guards preserved.
- Backwards-compatible API: fit_local(n_batches) still works (wraps streamed fit).
"""

from __future__ import annotations
import os
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, average_precision_score, log_loss

# --- ASILO imports (unchanged) ---
from Asilo_1.fl.trainers.base import LocalTrainer
from Asilo_1.fl.artifacts.prototypes import build_prototypes, apply_proto_pull
from Asilo_1.fl.artifacts.head_delta import pack_head


# ---------------------- #
# 🧠 1. Data Loader
# ---------------------- #
def _load_split_csvs(data_dir: str, subject_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load time-wise pre-split CSVs for a given subject (train/val).
    Each CSV already contains MinMax-scaled wrist features and label [1–4].
    Returns (X_train, y_train, X_val, y_val).
    """
    sub_dir = os.path.join(data_dir, subject_id)
    paths = {
        "train": os.path.join(sub_dir, f"{subject_id}_train.csv"),
        "val": os.path.join(sub_dir, f"{subject_id}_val.csv"),
    }

    if not all(os.path.exists(p) for p in paths.values()):
        raise FileNotFoundError(f"Missing CSVs for {subject_id} in {sub_dir}")

    # Load splits
    train_df = pd.read_csv(paths["train"]) 
    val_df = pd.read_csv(paths["val"]) 

    label_col = "label"
    feature_cols = [c for c in train_df.columns if c != label_col]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[label_col].values.astype(int)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[label_col].values.astype(int)

    # Map labels [1–4] -> [0–3]
    y_train -= 1
    y_val   -= 1

    # -------- windowing helpers --------
    def make_sequences(X, y, window_len=10, stride=10, label_mode="majority"):
        """
        X: [N, D], y: [N]
        Returns: X_seq [S, T, D], y_seq [S]
        """
        N = len(X)
        if N < window_len:
            return X.reshape(1, 1, -1), y[:1]  # degenerate but safe

        idxs = list(range(0, N - window_len + 1, stride))
        X_seq, y_seq = [], []
        for start in idxs:
            end = start + window_len
            seg_y = y[start:end]
            if label_mode == "majority":
                vals, counts = np.unique(seg_y, return_counts=True)
                y_seq.append(vals[np.argmax(counts)])
            elif label_mode == "last":
                y_seq.append(seg_y[-1])
            elif label_mode == "center":
                y_seq.append(seg_y[window_len // 2])
            else:  # default to majority
                vals, counts = np.unique(seg_y, return_counts=True)
                y_seq.append(vals[np.argmax(counts)])
            X_seq.append(X[start:end])
        return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=int)

    # TRAIN: non-overlap + majority (original semantics)
    Xtr, ytr = make_sequences(X_train, y_train, window_len=10, stride=10, label_mode="majority")

    # VAL: robust to single-class collapse
    Xv, yv = make_sequences(X_val, y_val, window_len=10, stride=1, label_mode="majority")
    if len(np.unique(yv)) < 2:
        Xv, yv = make_sequences(X_val, y_val, window_len=10, stride=1, label_mode="last")
    if len(np.unique(yv)) < 2:
        Xv, yv = make_sequences(X_val, y_val, window_len=10, stride=5, label_mode="center")

    uniq_tr, cnt_tr = np.unique(ytr, return_counts=True)
    uniq_v,  cnt_v  = np.unique(yv,  return_counts=True)
    print(f"[{subject_id}] train labels: {list(zip(uniq_tr.tolist(), cnt_tr.tolist()))}")
    print(f"[{subject_id}]   val labels: {list(zip(uniq_v.tolist(),  cnt_v.tolist()))}")
    if len(uniq_v) < 2:
        print(f"[{subject_id}] WARNING: validation still single-class after remediation. "
              f"Metrics (macro-F1/AUPRC) may be uninformative on this subject.", flush=True)

    return Xtr, ytr, Xv, yv


# ---------------------- #
# 🧠 2. Trainer Factory
# ---------------------- #

def make_trainer_for_subject(data_dir: str, subject_id: str):
    """Factory function to create a streamed LSTM trainer per subject."""
    try:
        Xtr, ytr, Xv, yv = _load_split_csvs(data_dir, subject_id)
        print(f"✅ Loaded subject {subject_id}: train={len(ytr)}, val={len(yv)}")
    except Exception as e:
        print(f"⚠️  Failed to load {subject_id}: {e}")
        n, t, d = 500, 10, 6
        Xtr = np.random.randn(n, t, d).astype(np.float32)
        ytr = np.random.randint(0, 4, n).astype(int)
        Xv = np.random.randn(n // 5, t, d).astype(np.float32)
        yv = np.random.randint(0, 4, n // 5).astype(int)

    return StreamedLSTMTrainer(Xtr, ytr, Xv, yv)


# ---------------------- #
# 🧠 3. LSTM Model
# ---------------------- #

class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


# ---------------------- #
# 🧠 4. Streamed Trainer
# ---------------------- #

class _RoundStream:
    """Persistent iterator over a DataLoader that yields a fixed number of steps per round.
    Reinitializes (optionally reshuffling) on StopIteration.
    """
    def __init__(self, dataset: TensorDataset, batch_size: int = 32, shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._rebuild_loader()

    def _rebuild_loader(self):
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last)
        self.iterator = iter(self.loader)

    def next_batch(self):
        try:
            return next(self.iterator)
        except StopIteration:
            # epoch wrapped → rebuild (reshuffle if enabled) and continue
            self._rebuild_loader()
            return next(self.iterator)

    def set_dataset(self, dataset: TensorDataset):
        self.dataset = dataset
        self._rebuild_loader()


class StreamedLSTMTrainer(LocalTrainer):
    def __init__(
        self,
        Xtr: np.ndarray,
        ytr: np.ndarray,
        Xv: np.ndarray,
        yv: np.ndarray,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        grad_clip: Optional[float] = 1.0,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.grad_clip = grad_clip

        # Tensors
        self.X_train = torch.tensor(Xtr, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(ytr, dtype=torch.long).to(self.device)
        self.X_val = torch.tensor(Xv, dtype=torch.float32).to(self.device)
        self.y_val = torch.tensor(yv, dtype=torch.long).to(self.device)

        input_dim = self.X_train.shape[2]
        num_classes = 4  # Fixed for WESAD (labels 0–3)

        self.model = LSTMModel(input_dim, hidden_dim=hidden_dim, num_classes=num_classes).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

        # Persistent round stream
        self._train_stream = _RoundStream(
            TensorDataset(self.X_train, self.y_train), batch_size=32, shuffle=True, drop_last=False
        )

    # ---------- Stream controls ---------- #
    def set_stream_params(self, batch_size: int = 32, shuffle: bool = True, drop_last: bool = False) -> None:
        """Change loader parameters used by the persistent stream (takes effect immediately)."""
        self._train_stream = _RoundStream(
            TensorDataset(self.X_train, self.y_train), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )

    def ingest_stream(self, X_new: np.ndarray, y_new: np.ndarray, max_buffer: Optional[int] = None) -> None:
        """Append new (windowed) samples into the training buffer for true online learning.
        If max_buffer is set, keep only the most recent max_buffer samples (FIFO).
        """
        x_new_t = torch.tensor(X_new, dtype=torch.float32).to(self.device)
        y_new_t = torch.tensor(y_new, dtype=torch.long).to(self.device)

        self.X_train = torch.cat([self.X_train, x_new_t], dim=0)
        self.y_train = torch.cat([self.y_train, y_new_t], dim=0)

        if max_buffer is not None and self.X_train.shape[0] > max_buffer:
            keep = self.X_train.shape[0] - max_buffer
            self.X_train = self.X_train[-keep:]
            self.y_train = self.y_train[-keep:]

        # Refresh stream dataset
        self._train_stream.set_dataset(TensorDataset(self.X_train, self.y_train))

    # ---------------------- #
    # Local Training (streamed)
    # ---------------------- #
    def fit_local_streamed(self, steps_per_round: int = 10) -> None:
        """Perform exactly `steps_per_round` optimizer steps using the persistent iterator.
        This is ideal for FL where each round has a fixed local compute budget.
        """
        self.model.train()
        for _ in range(max(0, int(steps_per_round))):
            xb, yb = self._train_stream.next_batch()
            xb, yb = xb.to(self.device), yb.to(self.device)

            self.opt.zero_grad()
            logits = self.model(xb)
            loss = self.loss_fn(logits, yb)
            loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()

    # Backwards-compatible wrapper (keeps older orchestration intact)
    def fit_local(self, n_batches: int) -> None:
        self.fit_local_streamed(steps_per_round=n_batches)

    # ---------------------- #
    # Evaluation
    # ---------------------- #
    def eval(self) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X_val)
            preds = logits.argmax(1)
            probs = torch.softmax(logits, dim=1)

        y_true = self.y_val.detach().cpu().numpy()
        y_pred = preds.detach().cpu().numpy()
        y_prob = probs.detach().cpu().numpy()

        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        try:
            auprc = average_precision_score(y_true, y_prob, average="macro")
        except Exception:
            auprc = f1
        try:
            loss = log_loss(y_true, y_prob, labels=[0, 1, 2, 3])
        except Exception:
            loss = 1.0 - f1

        psi = float(min(1.0, 0.6 * f1 + 0.4 * auprc))
        return {"f1_val": float(f1), "auprc": float(auprc), "loss_val": float(loss), "psi": float(psi)}

    # ---------------------- #
    # Utility computation
    # ---------------------- #
    def compute_utility(self, prev: Dict[str, float], curr: Dict[str, float]) -> float:
        f1_prev, f1_curr = prev.get("f1_val", 0.0), curr.get("f1_val", 0.0)
        loss_prev, loss_curr = prev.get("loss_val", 1.0), curr.get("loss_val", 1.0)
        auprc_prev, auprc_curr = prev.get("auprc", 0.0), curr.get("auprc", 0.0)

        delta_f1 = f1_curr - f1_prev
        delta_loss = loss_prev - loss_curr
        delta_auprc = auprc_curr - auprc_prev

        # normalized + weighted (non-negative)
        utility = 0.6 * delta_f1 + 0.3 * delta_auprc + 0.1 * delta_loss
        utility = max(0.0, utility)
        return float(min(1.0, 5.0 * utility))

    # ---------------------- #
    # Delta packing & applying
    # ---------------------- #
    def make_delta(self, strategy: str) -> Tuple[Dict[str, Any], int]:
        if strategy == "proto":
            payload = build_prototypes(self.X_train.detach().cpu().numpy(), self.y_train.detach().cpu().numpy())
            size = sum(len(v["mean"]) for v in payload["prototypes"].values()) * 4 + 16
            return payload, size
        # default: head only (small, communication-friendly)
        return pack_head(self.model.state_dict())

    def apply_delta(self, payload: Dict[str, Any], strategy: str) -> None:
        kind = payload.get("kind")
        if kind == "proto":
            apply_proto_pull(self.model, payload)
        elif kind == "head":
            state_dict = payload.get("weights", None)
            if state_dict is not None:
                self.model.load_state_dict(state_dict)
