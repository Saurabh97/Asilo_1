"""
Aligned LSTM Trainer for WESAD Wrist Dataset
Author: Saurabh Singh (ASILO Project)
Version: Chronological Split (Train/Val/Test)
"""

import os, numpy as np, pandas as pd
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, average_precision_score, log_loss

from Asilo_1.fl.trainers.base import LocalTrainer
from Asilo_1.fl.artifacts.prototypes import build_prototypes, apply_proto_pull
from Asilo_1.fl.artifacts.head_delta import pack_head


# ---------------------- #
# 🧠 1. Data Loader
# ---------------------- #
def _load_split_csvs(data_dir: str, subject_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load time-wise pre-split CSVs for a given subject (train/val/test).
    Each CSV already contains MinMax-scaled wrist features and label [1–4].
    Returns (X_train, y_train, X_val, y_val).
    """
    sub_dir = os.path.join(data_dir, subject_id)
    paths = {
        "train": os.path.join(sub_dir, f"{subject_id}_train.csv"),
        "val": os.path.join(sub_dir, f"{subject_id}_val.csv"),
        #"test": os.path.join(sub_dir, f"{subject_id}_test.csv"),
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
        Returns:
          X_seq: [S, T, D], y_seq: [S]
        """
        N = len(X)
        if N < window_len:
            return X.reshape(1, 1, -1), y[:1]  # degenerate but safe

        idxs = list(range(0, N - window_len + 1, stride))
        X_seq = []
        y_seq = []
        for start in idxs:
            end = start + window_len
            seg_y = y[start:end]
            if label_mode == "majority":
                # most frequent label in the window
                vals, counts = np.unique(seg_y, return_counts=True)
                y_seq.append(vals[np.argmax(counts)])
            elif label_mode == "last":
                y_seq.append(seg_y[-1])
            elif label_mode == "center":
                y_seq.append(seg_y[window_len // 2])
            else:
                # default to majority
                vals, counts = np.unique(seg_y, return_counts=True)
                y_seq.append(vals[np.argmax(counts)])
            X_seq.append(X[start:end])

        return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=int)

    # --- TRAIN: keep your original semantics (non-overlap + majority) ---
    Xtr, ytr = make_sequences(X_train, y_train, window_len=10, stride=10, label_mode="majority")

    # --- VAL: be more robust to single-class collapse ---
    # First try: overlapping windows + majority
    Xv, yv = make_sequences(X_val, y_val, window_len=10, stride=1, label_mode="majority")

    # If still single-class, try a different label selection
    if len(np.unique(yv)) < 2:
        Xv, yv = make_sequences(X_val, y_val, window_len=10, stride=1, label_mode="last")

    # If STILL single-class, relax stride to 5 with center label
    if len(np.unique(yv)) < 2:
        Xv, yv = make_sequences(X_val, y_val, window_len=10, stride=5, label_mode="center")

    # Final guard: if absolutely unavoidable, at least log it loudly
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
    """
    Factory function to create an LSTM trainer per subject.
    Loads the preprocessed chronological splits (train/val).
    """
    try:
        Xtr, ytr, Xv, yv = _load_split_csvs(data_dir, subject_id)
        print(f"✅ Loaded subject {subject_id}: train={len(ytr)}, val={len(yv)}")
    except Exception as e:
        print(f"⚠️  Failed to load {subject_id}: {e}")
        # fallback random dummy data for testing
        n, t, d = 500, 10, 6
        Xtr = np.random.randn(n, t, d)
        ytr = np.random.randint(0, 4, n)
        Xv = np.random.randn(n // 5, t, d)
        yv = np.random.randint(0, 4, n // 5)

    return SkTabularTrainer(Xtr, ytr, Xv, yv)


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
# 🧠 4. Trainer
# ---------------------- #
class SkTabularTrainer(LocalTrainer):
    def __init__(self, Xtr, ytr, Xv, yv):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X_train = torch.tensor(Xtr, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(ytr, dtype=torch.long).to(self.device)
        self.X_val = torch.tensor(Xv, dtype=torch.float32).to(self.device)
        self.y_val = torch.tensor(yv, dtype=torch.long).to(self.device)

        input_dim = self.X_train.shape[2]
        #num_classes = len(torch.unique(self.y_train))
        num_classes = 4  # Fixed for WESAD (labels 0–3)
        self.model = LSTMModel(input_dim, hidden_dim=64, num_classes=num_classes).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()

    # ---------------------- #
    # Local Training
    # ---------------------- #
    def fit_local(self, n_batches: int) -> None:
        self.model.train()
        loader = DataLoader(
            TensorDataset(self.X_train, self.y_train),
            batch_size=32,
            shuffle=True
        )

        for _ in range(n_batches):
            for xb, yb in loader:
                self.opt.zero_grad()
                preds = self.model(xb)
                loss = self.loss_fn(preds, yb)
                loss.backward()
                self.opt.step()

    # ---------------------- #
    # Evaluation
    # ---------------------- #
    def eval(self) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            # Get model predictions
            logits = self.model(self.X_val)     # shape [N, num_classes]
            preds = logits.argmax(1)            # predicted class labels
            probs = torch.softmax(logits, dim=1)

        # Convert to CPU numpy
        y_true = self.y_val.cpu().numpy()
        y_pred = preds.cpu().numpy()
        y_prob = probs.cpu().numpy()

        # Compute metrics using sklearn
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        try:
            auprc = average_precision_score(y_true, y_prob, average="macro")
        except Exception:
            auprc = f1

        try:
            loss = log_loss(y_true, y_prob, labels=[0,1,2,3])
        except Exception:
            loss = 1.0 - f1

        psi = float(min(1.0, 0.6 * f1 + 0.4 * auprc))

        return {
            "f1_val": float(f1),
            "auprc": float(auprc),
            "loss_val": float(loss),
            "psi": float(psi),
        }

    # ---------------------- #
    # Utility computation
    # ---------------------- #
    def compute_utility(self, prev, curr):
        f1_prev, f1_curr = prev.get("f1_val", 0.0), curr.get("f1_val", 0.0)
        loss_prev, loss_curr = prev.get("loss_val", 1.0), curr.get("loss_val", 1.0)
        auprc_prev, auprc_curr = prev.get("auprc", 0.0), curr.get("auprc", 0.0)

        delta_f1 = f1_curr - f1_prev
        delta_loss = loss_prev - loss_curr
        delta_auprc = auprc_curr - auprc_prev

        # normalized + weighted
        utility = 0.6 * delta_f1 + 0.3 * delta_auprc + 0.1 * delta_loss
        utility = max(0.0, utility)
        return float(min(1.0, 5.0 * utility))


    # ---------------------- #
    # Delta packing & applying
    # ---------------------- #
    def make_delta(self, strategy: str) -> Tuple[Dict[str, Any], int]:
        if strategy == "proto":
            payload = build_prototypes(self.X_train.cpu().numpy(), self.y_train.cpu().numpy())
            size = sum(len(v["mean"]) for v in payload["prototypes"].values()) * 4 + 16
            return payload, size
        return pack_head(self.model.state_dict())

    def apply_delta(self, payload: Dict[str, Any], strategy: str) -> None:
        kind = payload.get("kind")
        if kind == "proto":
            apply_proto_pull(self.model, payload)
        elif kind == "head":
            state_dict = payload.get("weights", None)
            if state_dict is not None:
                self.model.load_state_dict(state_dict)
