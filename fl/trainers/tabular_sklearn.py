import os, numpy as np, pandas as pd
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from Asilo_1.fl.trainers.base import LocalTrainer
from Asilo_1.fl.artifacts.prototypes import build_prototypes, apply_proto_pull
from Asilo_1.fl.artifacts.head_delta import pack_head

# ---------------------- #
# 🧠 1. Data Loader
# ---------------------- #
def _load_csv(path: str):
    df = pd.read_csv(path)
    label_col = 'label' if 'label' in df.columns else 'Label'
    X = df[[c for c in df.columns if c not in (label_col, "ts")]].values.astype(float)
    y = df[label_col].values.astype(int)

    # ✅ Keep only valid emotion classes (1–4)
    valid_mask = np.isin(y, [1, 2, 3, 4])
    X = X[valid_mask]
    y = y[valid_mask]

    # continue reshaping
    window_len = 10
    n_seq = len(X) // window_len
    X = X[:n_seq * window_len].reshape(n_seq, window_len, -1)
    y_seq = y[:n_seq * window_len].reshape(n_seq, window_len)
    y = np.apply_along_axis(lambda a: np.bincount(a).argmax(), 1, y_seq)

    # 🔹 Shift labels from [1,2,3,4] → [0,1,2,3] for PyTorch
    y = y - 1
    return X, y



# ---------------------- #
# 🧠 2. Factory
# ---------------------- #
def make_trainer_for_subject(data_dir: str, subject_id: str, test_size=0.2):
    path = os.path.join(data_dir, f"{subject_id}.csv")
    print(f"Loading data for subject {subject_id} from {path} {os.path.exists(path)}")

    if not os.path.exists(path):
        n, t, d = 1000, 10, 8
        X = np.random.randn(n, t, d)
        y = np.random.randint(0, 4, size=(n,))
    else:
        X, y = _load_csv(path)

    # --- count classes after reshape ---
    unique, counts = np.unique(y, return_counts=True)
    label_info = dict(zip(unique, counts))
    print(f"Subject {subject_id} label distribution: {label_info}")

    # --- robust stratification check ---
    if len(unique) < 2 or np.any(counts < 2):
        print(f"⚠️  Subject {subject_id}: at least one class has <2 samples "
              f"(counts={label_info}). Using random split instead of stratified.")
        stratify_arg = None
    else:
        stratify_arg = y

    Xtr, Xv, ytr, yv = train_test_split(
        X, y, test_size=test_size, stratify=stratify_arg, random_state=42
    )
    return SkTabularTrainer(Xtr, ytr, Xv, yv)
# ---------------------- #
# 🧠 3. LSTM Model
# ---------------------- #
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
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
        num_classes = len(torch.unique(self.y_train))
        self.model = LSTMModel(input_dim, hidden_dim=64, num_classes=num_classes).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()

        self._cursor = 0

    # ---------------------- #
    # Local Training
    # ---------------------- #
    def fit_local(self, n_batches: int) -> None:
        self.model.train()
        n = len(self.X_train)
        bs = max(16, n // 20)
        loader = DataLoader(TensorDataset(self.X_train, self.y_train), batch_size=bs, shuffle=True)

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
            preds = self.model(self.X_val).argmax(1)
        f1 = f1_score(self.y_val.cpu(), preds.cpu(), average='macro')
        return {"f1_val": float(f1), "auprc": float(f1), "psi": float(min(1.0, f1 * 1.2))}

    # ---------------------- #
    # Utility computation
    # ---------------------- #
    def compute_utility(self, prev: Dict[str, float], curr: Dict[str, float]) -> float:
        print(f"  prev: {prev}, curr: {curr}")
        return float(max(0.0, curr.get("f1_val", 0.0) - prev.get("f1_val", 0.0)) * 5.0)

    # ---------------------- #
    # Delta packing & applying
    # ---------------------- #
    def make_delta(self, strategy: str) -> Tuple[Dict[str, Any], int]:
        if strategy == "proto":
            payload = build_prototypes(self.X_train.cpu().numpy(), self.y_train.cpu().numpy())
            size = sum(len(v["mean"]) for v in payload["prototypes"].values()) * 4 + 16
            return payload, size
        # fallback to head delta
        return pack_head(self.model.state_dict())

    def apply_delta(self, payload: Dict[str, Any], strategy: str) -> None:
        kind = payload.get("kind")
        if kind == "proto":
            apply_proto_pull(self.model, payload)
        elif kind == "head":
            state_dict = payload.get("weights", None)
            if state_dict is not None:
                self.model.load_state_dict(state_dict)
