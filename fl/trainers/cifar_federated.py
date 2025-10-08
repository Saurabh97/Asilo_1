# Asilo_1/fl/trainers/cifar_federated.py
import os, pickle, random
import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from torchvision import transforms
from sklearn.metrics import f1_score, average_precision_score
from sklearn.preprocessing import label_binarize

from Asilo_1.fl.trainers.base import LocalTrainer
from Asilo_1.fl.artifacts.head_delta import pack_head, apply_head
from Asilo_1.fl.artifacts.prototypes import build_prototypes, apply_proto_pull

# -----------------------------------------------------
# -------------------- DATA HELPERS -------------------
# -----------------------------------------------------

def _load_cifar_batch(batch_path: str):
    """Load a single CIFAR batch file (as tensors)."""
    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    X = np.array(batch['data']).reshape((10000, 3, 32, 32)).astype(np.uint8)
    y = np.array(batch['labels']).astype(np.int64)
    return X, y


def make_trainer_for_subject(data_dir: str, subject_id: str, batch_size=64, test_size=0.2):
    """
    Create CIFAR trainer for subject id S01..S05.
    Each subject uses one of the 5 CIFAR data_batch_i files.
    Adds non-IID skew and capability-aware augmentations.
    """

    # ensure we use an absolute path
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", data_dir)
        data_dir = os.path.abspath(data_dir)

    batch_id = int(''.join(filter(str.isdigit, subject_id)))  # "S03" -> 3
    batch_id = max(1, min(5, batch_id))
    batch_path = os.path.join(data_dir, f"data_batch_{batch_id}")
    if not os.path.exists(batch_path):
        raise FileNotFoundError(f"Missing CIFAR batch file: {batch_path}")

    X, y = _load_cifar_batch(batch_path)

    # ---------------- Transform pipeline ----------------
    # Define augmentations (heterogeneity across agents)
    transform_fast = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    transform_slow = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    # Randomly assign some agents as “fast” with augmentations
    is_fast = random.random() > 0.5
    transform = transform_fast if is_fast else transform_slow

    # Reshape CIFAR-10 data (3, 32, 32) → (32, 32, 3) before converting to PIL
    images = []
    for img in X:
        if img.shape == (3072,):  # raw flat vector
            img = img.reshape(3, 32, 32).transpose(1, 2, 0)
        elif img.shape == (3, 32, 32):  # already channel-first
            img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray(img.astype('uint8'))
        images.append(transform(img))

    X_tensor = torch.stack(images)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)

    # ---------------- Non-IID simulation ----------------
    # 80% of one dominant class to simulate data skew
    dominant_class = random.randint(0, 9)
    indices_dom = [i for i, label in enumerate(y) if label == dominant_class]
    indices_oth = [i for i in range(len(y)) if i not in indices_dom]
    keep_dom = int(0.8 * len(indices_dom))
    keep_oth = int(0.2 * len(indices_oth))
    final_indices = random.sample(indices_dom, keep_dom) + random.sample(indices_oth, keep_oth)
    random.shuffle(final_indices)
    dataset = Subset(dataset, final_indices)

    # ---------------- Split into train/val ----------------
    n_val = int(test_size * len(dataset))
    n_train = len(dataset) - n_val
    trainset, valset = random_split(dataset, [n_train, n_val])

    return CIFARFederatedTrainer(trainset, valset, batch_size=batch_size, is_fast=is_fast, dom_class=dominant_class)


# -----------------------------------------------------
# ------------------ MODEL + TRAINER ------------------
# -----------------------------------------------------

class SimpleCNN(nn.Module):
    """A lightweight CNN for CIFAR-10."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class CIFARFederatedTrainer(LocalTrainer):
    def __init__(self, trainset, valset, batch_size=64, is_fast=False, dom_class=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN().to(self.device)
        self.trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        self.valloader = DataLoader(valset, batch_size=batch_size)
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.is_fast = is_fast
        self.dom_class = dom_class or "Mixed"
        self.bytes_sent = 0

    # ------------- Local training step -------------
    def fit_local(self, n_batches: int):
        self.model.train()
        for b_idx, (x, y) in enumerate(self.trainloader):
            if b_idx >= n_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.opt.step()

    # ------------- Evaluation -------------
    def eval(self) -> dict[str, float]:
        self.model.eval()
        probs_all, ys_all = [], []
        with torch.no_grad():
            for Xb, yb in self.valloader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                logits = self.model(Xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

                # --- ensure 2D ---
                if probs.ndim == 1:
                    probs = probs.reshape(1, -1)

                probs_all.append(probs)
                ys_all.append(yb.cpu().numpy())

        if len(probs_all) == 0:
            return {"f1_val": 0.0, "auprc": 0.0, "psi": 0.0}

        probs_all = np.concatenate(probs_all, axis=0)
        ys_all = np.concatenate(ys_all, axis=0)

        preds = probs_all.argmax(axis=1)
        f1 = f1_score(ys_all, preds, average="macro")

        ys_bin = label_binarize(ys_all, classes=list(range(probs_all.shape[1])))
        auprc = average_precision_score(ys_bin, probs_all, average="macro")

        return {"f1_val": float(f1), "auprc": float(auprc), "psi": float(min(1.0, auprc))}


    # ------------- Utility computation -------------
    def compute_utility(self, prev, curr):
        """Utility = smoothed ΔAUPRC with small weighting."""
        return float(max(0.0, curr["auprc"] - prev.get("auprc", 0.0)) * 5.0)

    # ------------- Federated artifacts -------------
    def make_delta(self, strategy: str):
        if strategy == "proto":
            payload = build_prototypes_from_loader(self.trainloader, self.model)
            size = sum(len(v["mean"]) for v in payload["prototypes"].values()) * 4 + 16
            return payload, size

        # --- head delta ---
        payload = pack_head(self.model)
        size = sum(t.numel() for t in self.model.state_dict().values()) * 4
        return payload, size


    def serialize_head(self):
        """Return serialized model delta bytes (for swarm sharing)."""
        import io, torch
        if not hasattr(self, "prev_weights"):
            self.prev_weights = {k: v.clone() for k, v in self.model.state_dict().items()}
            return b""  # only first round

        delta = {}
        for k, v in self.model.state_dict().items():
            dv = (v - self.prev_weights[k]).cpu()
            if torch.norm(dv) > 1e-6:  # ignore tiny changes
                delta[k] = dv
        self.prev_weights = {k: v.clone() for k, v in self.model.state_dict().items()}

        if len(delta) == 0:
            return b""  # nothing meaningful
        buf = io.BytesIO()
        torch.save(delta, buf)
        payload = buf.getvalue()
        self.bytes_sent = len(payload)
        return payload

    def apply_delta(self, delta_bytes):
        if not delta_bytes:
            return
        import io, torch
        buf = io.BytesIO(delta_bytes)
        delta = torch.load(buf, map_location=self.device)
        sd = self.model.state_dict()
        with torch.no_grad():
            for k, v in delta.items():
                if k in sd:
                    sd[k] += 0.5 * v  # weighted merge
        self.model.load_state_dict(sd)


# -----------------------------------------------------
# ---------------- PROTOTYPE HELPERS ------------------
# -----------------------------------------------------

def build_prototypes_from_loader(loader, model):
    """Compute class prototypes from dataloader embeddings."""
    model.eval()
    device = next(model.parameters()).device
    feats = {i: [] for i in range(10)}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            h = model.features(x).mean(dim=[2, 3])  # [B,128]
            for i, label in enumerate(y):
                feats[int(label.item())].append(h[i].cpu().numpy())
    protos = {cls: np.mean(v, axis=0) for cls, v in feats.items() if len(v) > 0}
    return {"kind": "proto", "prototypes": {k: {"mean": v.tolist()} for k, v in protos.items()}}
