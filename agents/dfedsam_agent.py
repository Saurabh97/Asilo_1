# Asilo_1/agents/dfedsam_agent.py
import os, asyncio, time, numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Set, Optional

from Asilo_1.fl.trainers.base import LocalTrainer
from Asilo_1.core.monitor import CSVMonitor
from Asilo_1.p2p.transport import P2PNode
from Asilo_1.p2p.messages import ModelDeltaMsg, Join, Welcome

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


# ===========================
# Orchestrator (embedded here)
# ===========================
class DFedSAMOrchestrator:
    """
    Collect per-round client heads → aggregate → broadcast global head.
    Robust DFedSAM weights:
      - w_i ∝ [(1 - f1_i) + eps_smooth] * num_samples_i
      - if validation is degenerate for a client → ignore metric part (use 1 * n_i)
      - if metrics are degenerate/NaN overall → uniform weights
    """
    def __init__(self, agent_id: str, host: str, port: int, model_id: str,
                 round_time_s: float = 0.1, eps: float = 1e-8):
        self.id = agent_id
        self.model_id = model_id
        self.node = P2PNode(host, port)
        self.round_time_s = float(round_time_s)
        self.eps = float(eps)

        # state
        self.clients: Set[str] = set()
        self.client_ep: Dict[str, Tuple[str, int]] = {}
        # Store triplets per round: {"vec": np.ndarray, "n": int, "deg": bool}
        self._buf: Dict[int, List[Dict[str, object]]] = {}
        self._f1: Dict[int, List[float]] = {}

        # routes
        self.node.route("Join", self._on_join)
        self.node.route("ModelDeltaMsg", self._on_model_delta)
        # add near: self.eps = float(eps)
        exp_name = os.environ.get("EXP_NAME", "dfedsam")
        self.monitor = CSVMonitor(self.id, exp_name)  # new

    async def start(self):
        await self.node.start()
        print(f"[{self.id}] DFedSAM-Orchestrator @ {self.node.host}:{self.node.port}", flush=True)

    async def shutdown(self):
        try:
            await self.node.stop()
        except Exception:
            pass

    # ---------- events ----------
    async def _on_join(self, msg: Join):
        self.clients.add(msg.agent_id)
        self.client_ep[msg.agent_id] = (msg.host, msg.port)
        # Symmetric welcome (optional)
        await self.node.send(msg.host, msg.port, Welcome(peers=[(self.id, self.node.host, self.node.port, "orchestrator")]))
        print(f"[{self.id}] JOIN {msg.agent_id} @ {msg.host}:{msg.port} (clients={sorted(self.clients)})", flush=True)

    async def _on_model_delta(self, msg: ModelDeltaMsg):
        if msg.strategy != "dfedsam-local":
            return
        r = int(getattr(msg, "t", -1))
        w = msg.payload.get("weights")
        if w is None:
            return

        arr = np.asarray(w, dtype=np.float32)
        f1 = float(msg.payload.get("f1_val", 0.0))
        n = int(msg.payload.get("num_samples", 0))
        deg = bool(msg.payload.get("degenerate_val", False))
        # NEW: remember sender id + bytes for decision logging later
        sender = msg.agent_id
        bytes_sz = int(getattr(msg, "bytes_size", 0))
        # Accept client hints and auto-derive degeneracy if necessary
        if not deg:
            val_size = int(msg.payload.get("val_size", 0))
            if val_size > 0 and abs(f1 - 1.0) < 1e-9:
                deg = True

        self._buf.setdefault(r, []).append({"vec": arr, "n": n, "deg": deg, "from": sender, "bytes": bytes_sz})
        self._f1.setdefault(r, []).append(f1)
        # NEW: edge log (client → orchestrator recv)
        try:
            self.monitor.log_edge(t=r, src=sender, dst=self.id, bytes_sz=bytes_sz,
                                kind="model_delta", action="recv")
        except Exception:
            pass
        print(
            f"[{self.id}] RECV r={r} from {msg.agent_id} size={arr.size} "
            f"norm={np.linalg.norm(arr):.6f} f1={f1:.4f} n={n} deg={deg} "
            f"count={len(self._buf[r])}",
            flush=True
        )


    # ---------- round API ----------
    async def run_round(self, round_idx: int, min_participation: float = 0.8, extra_wait_s: float = 0.2):
        """
        Wait for a quorum of the *snapshot* cohort, then aggregate and broadcast.
        """
        # snapshot cohort for this round
        cohort = sorted(self.clients)
        expected = len(cohort)
        quorum = max(1, int(round(expected * float(min_participation))))

        # wait loop
        waited = 0.0
        budget = float(self.round_time_s) + float(extra_wait_s)
        while waited < budget:
            have = len(self._buf.get(round_idx, []))
            if have >= quorum:
                break
            await asyncio.sleep(0.02)
            waited += 0.02

        triplets = self._buf.pop(round_idx, [])
        f1s = self._f1.pop(round_idx, [])
        if not triplets:
            print(f"[{self.id}] WARN r={round_idx} no updates (expected≈{expected}, quorum={quorum})", flush=True)
            return

        stack = np.stack([t["vec"] for t in triplets], axis=0)  # (K,D)
        f1 = np.asarray(f1s, dtype=np.float32)
        n_arr = np.asarray([int(t["n"]) for t in triplets], dtype=np.float32)
        deg = np.asarray([bool(t["deg"]) for t in triplets])

        # ---- robust weights ----
        # smoothing so f1=1.0 doesn't zero-out; ignore metric if degenerate
        eps_smooth = max(self.eps, 5e-2)
        f1c = np.clip(f1, 0.0, 1.0)
        quality = (1.0 - f1c) + eps_smooth
        quality = np.where(deg, np.ones_like(quality), quality)

        n_safe = np.where(np.isfinite(n_arr) & (n_arr > 0), n_arr, 1.0)
        w = quality * n_safe

        if (not np.all(np.isfinite(w))) or (w.sum() <= 0):
            w = np.ones(len(triplets), dtype=np.float32) / float(len(triplets))
            mode = "uniform"
        else:
            w = w / w.sum()
            mode = "dfedsam_smooth*n"

        agg = np.tensordot(w, stack, axes=(0, 0))
        print(f"[{self.id}] AGG r={round_idx} k={len(triplets)} mode={mode} agg_norm={float(np.linalg.norm(agg)):.6f}", flush=True)
        # NEW: log per-update aggregation decisions
        for i, t in enumerate(triplets):
            try:
                self.monitor.log_agg_decision(
                    t=round_idx,
                    agent=self.id,
                    from_id=t.get("from", "?"),
                    bytes_sz=int(t.get("bytes", 0)),
                    decision="keep",        # change to "drop" if you add trimming
                    cos_thresh=0.0,         # fill in if you add cosine gating later
                    mode=mode,              # e.g., "dfedsam_smooth*n" or "uniform"
                    trim_p=0.0,             # % trimmed if you add robust trimming
                    rolled_back=0           # 1 if you implement rollback on this delta
                )
            except Exception:
                pass
        # broadcast
        payload = {"weights": agg.tolist()}
        bmsg = ModelDeltaMsg(agent_id=self.id, t=round_idx, model_id=self.model_id,
                             strategy="dfedsam-agg", payload=payload, bytes_size=int(agg.nbytes))
        for cid in cohort:
            h, p = self.client_ep.get(cid, (None, None))
            if h is None:
                continue
            await self.node.send(h, p, bmsg)
            print(f"[{self.id}] BROADCAST dfedsam-agg → {cid}@{h}:{p} size={bmsg.bytes_size}", flush=True)
            # NEW: edge log (orchestrator → client send)
            try:
                self.monitor.log_edge(t=round_idx, src=self.id, dst=cid, bytes_sz=int(bmsg.bytes_size),
                                    kind="agg", action="send")
            except Exception:
                pass


# ======================
# Client (in same module)
# ======================
@dataclass
class DFedSAMClientConfig:
    agent_id: str
    host: str
    port: int
    model_id: str
    round_time_s: float = 0.1
    max_rounds: int = 50
    client_barrier_timeout_s: float = 5.0  # wait for agg(r)
    sam_rho: float = 0.05                  # SAM radius
    sam_enabled: bool = True               # toggle SAM


class DFedSAMClientAgent:
    """
    Per-round client (FedAvg-like structure):
      1) local train with SAM (if available) → eval (get f1_val)
      2) send head (weights+bias) + f1_val + num_samples + degenerate_val to orchestrator (strategy='dfedsam-local')
      3) WAIT for orchestrator broadcast agg(r) (strategy='dfedsam-agg')
      4) apply global head; finish round
    """
    def __init__(self, cfg: DFedSAMClientConfig, trainer: LocalTrainer, orchestrator: Tuple[str, Tuple[str, int]]):
        self.cfg = cfg
        self.trainer = trainer
        self.node = P2PNode(cfg.host, cfg.port)
        self.id = cfg.agent_id
        self.model_id = cfg.model_id
        self.bytes_sent = 0
        exp_name = os.environ.get("EXP_NAME", "dfedsam_orch")
        self.monitor = CSVMonitor(cfg.agent_id, exp_name)

        # orchestrator info (id, (host, port))
        self._orch_id, (self._orch_host, self._orch_port) = orchestrator

        # per-round barrier: set when agg(r) is applied
        self._agg_events: Dict[int, asyncio.Event] = {}

        # routes
        self.node.route("ModelDeltaMsg", self._on_model_delta)
        self.node.route("Welcome", self._on_welcome)

    # ---------------- lifecycle ----------------
    async def start(self):
        await self.node.start()
        # join hub
        j = Join(agent_id=self.id, host=self.cfg.host, port=self.cfg.port, capability="DFedSAMClient")
        await self.node.send(self._orch_host, self._orch_port, j)

    async def shutdown(self):
        try:
            await self.node.stop()
        except Exception:
            pass

    # ---------------- one round ----------------
    async def run_round(self, round_idx: int):
        # local train (SAM if possible), then eval
        await self._fit_local_sam_or_fallback(epochs=5, rho=self.cfg.sam_rho)
        metrics = self.trainer.eval()
        f1 = float(metrics.get("f1_val", 0.0))

        # ---- derive metadata for robust aggregation ----
        num_samples = self._infer_train_size()
        degenerate_val = self._infer_val_degenerate()

        # send local head + metadata
        head, nbytes = self._get_head_vector()
        if head is not None:
            val_size = 0
            for obj in (self.trainer, getattr(self.trainer, "data", None)):
                if obj is None:
                    continue
                for key in ("val_size", "num_val", "n_val"):
                    v = getattr(obj, key, None)
                    if isinstance(v, (int, np.integer)) and v > 0:
                        val_size = int(v); break
                if val_size:
                    break
                ds = getattr(obj, "val_dataset", None)
                if ds is not None:
                    try:
                        val_size = int(len(getattr(ds, "dataset", ds)))
                    except Exception:
                        pass
                if val_size == 0:
                    dl = getattr(obj, "val_loader", None)
                    if dl is not None and getattr(dl, "dataset", None) is not None:
                        try:
                            val_size = int(len(dl.dataset))
                        except Exception:
                            pass

            payload = {
                "weights": head.tolist(),
                "f1_val": f1,
                "num_samples": int(num_samples),
                "degenerate_val": bool(degenerate_val),
                "val_size": int(val_size),  # (optional hint for server)
            }

            msg = ModelDeltaMsg(agent_id=self.id, t=round_idx, model_id=self.model_id,
                                strategy="dfedsam-local", payload=payload, bytes_size=int(nbytes))
            await self.node.send(self._orch_host, self._orch_port, msg)
            self.bytes_sent += int(nbytes)
            # NEW: edge log (client → orchestrator send)
            try:
                self.monitor.log_edge(t=round_idx, src=self.id, dst=self._orch_id, bytes_sz=int(nbytes),
                                    kind="model_delta", action="send")
            except Exception:
                pass
            print(f"[{self.id}] SENT dfedsam-local → {self._orch_id}@{self._orch_host}:{self._orch_port} "
                  f"size={nbytes}, norm={np.linalg.norm(head):.6f}, f1={f1:.4f}, n={num_samples}, "
                  f"deg_val={degenerate_val}, r={round_idx}", flush=True)

        # wait for aggregated broadcast
        evt = self._agg_events.setdefault(round_idx, asyncio.Event())
        try:
            await asyncio.wait_for(evt.wait(), timeout=self.cfg.client_barrier_timeout_s)
            print(f"[{self.id}] BARRIER OK r={round_idx} (agg applied)", flush=True)
        except asyncio.TimeoutError:
            print(f"[{self.id}] BARRIER TIMEOUT r={round_idx} (agg may arrive late)", flush=True)

        # monitor
        self.monitor.log(round_idx, self.bytes_sent, 0.0, 0.0, f1)

    # -------------- handlers --------------
    async def _on_model_delta(self, msg: ModelDeltaMsg):
        if msg.strategy != "dfedsam-agg":
            return
        w = msg.payload.get("weights")
        if w is None:
            return
        vec = np.asarray(w, dtype=np.float32)
        self._set_head_vector(vec)
        print(f"[{self.id}] APPLIED dfedsam-agg r={msg.t} (size={vec.size}, norm={np.linalg.norm(vec):.6f})", flush=True)
        # NEW: edge log (orchestrator → client recv)
        try:
            # we don’t have msg.bytes_size on recv unless upstream populates it; guard with getattr.
            self.monitor.log_edge(t=int(msg.t), src=msg.agent_id, dst=self.id,
                                bytes_sz=int(getattr(msg, "bytes_size", 0)),
                                kind="agg", action="recv")
        except Exception:
            pass
        # release per-round barrier
        evt = self._agg_events.setdefault(int(msg.t), asyncio.Event())
        if not evt.is_set():
            evt.set()

    async def _on_welcome(self, msg: Welcome):
        return  # not needed

    # -------------- SAM / training --------------
    async def _fit_local_sam_or_fallback(self, epochs: int, rho: float):
        """
        Try, in order:
        1) trainer.fit_local_sam(epochs, rho)  # if your trainer implements it
        2) trainer.fit_local(epochs)           # fallback
        """
        fn = getattr(self.trainer, "fit_local_sam", None)
        if callable(fn) and self.cfg.sam_enabled:
            try:
                return fn(epochs, rho)
            except Exception as e:
                print(f"[{self.id}] WARN SAM hook failed → fallback ({e})", flush=True)
        return self.trainer.fit_local(epochs)

    def _infer_train_size(self) -> int:
        """
        Robustly infer the number of training samples across many trainer styles.
        Returns 0 if unknown.
        """
        t = self.trainer

        # 1) common numeric attrs on trainer or trainer.data
        for obj in (t, getattr(t, "data", None)):
            if obj is None:
                continue
            for key in ("train_size", "num_train", "n_train", "N_train", "n_samples_train"):
                v = getattr(obj, key, None)
                if isinstance(v, (int, np.integer)) and v > 0:
                    return int(v)

        # 2) arrays/datasets/dataloaders on trainer or trainer.data
        def _len_if_seq(x):
            try:
                return len(x)
            except Exception:
                return None

        for obj in (t, getattr(t, "data", None)):
            if obj is None:
                continue

            # dataset-like
            for key in ("train_dataset", "train_data", "train_set", "train_ds"):
                ds = getattr(obj, key, None)
                if ds is not None:
                    # torch Dataset (maybe wrapped)
                    base = getattr(ds, "dataset", ds)
                    n = _len_if_seq(base)
                    if isinstance(n, int) and n > 0:
                        return int(n)
                    # tuple/list (X, y)
                    if isinstance(ds, (tuple, list)) and len(ds) >= 1:
                        n = _len_if_seq(ds[0])
                        if isinstance(n, int) and n > 0:
                            return int(n)

            # dataloader-like
            for key in ("train_loader", "train_dataloader", "loader_train", "dl_train"):
                dl = getattr(obj, key, None)
                if dl is not None:
                    ds = getattr(dl, "dataset", None)
                    n = _len_if_seq(ds) if ds is not None else None
                    if isinstance(n, int) and n > 0:
                        return int(n)

        # 3) sklearn-like arrays – **avoid `or` on tensors**
        data_obj = getattr(t, "data", None)
        for key in ("X_train", "x_train", "train_X"):
            X = getattr(t, key, None)
            if X is None and data_obj is not None:
                X = getattr(data_obj, key, None)
            if X is not None:
                try:
                    # Prefer .shape[0], fall back to len(X)
                    if hasattr(X, "shape") and len(getattr(X, "shape")) >= 1:
                        return int(X.shape[0])
                    else:
                        return int(len(X))
                except Exception:
                    pass

        return 0



    def _infer_val_degenerate(self) -> bool:
        """
        Decide if validation is single-class (degenerate). Returns False if unknown.
        """
        t = self.trainer
        data = getattr(t, "data", None)

        # 1) histogram (preferred)
        val_hist = getattr(data, "val_label_hist", None) if data is not None else None
        if isinstance(val_hist, dict) and len(val_hist) > 0:
            nonzero = sum(1 for _c, cnt in val_hist.items() if cnt and cnt > 0)
            return nonzero <= 1

        # 2) collect raw val labels without using `or` (can be tensors)
        def _extract_labels(obj, keys=("y_val","Y_val","val_y","val_labels","targets","labels")):
            if obj is None:
                return None
            for k in keys:
                y = getattr(obj, k, None)
                if y is not None:
                    return y
            return None

        y = _extract_labels(t)
        if y is None and data is not None:
            y = _extract_labels(data)
        if y is None:
            val_ds = getattr(t, "val_dataset", None)
            y = _extract_labels(val_ds)
        if y is None:
            val_loader = getattr(t, "val_loader", None)
            val_loader_ds = getattr(val_loader, "dataset", None) if val_loader is not None else None
            y = _extract_labels(val_loader_ds)

        try:
            import numpy as _np
            if y is not None:
                # torch tensor → numpy (without truthiness checks)
                if hasattr(y, "detach") and hasattr(y, "cpu"):
                    y_np = y.detach().cpu().numpy()
                elif isinstance(y, list):
                    y_np = _np.array(y)
                else:
                    y_np = _np.asarray(y)
                if y_np.size > 0:
                    if y_np.dtype.kind == 'f':
                        y_np = y_np[~_np.isnan(y_np)]
                    uniq = _np.unique(y_np)
                    return uniq.size <= 1
        except Exception:
            pass

        # 3) heuristic: perfect macro-F1 strongly hints single-class eval
        try:
            f1 = float(self.trainer.eval().get("f1_val", 0.0))
            if abs(f1 - 1.0) < 1e-9:
                return True
        except Exception:
            pass

        return False



    # -------------- head helpers --------------
    @staticmethod
    def _find_linear_head_torch(model):
        if nn is None:
            return None
        def _get(root, path: str):
            cur = root
            for part in path.split("."):
                if not hasattr(cur, part): return None
                cur = getattr(cur, part)
            return cur
        for p in ["fc", "classifier", "classifier.6", "classifier.3", "head", "linear",
                  "model.head", "model.classifier"]:
            m = _get(model, p)
            if m is None: continue
            if isinstance(m, nn.Linear): return m
            if isinstance(m, nn.Sequential):
                last = None
                for sub in m.modules():
                    if isinstance(sub, nn.Linear): last = sub
                if last is not None: return last
        last = None
        for m in getattr(model, "modules", lambda: [])():
            if isinstance(m, nn.Linear): last = m
        return last

    def _get_head_vector(self):
        # sklearn linear
        coef = getattr(self.trainer.model, "coef_", None)
        inter = getattr(self.trainer.model, "intercept_", None)
        if coef is not None and inter is not None:
            vec = np.concatenate([np.ravel(coef), np.ravel(np.atleast_1d(inter))]).astype(np.float32)
            return vec, int(vec.nbytes)
        # torch linear head
        if torch is not None and isinstance(getattr(self.trainer, "model", None), nn.Module):
            head = self._find_linear_head_torch(self.trainer.model)
            if head is None or getattr(head, "weight", None) is None:
                return None, 0
            with torch.no_grad():
                w = head.weight.detach().cpu().numpy().astype(np.float32, copy=False)
                b = head.bias.detach().cpu().numpy().astype(np.float32, copy=False) if getattr(head, "bias", None) is not None else None
            vec = w.ravel() if b is None else np.concatenate([w.ravel(), b.ravel()])
            return vec, int(vec.nbytes)
        return None, 0

    def _set_head_vector(self, vec: np.ndarray):
        coef = getattr(self.trainer.model, "coef_", None)
        inter = getattr(self.trainer.model, "intercept_", None)
        if coef is not None and inter is not None:
            csz = coef.size
            self.trainer.model.coef_ = vec[:csz].reshape(coef.shape)
            self.trainer.model.intercept_ = vec[csz:]
            return
        if torch is not None and isinstance(getattr(self.trainer, "model", None), nn.Module):
            head = self._find_linear_head_torch(self.trainer.model)
            if head is None or getattr(head, "weight", None) is None:
                return
            w = head.weight
            csz = int(w.numel())
            w_new = vec[:csz].reshape(w.shape).astype(np.float32, copy=False)
            b_new = None
            if getattr(head, "bias", None) is not None:
                b_new = vec[csz:csz + head.bias.numel()].reshape(head.bias.shape).astype(np.float32, copy=False)
            with torch.no_grad():
                w.copy_(torch.from_numpy(w_new).to(device=w.device, dtype=w.dtype))
                if b_new is not None:
                    head.bias.copy_(torch.from_numpy(b_new).to(device=w.device, dtype=w.dtype))
