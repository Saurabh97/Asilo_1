# Asilo_1/agents/dfedsam_agent.py
import asyncio, time, numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass

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

@dataclass
class DFedSAMConfig:
    agent_id: str
    host: str
    port: int
    model_id: str
    round_time_s: float = 0.1
    max_rounds: int = 200
    is_server: bool = False
    rho: float = 0.05     # SAM radius proxy for weighting
    eps: float = 1e-8

class DFedSAMAgent:
    """
    DFedSAM (server-side weighting proxy):
    - Each client sends (weights, f1_val) after local SAM-like training (we keep your trainer intact).
    - Server computes weights for aggregation using 1 - f1_val (proxy for sharpness): lower f1 -> larger weight
      (this mimics giving more mass to clients with 'sharper' losses).
    NOTE: We do NOT mutate your local optimizer; we only change aggregation weights to preserve core algo constraints.
    """
    def __init__(self, cfg: DFedSAMConfig, trainer: LocalTrainer, role: str, peers: Dict[str, Tuple[str,int]]):
        self.cfg = cfg
        self.trainer = trainer
        self.node = P2PNode(cfg.host, cfg.port)
        self.id = cfg.agent_id
        self.is_server = cfg.is_server or (role == "server")
        self.role = role
        self.monitor = CSVMonitor(cfg.agent_id, exp_name="dfedsam")
        self.round = 0
        self.updates: List[np.ndarray] = []
        self.weights_raw: List[float] = []   # server: per-client scalar for weighting
        self.peers = peers
        self.bytes_sent = 0

        self.node.route("ModelDeltaMsg", self._on_model_delta)
        self.node.route("Join", self._on_join)
        self.node.route("Welcome", self._on_welcome)

    async def start(self):
        await self.node.start()
        if not self.is_server:
            for aid, (host, port) in self.peers.items():
                j = Join(agent_id=self.id, host=self.cfg.host, port=self.cfg.port, capability="DFedSAM")
                await self.node.send(host, port, j)

    async def run_forever(self):
        await self.start()
        while self.round < self.cfg.max_rounds:
            t0 = time.time()

            # local train + eval
            self.trainer.fit_local(5)
            metrics = self.trainer.eval()
            f1 = float(metrics.get("f1_val", 0.0))

            if not self.is_server:
                weights, size = self._get_weights()
                if weights is not None:
                    # send weights + the proxy statistic used by server
                    payload = {"weights": weights.tolist(), "f1_val": f1}
                    msg = ModelDeltaMsg(agent_id=self.id, t=self.round, model_id=self.cfg.model_id,
                                        strategy="dfedsam", payload=payload, bytes_size=size)
                    for aid, (host, port) in self.peers.items():
                        await self.node.send(host, port, msg)
                        self.bytes_sent += size
            else:
                if self.updates:
                    stack = np.stack(self.updates, axis=0)    # (n, D)
                    w_raw = np.asarray(self.weights_raw, dtype=np.float32)  # (n,)
                    # proxy: higher weight for lower f1 (sharper / worse)
                    # guard & normalize
                    w = 1.0 - np.clip(w_raw, 0.0, 1.0)
                    w = np.maximum(w, self.cfg.eps)
                    w = w / np.sum(w)

                    agg = np.tensordot(w, stack, axes=(0, 0))  # weighted average
                    self._set_weights(agg)

                    self.updates.clear()
                    self.weights_raw.clear()

            # log compatible with ASILO plots
            self.monitor.log(self.round, self.bytes_sent, 0.0, 0.0, f1)

            self.round += 1
            dt = self.cfg.round_time_s - (time.time() - t0)
            if dt > 0:
                await asyncio.sleep(dt)

    async def _on_model_delta(self, msg: ModelDeltaMsg):
        if self.is_server:
            w = msg.payload.get("weights", None)
            if w is None:
                return
            arr = np.asarray(w, dtype=np.float32)
            self.updates.append(arr)
            # record client's f1 proxy; default to 0.5 if missing
            self.weights_raw.append(float(msg.payload.get("f1_val", 0.5)))
            self.bytes_sent += int(msg.bytes_size or 0)

    async def _on_join(self, msg: Join):
        if self.is_server:
            peers = [(self.id, self.cfg.host, self.cfg.port, "server")]
            await self.node.send(msg.host, msg.port, Welcome(peers=peers))

    async def _on_welcome(self, msg: Welcome):
        return

    @staticmethod
    def _find_linear_head_torch(model):
        if nn is None:
            return None
        # Try common classifier heads first
        def _get(root, path: str):
            cur = root
            for part in path.split("."):
                if not hasattr(cur, part): return None
                cur = getattr(cur, part)
            return cur
        for p in ["fc","classifier","classifier.6","classifier.3","head","linear","model.head","model.classifier"]:
            m = _get(model, p)
            if m is None: continue
            if isinstance(m, nn.Linear): return m
            if isinstance(m, nn.Sequential):
                last = None
                for sub in m.modules():
                    if isinstance(sub, nn.Linear): last = sub
                if last is not None: return last
        # Fallback: last Linear anywhere
        last = None
        for m in getattr(model, "modules", lambda: [])():
            if isinstance(m, nn.Linear): last = m
        return last

    def _get_weights(self):
        # ---- sklearn path ----
        coef = getattr(self.trainer.model, "coef_", None)
        inter = getattr(self.trainer.model, "intercept_", None)
        if coef is not None and inter is not None:
            vec = np.concatenate([np.ravel(coef), np.ravel(np.atleast_1d(inter))]).astype(np.float32)
            return vec, int(vec.nbytes)

        # ---- torch path ----
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

    def _set_weights(self, vec: np.ndarray):
        # ---- sklearn path ----
        coef = getattr(self.trainer.model, "coef_", None)
        inter = getattr(self.trainer.model, "intercept_", None)
        if coef is not None and inter is not None:
            csz = coef.size
            self.trainer.model.coef_ = vec[:csz].reshape(coef.shape)
            self.trainer.model.intercept_ = vec[csz:]
            return

        # ---- torch path ----
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
