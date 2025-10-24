# Asilo_1/agents/decefl_agent.py
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
class DeceFLConfig:
    agent_id: str
    host: str
    port: int
    model_id: str
    round_time_s: float = 0.1
    max_rounds: int = 200
    is_server: bool = False
    exp_name: str = "decefl"
    agg_mode: str = "weighted"   # "weighted" (mean) or "median"

def coord_median(stack: np.ndarray) -> np.ndarray:
    return np.median(stack, axis=0)

def weighted_mean(stack: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    if w is None:  # simple mean by default
        return np.mean(stack, axis=0)
    w = np.asarray(w, dtype=np.float64)
    w = w / (np.sum(w) + 1e-12)
    return (stack * w[:, None]).sum(axis=0)

class DeceFLAgent:
    """
    Decentralized, round-synchronous DeceFL:
    - Orchestrator calls run_one_round() each round (no internal while-loop).
    - Each round: local fit -> send to K peers -> bounded wait -> aggregate only same-round messages.
    - Inbox bucketed by msg.t to avoid cross-round mixing.
    """
    def __init__(self, cfg: DeceFLConfig, trainer: LocalTrainer, role: str, peers: Dict[str, Tuple[str,int]]):
        self.cfg = cfg
        self.trainer = trainer
        self.node = P2PNode(cfg.host, cfg.port)
        self.id = cfg.agent_id
        self.role = role
        self.is_server = cfg.is_server
        self.monitor = CSVMonitor(cfg.agent_id, exp_name=cfg.exp_name)

        self.peers: Dict[str, Tuple[str,int]] = peers
        self.round = 0
        self.bytes_sent = 0

        # round_id -> List[np.ndarray]
        self.inbox: Dict[int, List[np.ndarray]] = {}

        self.node.route("ModelDeltaMsg", self._on_model_delta)
        self.node.route("Join", self._on_join)
        self.node.route("Welcome", self._on_welcome)

    async def start(self):
        await self.node.start()
        # lightweight handshake
        j = Join(agent_id=self.id, host=self.cfg.host, port=self.cfg.port, capability="DeceFL")
        for _, (h, p) in self.peers.items():
            await self.node.send(h, p, j)

    async def shutdown(self):
        stop = getattr(self.node, "stop", None)
        if callable(stop): await stop()

    async def run_forever(self):
        """Deprecated: kept for compatibility; no loop here."""
        await self.start()

    # --- inside class DeceFLAgent ---

    def _desired_min_incoming(self) -> int:
        # OLD: return max(1, min(len(self.peers), (len(self.peers)+1)//2))
        # NEW: aim to use ALL available neighbors this round
        return max(1, len(self.peers))

    async def run_one_round(self, local_epochs: int = 5, wait_timeout_s: float = 1.0, grace_s: float = 0.15):
        t0 = time.time()

        self.trainer.fit_local(local_epochs)
        metrics = self.trainer.eval()

        weights, size = self._get_weights()
        if weights is not None and self.peers:
            payload = {"weights": weights.tolist()}
            msg = ModelDeltaMsg(agent_id=self.id, t=self.round, model_id=self.cfg.model_id,
                                strategy="decefl", payload=payload, bytes_size=int(size))
            for _, (h, p) in self.peers.items():
                await self.node.send(h, p, msg)
                self.bytes_sent += int(size)
            print(f"[{self.id}] SENT r={self.round} -> K={len(self.peers)} size={size} norm={np.linalg.norm(weights):.6f}", flush=True)

        # Wait for ALL peers or until timeout
        target = self._desired_min_incoming()
        await self._await_round_updates(self.round, min_count=target, timeout_s=wait_timeout_s)

        # Grace window to catch stragglers
        if grace_s > 0:
            await asyncio.sleep(grace_s)

        # Aggregate what we have for THIS round
        buf = self.inbox.pop(self.round, [])
        got = len(buf)
        k = len(self.peers)
        late_key = f"_late_drop_r{self.round}"
        late = len(self.inbox.get(self.round, []))  # anything that slipped in during merge prints
        if got == 0:
            print(f"[{self.id}] WARN: no peer updates for r={self.round}", flush=True)
        else:
            stack = np.stack(buf, axis=0)
            agg = coord_median(stack) if self.cfg.agg_mode.lower() == "median" else weighted_mean(stack)
            before_vec, _ = self._get_weights()
            self._set_weights(agg)
            after_vec, _ = self._get_weights()
            norms = [float(np.linalg.norm(u)) for u in buf]
            print(f"[{self.id}] AGG r={self.round} n={got}/{k} norms={norms} mode={self.cfg.agg_mode}", flush=True)
            if late:
                # make it explicit that we are dropping stale messages for this round
                print(f"[{self.id}] NOTE: dropping {late} late arrivals for r={self.round}", flush=True)
                self.inbox.pop(self.round, None)

        self.monitor.log(self.round, self.bytes_sent, 0.0, 0.0, metrics.get("f1_val", 0.0))
        self.round += 1

        dt = self.cfg.round_time_s - (time.time() - t0)
        if dt > 0: await asyncio.sleep(dt)

    async def _await_round_updates(self, round_id: int, min_count: int, timeout_s: float):
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline:
            if len(self.inbox.get(round_id, [])) >= min_count:
                break
            await asyncio.sleep(0.01)


    # ---------- handlers ---------- #
    async def _on_model_delta(self, msg: ModelDeltaMsg):
        r = int(getattr(msg, "t", -1))
        w = msg.payload.get("weights", None)
        if w is None or r < 0:
            return
        arr = np.asarray(w, dtype=np.float32)
        self.inbox.setdefault(r, []).append(arr)
        print(f"[{self.id}] RECEIVED from {msg.agent_id} r={r} size={arr.size} norm={np.linalg.norm(arr):.6f} (buf_r={len(self.inbox[r])})", flush=True)

    async def _on_join(self, msg: Join):
        try:
            await self.node.send(msg.host, msg.port, Welcome(peers=[(self.id, self.cfg.host, self.cfg.port, self.role)]))
        except Exception:
            pass

    async def _on_welcome(self, msg: Welcome):
        return

    # ---------- weight helpers ---------- #
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
        for p in ["fc","classifier","classifier.6","classifier.3","head","linear","model.head","model.classifier"]:
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

    def _get_weights(self):
        # sklearn
        coef = getattr(self.trainer.model, "coef_", None)
        inter = getattr(self.trainer.model, "intercept_", None)
        if coef is not None and inter is not None:
            vec = np.concatenate([np.ravel(coef), np.ravel(np.atleast_1d(inter))]).astype(np.float32)
            return vec, int(vec.nbytes)

        # torch
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
        # sklearn
        coef = getattr(self.trainer.model, "coef_", None)
        inter = getattr(self.trainer.model, "intercept_", None)
        if coef is not None and inter is not None:
            csz = coef.size
            self.trainer.model.coef_ = vec[:csz].reshape(coef.shape)
            self.trainer.model.intercept_ = vec[csz:]
            return

        # torch
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
