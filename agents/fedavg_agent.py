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
class FedAvgConfig:
    agent_id: str
    host: str
    port: int
    model_id: str
    round_time_s: float = 0.1
    max_rounds: int = 200
    is_server: bool = False


class FedAvgAgent:
    def __init__(self, cfg: FedAvgConfig, trainer: LocalTrainer, role: str, peers: Dict[str, Tuple[str, int]]):
        self.cfg = cfg
        self.trainer = trainer
        self.node = P2PNode(cfg.host, cfg.port)
        self.id = cfg.agent_id
        self.is_server = cfg.is_server or (role == "server")
        self.role = role
        self.monitor = CSVMonitor(cfg.agent_id, exp_name="fedavg")
        self.round = 0
        self.updates: List[np.ndarray] = []   # server only
        self.peers = peers
        self.bytes_sent = 0

        # routes
        self.node.route("ModelDeltaMsg", self._on_model_delta)
        self.node.route("Join", self._on_join)
        self.node.route("Welcome", self._on_welcome)

    async def start(self):
        await self.node.start()
        if not self.is_server:
            # Join the server
            for aid, (host, port) in self.peers.items():
                j = Join(agent_id=self.id, host=self.cfg.host, port=self.cfg.port, capability="FedAvg")
                await self.node.send(host, port, j)

    async def run_forever(self):
        await self.start()
        while self.round < self.cfg.max_rounds:
            t0 = time.time()

            # ---- Round start diagnostics ----
            if self.is_server:
                vec, _ = self._get_weights()
                if vec is not None:
                    print(f"[{self.id}] START round={self.round} "
                          f"head_norm_before_train={np.linalg.norm(vec):.6f}", flush=True)
                else:
                    print(f"[{self.id}] START round={self.round} no weights yet", flush=True)

            # ---- Local train + eval ----
            self.trainer.fit_local(5)
            metrics = self.trainer.eval()
            f1 = float(metrics.get("f1_val", 0.0))

            # ---- CLIENT side ----
            if not self.is_server:
                weights, size = self._get_weights()
                if weights is not None:
                    payload = {"weights": weights.tolist()}
                    msg = ModelDeltaMsg(agent_id=self.id, t=self.round, model_id=self.cfg.model_id,
                                        strategy="fedavg", payload=payload, bytes_size=size)
                    for aid, (host, port) in self.peers.items():
                        await self.node.send(host, port, msg)
                        self.bytes_sent += size
                        print(f"[{self.id}] SENT update → to {aid}@{host}:{port} "
                              f"size={size} bytes, norm={np.linalg.norm(weights):.6f}, "
                              f"f1={f1:.4f}, round={self.round}", flush=True)

            # ---- SERVER side aggregation ----
            else:
                if self.updates:
                    stack = np.stack(self.updates, axis=0)
                    norms = [float(np.linalg.norm(u)) for u in stack]
                    print(f"[{self.id}] AGGREGATING {len(self.updates)} updates "
                          f"(shape={stack.shape}) stack_norms={norms}", flush=True)

                    agg = np.mean(stack, axis=0)
                    agg_norm = np.linalg.norm(agg)
                    print(f"[{self.id}] AGG mean_norm={agg_norm:.6f}", flush=True)

                    before_vec, _ = self._get_weights()
                    before_norm = np.linalg.norm(before_vec) if before_vec is not None else 0.0

                    self._set_weights(agg)

                    after_vec, _ = self._get_weights()
                    after_norm = np.linalg.norm(after_vec) if after_vec is not None else 0.0
                    delta_norm = np.linalg.norm(after_vec - before_vec) if (before_vec is not None and after_vec is not None) else 0.0

                    print(f"[{self.id}] SERVER MERGE: before_norm={before_norm:.6f}, "
                          f"after_norm={after_norm:.6f}, delta_norm={delta_norm:.6f}, "
                          f"round={self.round}", flush=True)

                    self.updates.clear()

            # ---- Monitoring ----
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
            print(f"[{self.id}] RECEIVED update from {msg.agent_id} "
                  f"(size={arr.size}, norm={np.linalg.norm(arr):.6f}, "
                  f"total_buffered={len(self.updates)})", flush=True)
            # count bytes received as "sent" to keep comms comparison fair
            self.bytes_sent += int(msg.bytes_size or 0)

    async def _on_join(self, msg: Join):
        if self.is_server:
            peers = [(self.id, self.cfg.host, self.cfg.port, "server")]
            await self.node.send(msg.host, msg.port, Welcome(peers=peers))

    async def _on_welcome(self, msg: Welcome):
        return

    # ---- model helpers ----
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
        for p in ["fc","classifier","classifier.6","classifier.3","head","linear",
                  "model.head","model.classifier"]:
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
        coef = getattr(self.trainer.model, "coef_", None)
        inter = getattr(self.trainer.model, "intercept_", None)
        if coef is not None and inter is not None:
            vec = np.concatenate([np.ravel(coef), np.ravel(np.atleast_1d(inter))]).astype(np.float32)
            return vec, int(vec.nbytes)

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
