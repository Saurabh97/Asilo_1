# Asilo_1/agents/fedavg_agent.py
import asyncio, time, numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass

from Asilo_1.fl.trainers.base import LocalTrainer
from Asilo_1.core.monitor import CSVMonitor
from Asilo_1.p2p.transport import P2PNode
from Asilo_1.p2p.messages import ModelDeltaMsg, Join, Welcome

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
    def __init__(self, cfg: FedAvgConfig, trainer: LocalTrainer, role: str, peers: Dict[str, Tuple[str,int]]):
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

            # local train + eval
            self.trainer.fit_local(5)
            metrics = self.trainer.eval()

            if not self.is_server:
                weights, size = self._get_weights()
                if weights is not None:
                    payload = {"weights": weights.tolist()}
                    msg = ModelDeltaMsg(agent_id=self.id, t=self.round, model_id=self.cfg.model_id,
                                        strategy="fedavg", payload=payload, bytes_size=size)
                    # send to server
                    for aid, (host, port) in self.peers.items():
                        await self.node.send(host, port, msg)
                        self.bytes_sent += size
            else:
                # aggregate if we have new updates
                if self.updates:
                    agg = np.mean(self.updates, axis=0)
                    self._set_weights(agg)
                    self.updates.clear()

            # CSV schema: (round, bytes_sent, utility, pheromone, f1_val)
            self.monitor.log(self.round, self.bytes_sent, 0.0, 0.0, metrics.get("f1_val", 0.0))

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
            # count bytes received as "sent" to keep comms-comparison fair
            self.bytes_sent += int(msg.bytes_size or 0)

    async def _on_join(self, msg: Join):
        # server acknowledges; we keep welcome minimal to match ASILO style
        if self.is_server:
            peers = [(self.id, self.cfg.host, self.cfg.port, "server")]
            await self.node.send(msg.host, msg.port, Welcome(peers=peers))

    async def _on_welcome(self, msg: Welcome):
        # no-op (already know server)
        return

    def _get_weights(self):
        """Flatten sklearn model coef_ + intercept_ into 1D float32 array."""
        coef = getattr(self.trainer.model, "coef_", None)
        inter = getattr(self.trainer.model, "intercept_", None)
        if coef is None or inter is None:
            return None, 0
        vec = np.concatenate([np.ravel(coef), np.ravel(np.atleast_1d(inter))]).astype(np.float32)
        return vec, int(vec.nbytes)

    def _set_weights(self, vec: np.ndarray):
        """Restore coef_ and intercept_ from flat vector shape."""
        coef = getattr(self.trainer.model, "coef_", None)
        inter = getattr(self.trainer.model, "intercept_", None)
        if coef is None or inter is None:
            return
        csz = coef.size
        new_coef = vec[:csz].reshape(coef.shape)
        new_inter = vec[csz:]
        self.trainer.model.coef_ = new_coef
        self.trainer.model.intercept_ = new_inter
