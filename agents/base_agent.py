import asyncio
import time
import socket
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from Asilo_1.core.pheromone import PheromoneManager, PheromoneConfig
from Asilo_1.core.policies import PolicyManager, CapabilityProfile
from Asilo_1.p2p.transport import P2PNode
from Asilo_1.p2p.messages import Hello, PheromoneMsg, ModelDeltaMsg, StatsMsg
from Asilo_1.p2p.neighbor_manager import NeighborManager
from Asilo_1.fl.trainers.base import LocalTrainer
from Asilo_1.core.monitor import CSVMonitor
from Asilo_1.fl.privacy.sketches import tiny_bitset, jaccard_bits

@dataclass
class AgentConfig:
    agent_id: str
    host: str
    port: int
    model_id: str
    pheromone: PheromoneConfig
    capability: CapabilityProfile
    round_time_s: float

class Agent:
    def __init__(self, cfg: AgentConfig, trainer: LocalTrainer, peers: Dict[str, Tuple[str, int]]):
        self.cfg = cfg
        self.trainer = trainer
        self.node = P2PNode(cfg.host, cfg.port)
        self.phero = PheromoneManager(cfg.pheromone)
        self.policy = PolicyManager({"_": cfg.capability}, "_")
        self.neighbors = NeighborManager(peers, k_default=cfg.capability.k_peers)
        self.t_round = 0
        self.bytes_sent = 0

        # stable identifiers for logging (NEW)
        self.id = getattr(self.cfg, "agent_id", "agent")
        self.name = self.id

        # routes
        self.node.route('PheromoneMsg', self._on_pheromone)
        self.node.route('ModelDeltaMsg', self._on_model_delta)
        self.node.route('Hello', self._on_hello)

        self.monitor = CSVMonitor(cfg.agent_id)
        self.bitset_last = 0  # last local sketch
        self.collective_alert_q = 2  # quorum for demo
        self.alert_threshold = 0.6   # sketch overlap threshold

        self._t0 = time.time()
        self._last_progress = time.time()
        self._hb_interval = 5.0     # seconds between heartbeats
        self._stall_seconds = 30.0  # warn if no progress for this long
        asyncio.create_task(self._heartbeat())

    async def _on_pheromone(self, msg: PheromoneMsg):
        print(f"[{self.cfg.agent_id}] ← received {type(msg).__name__} from {getattr(msg,'agent_id','?')}", flush=True)
        self.neighbors.update_pheromone(msg.agent_id, msg.p)

    async def _on_model_delta(self, msg: ModelDeltaMsg):
        print(f"[{self.cfg.agent_id}] ← received {type(msg).__name__} from {getattr(msg,'agent_id','?')}", flush=True)
        if msg.model_id == self.cfg.model_id:
            self.trainer.apply_delta(msg.payload, msg.strategy)

    async def _on_hello(self, msg: Hello):
        # Passive; neighbor_manager already seeded at bootstrap
        print(f"[{self.cfg.agent_id}] ← received {type(msg).__name__} from {getattr(msg,'agent_id','?')}", flush=True)
        pass

    async def start(self):
        await self.node.start()

    async def run_forever(self):
        await self.start()
        prev_metrics = self.trainer.eval()
        # keep a copy for safe progress printing (NEW)
        self._last_metrics = dict(prev_metrics) if isinstance(prev_metrics, dict) else {}

        while True:
            t0 = time.time()
            # 1) local step
            self.trainer.fit_local(self.policy.profile.local_batches)
            curr_metrics = self.trainer.eval()

            # 2) update pheromone from utility
            u = self.trainer.compute_utility(prev_metrics, curr_metrics)
            self.bitset_last = tiny_bitset([curr_metrics.get("f1_val", 0.0), u])
            p = self.phero.update_with_utility(u)

            # mark progress (NEW)
            self._last_progress = time.time()
            self._last_metrics = dict(curr_metrics) if isinstance(curr_metrics, dict) else {}

            # print every 20 rounds (robust) (NEW)
            if (getattr(self, "t_round", getattr(self, "t", 0))) % 20 == 0:
                r = getattr(self, "t_round", getattr(self, "t", 0))
                pval = getattr(self.phero if hasattr(self, "phero") else self, "p", None)
                # prefer current-round f1 if present
                lm = self._last_metrics if isinstance(self._last_metrics, dict) else {}
                f1 = lm.get("f1", lm.get("f1_val"))
                print(f"[{self.cfg.agent_id}] r={r} u={u} p={pval} f1={f1} bytes={getattr(self,'bytes_sent',0)}", flush=True)

            # 3) broadcast pheromone
            await self._broadcast_pheromone(p)

            # 4) maybe share model delta under byte budget
            if self.phero.should_share():
                payload, size = self.trainer.make_delta(self.policy.profile.delta)
                if size <= self.policy.profile.max_bytes_round:
                    await self._broadcast_delta(payload, size)
                    self.bytes_sent += size

            # 5) report stats (optional: could gossip too)
            # self.monitor.log(self.t_round, self.bytes_sent, u, p, curr_metrics.get("f1_val", 0.0))

            # 6) round book-keeping
            prev_metrics = curr_metrics
            self.t_round += 1

            # 7) sleep until next round
            dt = self.cfg.round_time_s - (time.time() - t0)
            if dt > 0:
                await asyncio.sleep(dt)

    async def _broadcast_pheromone(self, p: float):
        for aid, peer in self.neighbors.topk(self.policy.profile.k_peers):
            host, port = peer.host, peer.port
            # If your PheromoneMsg supports sketches, include them; otherwise remove sketch_bits=
            msg = PheromoneMsg(
                agent_id=self.cfg.agent_id,
                t=self.t_round,
                p=p,
                # sketch_bits=getattr(self, "bitset_last", None)  # ← uncomment only if field exists
            )
            print(f"[{self.cfg.agent_id}] ⇄ pheromone p={p:.3f} → {aid}@{host}:{port}", flush=True)
            try:
                await self.node.send(host, port, msg)
            except Exception as e:
                print(f"[{self.cfg.agent_id}] ! pheromone send failed to {aid}@{host}:{port}: {e}", flush=True)
                pass

    async def _broadcast_delta(self, payload: Dict[str, Any], size: int):
        for aid, peer in self.neighbors.topk(self.policy.profile.k_peers):
            host, port = peer.host, peer.port
            msg = ModelDeltaMsg(
                agent_id=self.cfg.agent_id,
                t=self.t_round,
                model_id=self.cfg.model_id,
                strategy=self.policy.profile.delta,
                payload=payload,
                bytes_size=size
            )
            print(f"[{self.cfg.agent_id}] → sending delta(kind={self.policy.profile.delta}) to {aid}@{host}:{port} bytes={size}", flush=True)
            try:
                await self.node.send(host, port, msg)
            except Exception as e:
                print(f"[{self.cfg.agent_id}] ! send failed to {aid}@{host}:{port}: {e}", flush=True)
                pass

    async def _heartbeat(self):
        while True:
            await asyncio.sleep(self._hb_interval)
            # Safely read metrics even if attributes differ
            r = getattr(self, "t_round", getattr(self, "t", 0))
            pman = getattr(self, "phero", None)
            pval = getattr(pman, "p", None) if pman is not None else None
            bytes_sent = getattr(self, "bytes_sent", 0)
            # Top-k peers if neighbor manager exists
            peers_ids = []
            nm = getattr(self, "neighbors", None)
            if nm and hasattr(nm, "topk"):
                try:
                    peers_ids = [aid for aid, _ in nm.topk(self.policy.profile.k_peers)]
                except Exception:
                    peers_ids = []
            print(f"[{self.cfg.agent_id}] ♥ heartbeat r={r} p={pval} bytes={bytes_sent} peers={peers_ids}", flush=True)
            # stall detector
            if (time.time() - getattr(self, "_last_progress", 0)) > self._stall_seconds:
                print(f"[{self.cfg.agent_id}] ⚠ possible stall: no progress for > {self._stall_seconds:.0f}s", flush=True)
