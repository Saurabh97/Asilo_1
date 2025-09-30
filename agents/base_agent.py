# Asilo_1/agents/base_agent.py
from __future__ import annotations
import math
import asyncio, time
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from Asilo_1.core.pheromone import PheromoneConfig, PheromoneTable
from Asilo_1.core.capability import CapabilityProfile
from Asilo_1.core.trigger import TriggerConfig, Trigger
from Asilo_1.p2p.transport import P2PNode
from Asilo_1.p2p.messages import Hello, PheromoneMsg, ModelDeltaMsg, StatsMsg, Join, Welcome, Introduce, Bye
from Asilo_1.p2p.neighbor_manager import NeighborManager
from Asilo_1.fl.trainers.base import LocalTrainer
from Asilo_1.core.monitor import CSVMonitor
from Asilo_1.fl.artifacts.prototypes import build_prototypes, apply_proto_pull
from Asilo_1.fl.artifacts.head_delta import pack_head, apply_head
from Asilo_1.fl.artifacts.robust_agg import cosine_similarity
from Asilo_1.fl.privacy.dp import clip_and_noise

@dataclass
class AgentConfig:
    agent_id: str
    host: str
    port: int
    model_id: str
    pheromone: PheromoneConfig
    capability: CapabilityProfile
    round_time_s: float
    trigger: TriggerConfig = TriggerConfig()
    ttl_seconds: float = 20.0
    max_rounds: int | None = None
    max_seconds: int | None = None

class Agent:
    def __init__(self, cfg: AgentConfig, trainer: LocalTrainer, peers: Dict[str, Tuple[str, int]]):
        self.cfg = cfg
        self.trainer = trainer
        self.node = P2PNode(cfg.host, cfg.port)
        self.phero = PheromoneTable(cfg.pheromone, self_id=cfg.agent_id)
        self.trigger = Trigger(cfg.trigger)
        self.neighbors = NeighborManager(peers, k_default=cfg.capability.k_peers, ttl_seconds=cfg.ttl_seconds)
        self.t_round = 0
        self.bytes_sent = 0
        self.id = cfg.agent_id
        self.monitor = CSVMonitor(cfg.agent_id)
        self._p_local = self.cfg.pheromone.tau0  # running pheromone we broadcast
        self._t_start = time.time()
        self._last_eval = None

        # routes
        self.node.route('PheromoneMsg', self._on_pheromone)
        self.node.route('ModelDeltaMsg', self._on_model_delta)
        self.node.route('Hello', self._on_hello)
        self.node.route('Join', self._on_join)
        self.node.route('Welcome', self._on_welcome)
        self.node.route('Introduce', self._on_introduce)
        self.node.route('Bye', self._on_bye)
        # heartbeat
        asyncio.create_task(self._heartbeat())

    async def start(self):
        await self.node.start()
        # try joining via any configured peer as bootstrap
        known = self.neighbors.all_peers()
        if known:
            _, pi = known[0]
            j = Join(agent_id=self.id, host=self.cfg.host, port=self.cfg.port, capability=self.cfg.capability.name)
            print(f"[{self.id}] → join {pi.host}:{pi.port}", flush=True)
            await self.node.send(pi.host, pi.port, j)
        else:
            print(f"[{self.id}] seed node; no bootstrap", flush=True)

    async def run_forever(self):
        await self.start()
        prev_metrics = self.trainer.eval()
        last_bytes_sent = 0
        self._no_improve = 0
        while True:
            # stop guards
            if self.cfg.max_rounds is not None and self.t_round >= self.cfg.max_rounds:
                print(f"[{self.id}] ✓ reached max_rounds={self.cfg.max_rounds}", flush=True); return
            if self.cfg.max_seconds is not None and (time.time() - self._t_start) >= self.cfg.max_seconds:
                print(f"[{self.id}] ✓ reached max_seconds={self.cfg.max_seconds}", flush=True); return

            t0 = time.time()
            # --- local step ---
            self.trainer.fit_local(self.cfg.capability.local_batches)
            curr_metrics = self.trainer.eval()
            self._last_eval = curr_metrics
            u = self.trainer.compute_utility(prev_metrics, curr_metrics)
            # after computing u
            eps = 1e-3  # or 1e-3
            if abs(u) < eps:
                u = 0.0

            self._no_improve = 0 if u > eps else (self._no_improve + 1)
            if self._no_improve in (50, 100, 200):  # escalate gently
                self.trigger.cooldown_s *= 1.5

            if self._no_improve >= 300:  # ~300 rounds flat
                print(f"[{self.id}] plateau reached; idling", flush=True)
                await asyncio.sleep(5.0)
                continue  

            # update local pheromone from utility (EMA-style)
            alpha = 0.2  # smoothing
            self._p_local = (1 - alpha) * self._p_local + alpha * max(0.0, u)
            self.phero.update_self(self._p_local)   # <— keep broadcast value in the table


            # --- evaporation ---
            self.phero.evaporate()

            # --- decide to speak ---
            psi = curr_metrics.get("psi", u)  # allow richer utility if your trainer computes it
            peers = self.phero.choose_peers(1)
            best_peer_score = self.phero.score_of(peers[0]) if peers else 0.0

            if (self._p_local >= 0.9 * (best_peer_score + 1e-6)) and (u > eps) and self.trigger.should_send(psi):
                payload, size = self._build_payload()
                if size <= self.cfg.capability.max_bytes_round:
                    await self._broadcast_delta(payload, size)
                    self.bytes_sent += size
                    self.trigger.on_send(size)

            # --- always broadcast pheromone (heartbeat) ---
            if (self.t_round % 5) == 0:
                await self._broadcast_pheromone(self._p_local)
            # --- log ---
            self.monitor.log(self.t_round, self.bytes_sent, u, self.phero.get_self(), curr_metrics.get("f1_val", 0.0))

            prev_metrics = curr_metrics
            self.t_round += 1
            # pace rounds
            dt = self.cfg.round_time_s - (time.time() - t0)
            if self.t_round % 20 == 0:
                f1 = curr_metrics.get("f1_val")
                au = curr_metrics.get("auprc")
                print(f"[{self.id}] r={self.t_round} psi={curr_metrics.get('psi'):.3f} "
                    f"u={u:.4f} p_local={self._p_local:.4f} f1={f1:.3f} auprc={au:.3f} bytes={self.bytes_sent}", flush=True)
            if dt > 0: await asyncio.sleep(dt)

    # ===================== messaging =====================
    async def _on_pheromone(self, msg: PheromoneMsg):
        self.neighbors.update_pheromone(msg.agent_id, msg.p)

    async def _on_model_delta(self, msg: ModelDeltaMsg):
        if msg.model_id != self.cfg.model_id:
            return
        # robust merge gates (simple demo: cosine gate on head if present)
        if msg.payload.get("kind") == "head" and "npz" in msg.payload:
            # Accept; apply as averaging inside head_delta.apply_head
            apply_head(self.trainer.model, msg.payload)
        elif msg.payload.get("kind") == "proto":
            apply_proto_pull(self.trainer.model, msg.payload)
        # after applying, estimate ΔU for deposition (privacy-lite)
        pre = (self._last_eval or {"auprc": 0.0})
        post = self.trainer.eval()
        delta_u = max(0.0, post.get("auprc", 0.0) - pre.get("auprc", 0.0))
        self._last_eval = post
        du = clip_and_noise(delta_u, self.cfg.pheromone.u_max, sigma=0.0)
        self.phero.deposit(msg.agent_id, du, max(1, msg.bytes_size), reputation_badness=0.0)

    async def _on_hello(self, msg: Hello):
        pass

    async def _on_join(self, msg: Join):
        self.neighbors.add_or_update(msg.agent_id, msg.host, msg.port, msg.capability)
        peers = [(aid, pi.host, pi.port, pi.capability) for aid, pi in self.neighbors.all_peers()]
        await self.node.send(msg.host, msg.port, Welcome(peers=peers))
        intro = Introduce(agent_id=msg.agent_id, host=msg.host, port=msg.port, capability=msg.capability)
        for aid, pi in self.neighbors.all_peers():
            if aid not in (msg.agent_id, self.id):
                try: await self.node.send(pi.host, pi.port, intro)
                except Exception: pass

    async def _on_welcome(self, msg: Welcome):
        for aid, host, port, cap in msg.peers:
            if aid != self.id:
                self.neighbors.add_or_update(aid, host, port, cap)
        print(f"[{self.id}] ✓ Welcome: {len(msg.peers)} peers known", flush=True)

    async def _on_introduce(self, msg: Introduce):
        if msg.agent_id != self.id:
            self.neighbors.add_or_update(msg.agent_id, msg.host, msg.port, msg.capability)
            print(f"[{self.id}] + discovered {msg.agent_id}", flush=True)

    async def _on_bye(self, msg: Bye):
        self.neighbors.remove(msg.agent_id)

    # ===================== helpers =====================
    async def _heartbeat(self):
        while True:
            await asyncio.sleep(5.0)
            dead = self.neighbors.sweep_dead()
            for aid in dead:
                print(f"[{self.id}] ✖ timeout peer {aid}", flush=True)

    async def _broadcast_pheromone(self, p: float):
        for aid, peer in self.neighbors.topk(self.cfg.capability.k_peers):
            msg = PheromoneMsg(agent_id=self.id, t=self.t_round, p=p)
            print(f"[{self.id}] ⇄ pheromone p={p:.3f} → {aid}@{peer.host}:{peer.port}", flush=True)
            try: await self.node.send(peer.host, peer.port, msg)
            except Exception: pass

    def _build_payload(self) -> Tuple[Dict[str, Any], int]:
        # choose artifact (prefer prototypes then head under budget)
        # trainer must expose X_train/y_train or a feature buffer for prototypes
        try:
            X, y = self.trainer.X_train, self.trainer.y_train
            proto = build_prototypes(X, y)
            size_p = sum(len(v["mean"]) for v in proto["prototypes"].values()) * 4 + 16
        except Exception:
            proto, size_p = None, 0
        if proto and size_p <= self.cfg.capability.max_bytes_round // 2:
            return proto, size_p
        head, size_h = pack_head(self.trainer.model)
        return head, size_h

    async def _broadcast_delta(self, payload: Dict[str, Any], size: int):
        for aid, peer in self.neighbors.topk(self.cfg.capability.k_peers):
            msg = ModelDeltaMsg(agent_id=self.id, t=self.t_round, model_id=self.cfg.model_id,
                                strategy=payload.get("kind", "unknown"), payload=payload, bytes_size=size)
            print(f"[{self.id}] → sending delta(kind={payload.get('kind')} bytes={size}) to {aid}@{peer.host}:{peer.port}", flush=True)
            try: await self.node.send(peer.host, peer.port, msg)
            except Exception: pass

    async def send_bye(self):
        b = Bye(agent_id=self.id)
        for aid, pi in self.neighbors.all_peers():
            try: await self.node.send(pi.host, pi.port, b)
            except Exception: pass
