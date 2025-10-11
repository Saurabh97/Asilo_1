# Asilo_1/agents/base_agent.py
from __future__ import annotations
import io
import numpy as np
import asyncio, time
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from Asilo_1.core.pheromone import PheromoneConfig, PheromoneTable
from Asilo_1.core.capability import CapabilityProfile
from Asilo_1.core.trigger import TriggerConfig, Trigger
from Asilo_1.p2p.transport import P2PNode
from Asilo_1.p2p.neighbor_manager import NeighborManager
from Asilo_1.fl.trainers.base import LocalTrainer
from Asilo_1.core.monitor import CSVMonitor
from Asilo_1.fl.artifacts.prototypes import build_prototypes, apply_proto_pull,pack_prototypes
from Asilo_1.fl.artifacts.head_delta import pack_head
from Asilo_1.p2p.messages import Hello, PheromoneMsg, ModelDeltaMsg, Join, Welcome, Introduce, Bye
from Asilo_1.fl.artifacts.head_delta import pack_head
from Asilo_1.fl.artifacts.robust_agg import cosine_similarity, trimmed_mean, geometric_median
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

    # NEW knobs
    u_eps: float = 1e-3
    speak_gate_factor: float = 0.9
    artifact_order: tuple[str, ...] = ("proto", "head")
    head_every: int = 0  # 0 = disabled

    # robust config already present in your previous patch:
    cos_gate_thresh: float = 0.2
    agg_mode: str = "median"
    trim_p: float = 0.1
    aggregate_every: int = 5
    rollback_tol: float = 0.005

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
        self.monitor = CSVMonitor(cfg.agent_id, exp_name="asilo")
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

        # --- poisoning/robustness state ---
        self._incoming_heads: list[tuple[str, np.ndarray, int]] = []  # (sender_id, vec, bytes)
        self._last_good_state = None     # (coef, intercept)
        self._last_good_eval = None
        self._inbuf_lock = asyncio.Lock()  # NEW: guard shared buffer


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
        self._no_improve = 0
        while True:
            # stop guards
            if self.cfg.max_rounds is not None and self.t_round >= self.cfg.max_rounds:
                print(f"[{self.id}] reached max_rounds={self.cfg.max_rounds}", flush=True); return
            if self.cfg.max_seconds is not None and (time.time() - self._t_start) >= self.cfg.max_seconds:
                print(f"[{self.id}] reached max_seconds={self.cfg.max_seconds}", flush=True); return

            t0 = time.time()
            # --- local step ---
            self.trainer.fit_local(self.cfg.capability.local_batches)
            curr_metrics = self.trainer.eval()
            self._last_eval = curr_metrics
            u = self.trainer.compute_utility(prev_metrics, curr_metrics)
            # after computing u
            eps = self.cfg.u_eps
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
            # --- decide to speak ---
            psi = curr_metrics.get("psi", u)  # allow richer utility if your trainer computes it
            peers = self.phero.choose_peers(1)
            best_peer_score = self.phero.score_of(peers[0]) if peers else 0.0
            factor = self.cfg.speak_gate_factor

            reason = None
            if u <= eps:
                reason = f"utility too small (u={u:.4f} ≤ eps={eps})"
            elif not self.trigger.should_send(psi):
                reason = f"trigger blocked (psi={psi:.3f})"
            elif self._p_local < factor * (best_peer_score + 1e-6):
                reason = f"pheromone gate failed (p_local={self._p_local:.4f}, best_peer={best_peer_score:.4f}, factor={factor})"

            if reason:
                print(f"[{self.id}] not sending this round → {reason}", flush=True)
            else:
                payload, size = self._build_payload()
                total_send = size * self.cfg.capability.k_peers
                if total_send <= self.cfg.capability.max_bytes_round:
                    await self._broadcast_delta(payload, size)
                    self.bytes_sent += total_send
                    self.trigger.on_send(total_send)
                    print(f"[{self.id}] sending update (kind={payload.get('kind')} size={size} "
                        f"to {self.cfg.capability.k_peers} peers)", flush=True)
                else:
                    print(f"[{self.id}]  payload too large (size={size}, "
                        f"limit={self.cfg.capability.max_bytes_round})", flush=True)

            
            # --- aggregate buffered heads every N rounds ---
            if (self.t_round % max(1, self.cfg.aggregate_every) == 0):
                async with self._inbuf_lock:
                    batch = self._incoming_heads[:]
                    self._incoming_heads.clear()

                agg_count = len(batch)
                rolled_back = 0

                if batch:
                    snap = self._snapshot_head()
                    pre_eval = self._last_eval or self.trainer.eval()

                    # stack accepted head vectors
                    V = np.stack([v for _, v, _ in batch])

                    # robust aggregation
                    if self.cfg.agg_mode == "median":
                        agg = geometric_median([V[i] for i in range(V.shape[0])])
                    elif self.cfg.agg_mode == "trimmed":
                        agg = trimmed_mean([V[i] for i in range(V.shape[0])], trim_p=self.cfg.trim_p)
                    else:  # "mean"
                        agg = V.mean(axis=0)

                    # apply aggregate head, then evaluate
                    self._apply_head_vector(agg)
                    post_eval = self.trainer.eval()

                    # rollback if harmful
                    if post_eval.get("auprc", 0.0) + 1e-9 < pre_eval.get("auprc", 0.0) - self.cfg.rollback_tol:
                        self._restore_head(snap)
                        rolled_back = 1
                        # penalize all senders for this batch (bytes still “wasted”)
                        for sid, _, bsz in batch:
                            self.phero.deposit(sid, 0.0, max(1, bsz), reputation_badness=1.0)
                    else:
                        # good update: credit equally per sender (capped, no noise here)
                        delta_u = max(0.0, post_eval.get("auprc", 0.0) - pre_eval.get("auprc", 0.0))
                        share = (delta_u / max(1, len(batch)))
                        for sid, _, bsz in batch:
                            du = clip_and_noise(share, self.cfg.pheromone.u_max, sigma=0.0)
                            self.phero.deposit(sid, du, max(1, bsz), reputation_badness=0.0)


                # clear buffer
                self._incoming_heads.clear()


            # --- always broadcast pheromone (heartbeat) ---
            if (self.t_round % 5) == 0:
                await self._broadcast_pheromone(self._p_local)
            # --- log ---
            self.monitor.log(self.t_round, self.bytes_sent, u, self.phero.get_self(), curr_metrics.get("f1_val", 0.0), agg_count=agg_count, rollback=rolled_back)

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

        reputation_badness = 0.0  # 0 good, 1 bad (used in deposit)

        if msg.payload.get("kind") == "head" and "npz" in msg.payload:
            # decode incoming head to a flat vector
            try:
                arr = np.load(io.BytesIO(msg.payload["npz"]), allow_pickle=False)  # NEW: no pickles
                vec_in = np.concatenate([arr["coef"].ravel(), arr["intercept"].ravel()]).astype(np.float32)
            except Exception:
                reputation_badness = 1.0
                self.phero.deposit(msg.agent_id, 0.0, max(1, msg.bytes_size), reputation_badness=reputation_badness)
                return

            # cosine gate vs our current head
            vec_cur = self._get_head_vector()
            if vec_cur is not None:
                cos = cosine_similarity(vec_cur, vec_in)
                if cos < self.cfg.cos_gate_thresh:
                    reputation_badness = 1.0
                    self.phero.deposit(msg.agent_id, 0.0, max(1, msg.bytes_size), reputation_badness=reputation_badness)
                    return

            # passed gate → buffer for robust aggregation later (LOCKED)
            async with self._inbuf_lock:
                self._incoming_heads.append((msg.agent_id, vec_in, msg.bytes_size))
            return  # defer applying until aggregation tick


        elif msg.payload.get("kind") == "proto":
            # prototypes path: apply directly (usually safe/compact)
            apply_proto_pull(self.trainer.model, msg.payload)
            post = self.trainer.eval()
            pre = (self._last_eval or {"auprc": 0.0})
            delta_u = max(0.0, post.get("auprc", 0.0) - pre.get("auprc", 0.0))
            self._last_eval = post
            du = clip_and_noise(delta_u, self.cfg.pheromone.u_max, sigma=0.0)
            self.phero.deposit(msg.agent_id, du, max(1, msg.bytes_size), reputation_badness=0.0)
            return

        # unknown payload → ignore quietly


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
        print(f"[{self.id}]  Welcome: {len(msg.peers)} peers known", flush=True)

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
                print(f"[{self.id}]  timeout peer {aid}", flush=True)

    async def _broadcast_pheromone(self, p: float):
        for aid, peer in self.neighbors.topk(self.cfg.capability.k_peers):
            msg = PheromoneMsg(agent_id=self.id, t=self.t_round, p=p)
            print(f"[{self.id}] ⇄ pheromone p={p:.3f} → {aid}@{peer.host}:{peer.port}", flush=True)
            try: await self.node.send(peer.host, peer.port, msg)
            except Exception: pass

    def _build_payload(self) -> Tuple[Dict[str, Any], int]:

        budget = getattr(self.cfg.capability, "max_bytes_round", 0) or 0

        order = list(self.cfg.artifact_order)
        if self.cfg.head_every and (self.t_round % self.cfg.head_every == 0):
            order = ["head"] + [o for o in order if o != "head"]

        proto_payload, proto_bytes = None, 0
        head_payload, head_bytes = None, 0

        print(f"[payload] order={order} budget={budget}")

        for kind in order:
            if kind == "head" and head_payload is None:
                alpha = getattr(self.cfg, "head_alpha", 0.5)
                head_payload, head_bytes = pack_head(self.trainer.model, alpha=alpha)
                print(f"[payload] head_bytes={head_bytes} reason={head_payload.get('reason')}")
                if head_bytes > 0 and head_bytes <= budget:
                    return head_payload, head_bytes

            if kind == "proto" and proto_payload is None:
                try:
                    X, y = self.trainer.X_train, self.trainer.y_train
                    proto = build_prototypes(X, y)
                    proto_payload, proto_bytes = pack_prototypes(proto)
                    print(f"[payload] proto_bytes={proto_bytes}")
                except Exception as e:
                    print(f"[payload] proto_built=False err={e!r}")
                    proto_payload, proto_bytes = None, 0
                if proto_bytes > 0 and proto_bytes <= budget:
                    return proto_payload, proto_bytes

        # fallback: prefer any >0-byte artifact that fits
        for payload, nbytes, name in ((head_payload, head_bytes, "head"),
                                    (proto_payload, proto_bytes, "proto")):
            if nbytes > 0 and nbytes <= budget:
                print(f"[payload] fallback→{name} bytes={nbytes}")
                return payload, nbytes

        reason = (head_payload or {}).get("reason") or (proto_payload or {}).get("reason") or "no_artifact_fits"
        print(f"[payload] none reason={reason} budget={budget} head={head_bytes} proto={proto_bytes}")
        return {"kind": "none", "reason": reason, "budget": budget}, 0



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
    
    # ---- head vector helpers ----
    def _get_head_vector(self) -> np.ndarray | None:
        coef = getattr(self.trainer.model, "coef_", None)
        inter = getattr(self.trainer.model, "intercept_", None)
        if coef is None or inter is None:
            return None
        return np.concatenate([coef.ravel(), np.atleast_1d(inter).ravel()]).astype(np.float32)

    def _apply_head_vector_avg(self, vec: np.ndarray):
        """Average current head with vec (same shape) — lightweight safe merge."""
        coef = getattr(self.trainer.model, "coef_", None)
        inter = getattr(self.trainer.model, "intercept_", None)
        if coef is None or inter is None:
            return
        cur = self._get_head_vector()
        if cur is None or cur.shape != vec.shape:
            return
        merged = (cur + vec) / 2.0
        csz = coef.size
        new_coef = merged[:csz].reshape(coef.shape)
        new_inter = merged[csz:]
        self.trainer.model.coef_ = new_coef
        self.trainer.model.intercept_ = new_inter

    def _snapshot_head(self):
        coef = getattr(self.trainer.model, "coef_", None)
        inter = getattr(self.trainer.model, "intercept_", None)
        if coef is None or inter is None:
            return None
        return (coef.copy(), inter.copy())

    def _restore_head(self, snap):
        if not snap: return
        coef, inter = snap
        self.trainer.model.coef_ = coef
        self.trainer.model.intercept_ = inter

    def _apply_head_vector(self, vec: np.ndarray):
        """Commit the aggregated head vector directly (no averaging)."""
        coef = getattr(self.trainer.model, "coef_", None)
        inter = getattr(self.trainer.model, "intercept_", None)
        if coef is None or inter is None:
            return
        total = coef.size + np.atleast_1d(inter).size
        if vec.size != total:
            return
        csz = coef.size
        self.trainer.model.coef_ = vec[:csz].reshape(coef.shape)
        self.trainer.model.intercept_ = vec[csz:]

