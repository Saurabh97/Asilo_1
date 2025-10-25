from __future__ import annotations
import io
import os
import random
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

from Asilo_1.fl.artifacts.prototypes import (
    build_prototypes, apply_proto_pull, pack_prototypes
)
from Asilo_1.fl.artifacts.head_delta import pack_head
from Asilo_1.p2p.messages import (
    Hello, PheromoneMsg, ModelDeltaMsg, Join, Welcome, Introduce, Bye,RoundReady, CommDone
)
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

    # knobs
    u_eps: float = 1e-3
    speak_gate_factor: float = 0.9
    artifact_order: tuple[str, ...] = ("proto", "head")
    head_every: int = 0  # 0 = disabled

    # robust aggregation
    cos_gate_thresh: float = 0.2
    agg_mode: str = "median"      # "median" | "trimmed" | "mean"
    trim_p: float = 0.1
    aggregate_every: int = 5
    rollback_tol: float = 0.005   # absolute tolerance


class Agent:
    def __init__(self, cfg: AgentConfig, trainer: LocalTrainer, peers: Dict[str, Tuple[str, int]],start_gate: asyncio.Event | None = None,
                 min_peers: int | None = None, round_slot_s: float | None = None):
        self.cfg = cfg
        self.trainer = trainer
        self.node = P2PNode(cfg.host, cfg.port)
        self.phero = PheromoneTable(cfg.pheromone, self_id=cfg.agent_id)
        self.trigger = Trigger(cfg.trigger)
        self.neighbors = NeighborManager(peers, k_default=cfg.capability.k_peers, ttl_seconds=cfg.ttl_seconds)
        self.t_round = 0
        self.bytes_sent = 0
        self.id = cfg.agent_id
        exp_name = os.environ.get("EXP_NAME", "asilo")
        self.monitor = CSVMonitor(cfg.agent_id, exp_name)
        self._p_local = self.cfg.pheromone.tau0
        self._t_start = time.time()
        self._last_eval = None
        self._start_gate = start_gate or asyncio.Event()
        if start_gate is None:
            self._start_gate.set()  # default: no waiting if not provided

        # routes
        self.node.route('PheromoneMsg', self._on_pheromone)
        self.node.route('ModelDeltaMsg', self._on_model_delta)
        self.node.route('Hello', self._on_hello)
        self.node.route('Join', self._on_join)
        self.node.route('Welcome', self._on_welcome)
        self.node.route('Introduce', self._on_introduce)
        self.node.route('Bye', self._on_bye)

        # background heartbeat
        asyncio.create_task(self._heartbeat())

        # robustness state
        self._incoming_heads: list[tuple[str, np.ndarray, int]] = []  # (sender_id, vec, bytes)
        self._last_good_state = None
        self._last_good_eval = None
        self._inbuf_lock = asyncio.Lock()
        self._last_merge_vec = None
        self._last_merge_round = -1

        # discovery + sync
        self._min_peers = min_peers if min_peers is not None else max(0, len(peers))  # usually total-1
        self._discovered_evt = asyncio.Event()
        self._round_slot_s = round_slot_s or max(0.05, cfg.round_time_s)
        self._epoch_t0 = None   # set when gate opens
        self.cfg.round_time_s=self._round_slot_s
        self.node.route('RoundReady', self._on_round_ready)   
        self.node.route('CommDone', self._on_comm_done)
        # per-round sync tokens
        self._rr_tokens: set[tuple[int,str]] = set()  # (round, peer_id)
        self._cd_tokens: set[tuple[int,str]] = set()
        self._epoch_set = False

    async def _broadcast_all(self, msg_obj):
        for _, pi in self.neighbors.all_peers():
            try:
                await self.node.send(pi.host, pi.port, msg_obj)
            except Exception:
                pass       
    async def _on_round_ready(self, msg: RoundReady):
        if msg.t == self.t_round:  # only for our current round
            self._rr_tokens.add((msg.t, msg.agent_id))

    async def _on_comm_done(self, msg: CommDone):
        if msg.t == self.t_round:
            self._cd_tokens.add((msg.t, msg.agent_id))
    async def start(self):
        await self.node.start()
        known = self.neighbors.all_peers()
        if known:
            _, pi = known[0]
            j = Join(agent_id=self.id, host=self.cfg.host, port=self.cfg.port, capability=self.cfg.capability.name)
            print(f"[{self.id}] → join {pi.host}:{pi.port}", flush=True)
            await self.node.send(pi.host, pi.port, j)
        else:
            print(f"[{self.id}] seed node; no bootstrap", flush=True)
    
    def _have_quorum(self) -> bool:
        # NeighborManager exposes all_peers(); exclude self
        return len(self.neighbors.all_peers()) >= self._min_peers

    async def wait_discovery(self, timeout_s: float = 30.0):
        # quick poll loop; simple and robust
        end = asyncio.get_event_loop().time() + timeout_s
        while True:
            if self._have_quorum():
                self._discovered_evt.set()
                return True
            if asyncio.get_event_loop().time() >= end:
                # proceed even if not full quorum (but print)
                print(f"[{self.id}] discovery timeout: have={len(self.neighbors.all_peers())} "
                      f"need={self._min_peers} — proceeding.", flush=True)
                return False
            await asyncio.sleep(0.2)
    def _set_epoch_now(self):
        self._epoch_t0 = asyncio.get_event_loop().time()

    async def _wait_for_round_slot(self, r: int):
        # Align each round start to epoch + r*round_slot
        if self._epoch_t0 is None:
            return
        target = self._epoch_t0 + r * self._round_slot_s
        now = asyncio.get_event_loop().time()
        if target > now:
            await asyncio.sleep(target - now)


    # ---------------------- NEW: one-time setup ----------------------
    async def setup(self):
        # Start networking and heartbeat already created in __init__
        await self.start()
        await self.node.wait_started()

        # Try to bootstrap & discover peers, then wait for quorum
        try:
            await self.bootstrap_join()
        except Exception:
            pass

        await self.wait_discovery(timeout_s=60.0)


        # Install per-round sync routes once
        self.node.route('RoundReady', self._on_round_ready)
        self.node.route('CommDone',  self._on_comm_done)

        # Fresh token sets for barriers
        self._rr_tokens = set()
        self._cd_tokens = set()

        # Internal helpers
        self._barrier_timeout_s = float(os.getenv("BARRIER_TIMEOUT_S", "20"))

    # ---------------------- NEW: one round only ----------------------
    async def run_round(self):
        """Run exactly one ASILO round (no while-loop here)."""
        # Align to slot boundary for this round (global lock-step)
        await self._wait_for_round_slot(self.t_round)

        if not self._epoch_set:
            await self._start_gate.wait()
            self._set_epoch_now()
            self._epoch_set = True
            print(f"[{self.id}] start gate passed; epoch aligned for round 0", flush=True)

        # --- stop guards are enforced by the runner; keep them here as safety ---
        if self.cfg.max_rounds is not None and self.t_round >= self.cfg.max_rounds:
            return
        if self.cfg.max_seconds is not None and (time.time() - self._t_start) >= self.cfg.max_seconds:
            return

        t0 = time.time()

        # ===== verify persistence of last merge (intact) =====
        vec_start = self._get_head_vector()
        if self._last_merge_vec is not None and vec_start is not None:
            cos_after_merge = float(np.dot(vec_start, self._last_merge_vec)
                                    / ((np.linalg.norm(vec_start) + 1e-9) * (np.linalg.norm(self._last_merge_vec) + 1e-9)))
            print(f"[{self.id}] START round={self.t_round} "
                  f"head_norm_before_train={np.linalg.norm(vec_start):.6f} "
                  f"cos_to_last_merge={cos_after_merge:.4f} "
                  f"(last merge round={self._last_merge_round})", flush=True)
            if cos_after_merge < 0.95:
                print(f"[{self.id}] ⚠️ Model drifted since last merge! cos={cos_after_merge:.3f}", flush=True)
        else:
            print(f"[{self.id}] START round={self.t_round} "
                  f"head_norm_before_train={0.0 if vec_start is None else np.linalg.norm(vec_start):.6f} "
                  f"(no previous merge yet)", flush=True)

        # ===== local step (unchanged) =====
        self.trainer.fit_local(self.cfg.capability.local_batches)
        curr_metrics = self.trainer.eval()
        self._last_eval = curr_metrics

        prev_metrics = getattr(self, "_prev_metrics", None)
        u = 0.0
        if self.t_round > 0 and prev_metrics is not None:
            u = self.trainer.compute_utility(prev_metrics, curr_metrics)
            print(f"[{self.id}] round={self.t_round} local utility u={u:.6f} "
                  f"metrics={curr_metrics}-{prev_metrics}", flush=True)

        eps = max(self.cfg.u_eps, 1e-6)
        if abs(u) < eps:
            u = 0.0

        self._no_improve = 0 if u > eps else (getattr(self, "_no_improve", 0) + 1)
        if self._no_improve in (50, 100, 200):
            self.trigger.cooldown_s *= 1.5
        if self._no_improve >= 300:
            print(f"[{self.id}] plateau reached; idling", flush=True)
            await asyncio.sleep(5.0)
            self._prev_metrics = curr_metrics
            self.t_round += 1
            return

        # pheromone EMA + evaporation (unchanged)  :contentReference[oaicite:1]{index=1}
        alpha = 0.2
        if self.t_round > 0:
            self._p_local = (1 - alpha) * self._p_local + alpha * max(0.0, u)
            self.phero.update_self(self._p_local)
            self.phero.evaporate()

        # ====== BARRIER A: RoundReady ======
        await self._broadcast_all(RoundReady(agent_id=self.id, t=self.t_round))
        await self._wait_barrier(self._rr_tokens, tag="RoundReady")

        # ===== speak gate (unchanged core checks) =====
        psi = curr_metrics.get("psi", u)
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

        agg_count = 0
        rolled_back = 0

        # ====== BARRIER B: CommDone ======
        await self._broadcast_all(CommDone(agent_id=self.id, t=self.t_round))
        await self._wait_barrier(self._cd_tokens, tag="CommDone")

        # ===== aggregate buffered heads every N rounds (unchanged core) =====
        if (self.t_round > 0) and (self.t_round % max(1, self.cfg.aggregate_every) == 0):
            async with self._inbuf_lock:
                batch = self._incoming_heads[:]
                self._incoming_heads.clear()

            agg_count, rolled_back = await self._aggregate_batch_if_any(batch)

        # ===== heartbeat & logging (unchanged) =====
        if (self.t_round % 5) == 0:
            await self._broadcast_pheromone(self._p_local)

        self.monitor.log(
            self.t_round, self.bytes_sent, u, self.phero.get_self(),
            curr_metrics.get("f1_val", 0.0),
            agg_count=agg_count, rollback=rolled_back
        )

        # progress and pacing
        self._prev_metrics = curr_metrics
        self.t_round += 1

        dt = self.cfg.round_time_s - (time.time() - t0)
        if self.t_round % 20 == 0:
            f1 = curr_metrics.get("f1_val")
            au = curr_metrics.get("auprc")
            print(f"[{self.id}] r={self.t_round} psi={curr_metrics.get('psi'):.3f} "
                  f"u={u:.4f} p_local={self._p_local:.4f} f1={f1:.3f} auprc={au:.3f} "
                  f"bytes={self.bytes_sent}", flush=True)
        if dt > 0:
            await asyncio.sleep(dt)

    # ---------------------- NEW: teardown ----------------------
    async def teardown(self):
        try:
            await self.send_bye()
        except Exception:
            pass
        try:
            await self.shutdown()
        except Exception:
            pass

    # ---------------------- helpers extracted from old loop ----------------------
    async def _broadcast_all(self, msg_obj):
        for _, pi in self.neighbors.all_peers():
            try:
                await self.node.send(pi.host, pi.port, msg_obj)
            except Exception:
                pass

    def _dynamic_need(self) -> int:
        # number of other currently live peers
        live = self.neighbors.all_peers()
        return max(0, len(live) - 1)

    async def _wait_barrier(self, token_set: set[tuple[int, str]], tag: str):
        deadline = time.time() + getattr(self, "_barrier_timeout_s", 20.0)
        while True:
            have = len({pid for (rt, pid) in token_set if rt == self.t_round})
            need = self._dynamic_need()
            if have >= need:
                return
            if time.time() >= deadline:
                print(f"[{self.id}] {tag} barrier timeout: have={have} need={need} — continuing", flush=True)
                return
            await asyncio.sleep(0.05)

    async def _aggregate_batch_if_any(self, batch):
        agg_count = len(batch)
        rolled_back = 0
        if not batch:
            return 0, 0

        snap = self._snapshot_head()
        pre_eval = self._last_eval or self.trainer.eval()

        sids  = [sid for sid, _, _ in batch]
        vecs  = [v   for _,   v, _ in batch]
        bytes_each = [b for _,   _, b in batch]

        V = np.stack(vecs)
        n = V.shape[0]
        m = V.mean(axis=0)

        def _cos(a, b):
            na = np.linalg.norm(a) + 1e-9
            nb = np.linalg.norm(b) + 1e-9
            return float(np.dot(a, b) / (na * nb))

        cos_th = float(getattr(self.cfg, "cos_gate_thresh", 0.2))
        kept_idx, dropped_idx = [], []
        for i in range(n):
            if _cos(V[i], m) >= cos_th:
                kept_idx.append(i)
            else:
                dropped_idx.append(i)

        if kept_idx:
            Vf = V[kept_idx]
            kept_sids  = [sids[i] for i in kept_idx]
            kept_bytes = [bytes_each[i] for i in kept_idx]
        else:
            Vf = V
            kept_sids  = sids[:]
            kept_bytes = bytes_each[:]
            dropped_idx = []

        dropped_sids  = [sids[i] for i in dropped_idx]
        dropped_bytes = [bytes_each[i] for i in dropped_idx]

        print(f"[{self.id}] aggregate: batch={n} kept={len(kept_sids)} "
              f"dropped={len(dropped_sids)} (cos_th={cos_th})", flush=True)

        # robust aggregation modes preserved  :contentReference[oaicite:2]{index=2}
        mode   = getattr(self.cfg, "agg_mode", "trimmed")
        trim_p = float(getattr(self.cfg, "trim_p", 0.10))
        if mode == "median":
            X = np.stack(Vf)
            if X.shape[0] == 1:
                agg = X[0]
            else:
                m_est = X.mean(axis=0)
                for _ in range(50):
                    d = np.linalg.norm(X - m_est, axis=1) + 1e-6
                    w = 1.0 / d
                    m_new = (X * w[:, None]).sum(axis=0) / w.sum()
                    if np.linalg.norm(m_new - m_est) < 1e-6:
                        break
                    m_est = m_new
                agg = m_est
        elif mode == "trimmed":
            A = [Vf[i] for i in range(Vf.shape[0])]
            # safe trimmed mean
            k = int(trim_p * len(A))
            if k <= 0 or (2*k) >= len(A):
                agg = np.stack(A).mean(axis=0)
            else:
                A = np.sort(np.stack(A), axis=0)[k:-k]
                agg = A.mean(axis=0)
        else:
            agg = Vf.mean(axis=0)

        vec_before = self._get_head_vector()
        pre_m = self._get_metric(pre_eval)
        print(f"[{self.id}] DEBUG before merge: round={self.t_round} metric={pre_m:.4f} "
              f"head_norm={0.0 if vec_before is None else np.linalg.norm(vec_before):.6f}", flush=True)

        self._apply_head_vector(agg)
        vec_after = self._get_head_vector()
        self._last_merge_vec = None if vec_after is None else vec_after.copy()
        self._last_merge_round = self.t_round
        print(f"[{self.id}] stored merged vector for next round verification (round={self.t_round})", flush=True)

        post_eval = self.trainer.eval()
        post_m = self._get_metric(post_eval)
        print(f"[{self.id}] DEBUG after merge:  round={self.t_round} metric={post_m:.4f} "
              f"head_norm={0.0 if vec_after is None else np.linalg.norm(vec_after):.6f} "
              f"delta_norm={0.0 if (vec_after is None or vec_before is None) else np.linalg.norm(vec_after - vec_before):.6f}",
              flush=True)

        tol_abs, tol_rel = self._rollback_tols()
        harm = (post_m + 1e-12) < (pre_m - max(tol_abs, tol_rel * max(1e-12, pre_m)))

        if harm:
            self._restore_head(snap)
            self._last_eval = pre_eval
            rolled_back = 1
            for sid, bsz in zip(dropped_sids, dropped_bytes):
                self.phero.deposit(sid, 0.0, max(1, bsz), reputation_badness=1.0)
            for sid, bsz in zip(kept_sids, kept_bytes):
                self.phero.deposit(sid, 0.0, max(1, bsz), reputation_badness=0.5)
            print(f"[{self.id}] aggregate: ROLLBACK ({self._primary_metric_name()} "
                  f"pre={pre_m:.3f} post={post_m:.3f} tol_abs={tol_abs} tol_rel={tol_rel})", flush=True)
        else:
            self._last_eval = post_eval
            delta_u = max(0.0, post_m - pre_m)
            if kept_sids:
                share = delta_u / len(kept_sids)
                u_cap = float(self.cfg.pheromone.u_max)
                for sid, bsz in zip(kept_sids, kept_bytes):
                    du = min(share, u_cap)
                    self.phero.deposit(sid, du, max(1, bsz), reputation_badness=0.0)
            print(f"[{self.id}] aggregate: ACCEPT kept={kept_sids} "
                  f"(k={len(kept_sids)}/{n}) mode={mode} trim_p={trim_p}", flush=True)

        return agg_count, rolled_back

    # ---------------------- compatibility shim (no while-loop) ----------------------
    async def run_forever(self):
        """Kept for compatibility but WITHOUT an internal while-loop."""
        await self.setup()
        # Do not loop here. The runner will call run_round() repeatedly.
        return




    # ===================== messaging =====================
    async def _on_pheromone(self, msg: PheromoneMsg):
        self.neighbors.update_pheromone(msg.agent_id, msg.p)

    async def _on_model_delta(self, msg: ModelDeltaMsg):
        print(f"[{self.id}] RECEIVED delta(kind={msg.payload.get('kind')} bytes={msg.bytes_size}) "
          f"from {msg.agent_id}", flush=True)
        print(f"[{self.id}] model_id mismatch: {msg.model_id} != {self.cfg.model_id}", flush=True)
        if msg.model_id != self.cfg.model_id:
            return
        print(f"[{self.id}] after if model_id mismatch: {msg.model_id} != {self.cfg.model_id}", flush=True)
        # HEAD payload
        if msg.payload.get("kind") == "head":
            try:
                if "npz" in msg.payload:
                    # sklearn-style model (coef_ + intercept_)
                    arr = np.load(io.BytesIO(msg.payload["npz"]), allow_pickle=False)
                    keys = [k for k in arr.keys() if k not in ("_kind", "kind", "alpha")]

                    # Try sklearn-style
                    if "coef" in keys and "intercept" in keys:
                        vec_in = np.concatenate([arr["coef"].ravel(), arr["intercept"].ravel()]).astype(np.float32)
                    # Try torch-style
                    elif "weight" in keys:
                        parts = [arr["weight"].ravel()]
                        if "bias" in keys:
                            parts.append(arr["bias"].ravel())
                        vec_in = np.concatenate(parts).astype(np.float32)
                    else:
                        # generic numeric flatten, ignoring string arrays
                        parts = []
                        for k in keys:
                            if arr[k].dtype.kind not in {"U", "S", "O"}:  # skip string/object fields
                                parts.append(arr[k].ravel())
                        vec_in = np.concatenate(parts).astype(np.float32)

                    print(f"[{self.id}] loaded HEAD from {msg.agent_id} (keys={keys}, vec_norm={np.linalg.norm(vec_in):.6f})", flush=True)

                elif "state_dict" in msg.payload:
                    # pytorch-style model delta
                    import torch
                    state_dict_bytes = io.BytesIO(msg.payload["state_dict"])
                    state_dict = torch.load(state_dict_bytes, map_location="cpu")
                    flat_params = []
                    for v in state_dict.values():
                        flat_params.append(v.cpu().numpy().ravel())
                    vec_in = np.concatenate(flat_params).astype(np.float32)
                else:
                    raise ValueError("Unsupported head payload format")

                print(f"[{self.id}] loaded HEAD from {msg.agent_id} (vec_in norm={np.linalg.norm(vec_in):.6f})", flush=True)

            except Exception as e:
                print(f"[{self.id}] failed to parse HEAD from {msg.agent_id}: {e!r}", flush=True)
                self.phero.deposit(msg.agent_id, 0.0, max(1, msg.bytes_size), reputation_badness=1.0)
                return


            # cosine gate vs our current head
            vec_cur = self._get_head_vector()
            if vec_cur is None or vec_in is None or vec_cur.shape != vec_in.shape:
                print(f"[{self.id}] skipped cosine check: shape mismatch "
                    f"{None if vec_cur is None else vec_cur.shape} vs {vec_in.shape}", flush=True)
                async with self._inbuf_lock:
                    self._incoming_heads.append((msg.agent_id, vec_in, msg.bytes_size))
                return
            if vec_cur is not None:
                cos = cosine_similarity(vec_cur, vec_in)
                print(f"[{self.id}] cosine check vs {msg.agent_id}: cos={cos:.4f}", flush=True)
                if cos < self.cfg.cos_gate_thresh:
                    print(f"[{self.id}] DROPPED delta from {msg.agent_id} (cos={cos:.4f} < {self.cfg.cos_gate_thresh})", flush=True)
                    self.phero.deposit(msg.agent_id, 0.0, max(1, msg.bytes_size), reputation_badness=1.0)
                    return

            # buffer for aggregation
            async with self._inbuf_lock:
                self._incoming_heads.append((msg.agent_id, vec_in, msg.bytes_size))
                print(f"[{self.id}] BUFFERED delta from {msg.agent_id} (total buffered={len(self._incoming_heads)})", flush=True)
            return

        # PROTO payload
        elif msg.payload.get("kind") == "proto":
            try:
                arr = np.load(io.BytesIO(msg.payload["npz"]), allow_pickle=False)
                protos = {"kind": "proto", "prototypes": {}}
                for k in arr.files:
                    if k == "kind":
                        continue
                    cls = k[1:] if k.startswith("c") else k
                    protos["prototypes"][str(int(cls))] = {
                        "mean": arr[k].astype(np.float32).tolist(),
                        "count": 1
                    }
            except Exception:
                self.phero.deposit(msg.agent_id, 0.0, max(1, msg.bytes_size), reputation_badness=1.0)
                return

            # rollback-guarded proto apply
            snap = self._snapshot_head()
            pre  = self._last_eval or self.trainer.eval()
            pre_m = self._get_metric(pre)
            apply_proto_pull(self.trainer.model, protos)
            post = self.trainer.eval()
            post_m = self._get_metric(post)

            tol_abs, tol_rel = self._rollback_tols()
            harm = (post_m + 1e-12) < (pre_m - max(tol_abs, tol_rel * max(1e-12, pre_m)))
            if harm:
                self._restore_head(snap)
                self._last_eval = pre
                self.phero.deposit(msg.agent_id, 0.0, max(1, msg.bytes_size), reputation_badness=0.7)
                print(f"[{self.id}] rollback PROTO: {self._primary_metric_name()} "
                      f"pre={pre_m:.4f} post={post_m:.4f}", flush=True)
            else:
                self._last_eval = post
                du = min(max(0.0, post_m - pre_m), self.cfg.pheromone.u_max)
                self.phero.deposit(msg.agent_id, du, max(1, msg.bytes_size), reputation_badness=0.0)
            return

        # unknown payload → ignore

    async def _on_hello(self, msg: Hello):
        pass

    async def _on_join(self, msg: Join):
        self.neighbors.add_or_update(msg.agent_id, msg.host, msg.port, msg.capability)
        peers = [(aid, pi.host, pi.port, pi.capability) for aid, pi in self.neighbors.all_peers()]
        await self.node.send(msg.host, msg.port, Welcome(peers=peers))
        intro = Introduce(agent_id=msg.agent_id, host=msg.host, port=msg.port, capability=msg.capability)
        for aid, pi in self.neighbors.all_peers():
            if aid not in (msg.agent_id, self.id):
                try:
                    await self.node.send(pi.host, pi.port, intro)
                except Exception:
                    pass

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
            try:
                await self.node.send(peer.host, peer.port, msg)
            except Exception:
                pass

    def _build_payload(self) -> Tuple[Dict[str, Any], int]:
        # Align builder budget with sender: per-peer cap so size*k_peers <= round cap
        round_cap = int(getattr(self.cfg.capability, "max_bytes_round", 0) or 0)
        k = max(1, int(self.cfg.capability.k_peers))
        budget = round_cap // k

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

        # fallback
        for payload, nbytes, name in (
            (head_payload, head_bytes, "head"),
            (proto_payload, proto_bytes, "proto")
        ):
            if nbytes > 0 and nbytes <= budget:
                print(f"[payload] fallback→{name} bytes={nbytes}")
                return payload, nbytes

        reason = (head_payload or {}).get("reason") or (proto_payload or {}).get("reason") or "no_artifact_fits"
        print(f"[payload] none reason={reason} budget={budget} head={head_bytes} proto={proto_bytes}")
        return {"kind": "none", "reason": reason, "budget": budget}, 0

    async def _broadcast_delta(self, payload: Dict[str, Any], size: int):
        # --- choose peers ---
        if self.t_round < 3:                          # <-- first threeround?
            peers = list(self.neighbors.all_peers())   # [(id, PeerInfo), ...]
            random.shuffle(peers)
            selected = peers[:self.cfg.capability.k_peers]
        else:
            selected = self.neighbors.topk(self.cfg.capability.k_peers)

        # --- broadcast ---
        for aid, peer in selected:
            msg = ModelDeltaMsg(
                agent_id=self.id,
                t=self.t_round,
                model_id=self.cfg.model_id,
                strategy=payload.get("kind", "unknown"),
                payload=payload,
                bytes_size=size,
            )
            print(f"[{self.id}] → sending delta(kind={payload.get('kind')} bytes={size}) "
                f"to {aid}@{peer.host}:{peer.port}", flush=True)
            try:
                await self.node.send(peer.host, peer.port, msg)
            except Exception:
                pass


    async def send_bye(self):
        b = Bye(agent_id=self.id)
        for _, pi in self.neighbors.all_peers():
            try:
                await self.node.send(pi.host, pi.port, b)
            except Exception:
                pass

    def _get_head_vector(self) -> np.ndarray | None:
        """Return flattened head (classifier) vector for both sklearn and torch models."""
        # sklearn models
        coef = getattr(self.trainer.model, "coef_", None)
        inter = getattr(self.trainer.model, "intercept_", None)
        if coef is not None and inter is not None:
            return np.concatenate([coef.ravel(), np.atleast_1d(inter).ravel()]).astype(np.float32)

        # torch models — only final Linear head
        try:
            import torch
            from Asilo_1.fl.artifacts.head_delta import _find_linear_head_torch
            head = _find_linear_head_torch(self.trainer.model)
            if head is None or not hasattr(head, "weight"):
                return None
            w = head.weight.detach().cpu().numpy().ravel()
            b = head.bias.detach().cpu().numpy().ravel() if getattr(head, "bias", None) is not None else np.array([], dtype=np.float32)
            return np.concatenate([w, b]).astype(np.float32)
        except Exception:
            return None



    def _apply_head_vector(self, vec: np.ndarray):
        # Dispatch so existing call sites work
        return self._apply_head_vector_avg(vec)

    def _apply_head_vector_avg(self, vec: np.ndarray):
        """
        Average current head vector with `vec` and write back.
        Supports sklearn (coef_/intercept_) and torch Linear head.
        """
        cur = self._get_head_vector()
        if cur is None:
            print(f"[{self.id}] _apply_head_vector_avg: current head is None → no-op", flush=True)
            return
        if not isinstance(vec, np.ndarray):
            vec = np.asarray(vec)
        if cur.shape != vec.shape:
            print(f"[{self.id}] _apply_head_vector_avg: shape mismatch cur={cur.shape} vec={vec.shape} → no-op", flush=True)
            return

        merged = (cur.astype(np.float64) + vec.astype(np.float64)) / 2.0  # stable avg

        # --- sklearn path ---
        coef = getattr(self.trainer.model, "coef_", None)
        inter = getattr(self.trainer.model, "intercept_", None)
        if coef is not None and inter is not None:
            csz = coef.size
            new_coef = merged[:csz].reshape(coef.shape).astype(coef.dtype, copy=False)
            new_inter = merged[csz:].reshape(inter.shape).astype(inter.dtype, copy=False)
            self.trainer.model.coef_ = new_coef
            self.trainer.model.intercept_ = new_inter

            # optional verification print
            after = self._get_head_vector()
            dn = float(np.linalg.norm(after - cur)) if after is not None else float("nan")
            print(f"[{self.id}] _apply_head_vector_avg (sklearn): Δ‖w‖={dn:.6e}", flush=True)
            return

        # --- torch path (final Linear head only) ---
        try:
            import torch
            from Asilo_1.fl.artifacts.head_delta import _find_linear_head_torch
            head = _find_linear_head_torch(self.trainer.model)
            if head is None or not hasattr(head, "weight"):
                print(f"[{self.id}] _apply_head_vector_avg: torch head not found → no-op", flush=True)
                return

            with torch.no_grad():
                w_num = head.weight.numel()
                W = merged[:w_num].reshape(tuple(head.weight.shape))
                device = head.weight.device
                head.weight.copy_(torch.from_numpy(W).to(device=device, dtype=head.weight.dtype))

                b_has = getattr(head, "bias", None) is not None
                if b_has:
                    b_num = head.bias.numel()
                    B = merged[w_num:w_num + b_num].reshape(tuple(head.bias.shape))
                    head.bias.copy_(torch.from_numpy(B).to(device=device, dtype=head.bias.dtype))

            # optional verification print
            after = self._get_head_vector()
            dn = float(np.linalg.norm(after - cur)) if after is not None else float("nan")
            print(f"[{self.id}] _apply_head_vector_avg (torch): Δ‖w‖={dn:.6e}", flush=True)
            return
        except Exception as e:
            print(f"[{self.id}] _apply_head_vector_avg torch apply failed: {e}", flush=True)
            return


    def _snapshot_head(self):
        coef = getattr(self.trainer.model, "coef_", None)
        inter = getattr(self.trainer.model, "intercept_", None)
        if coef is None or inter is None:
            return None
        return (coef.copy(), inter.copy())

    def _restore_head(self, snap):
        if not snap:
            return
        coef, inter = snap
        self.trainer.model.coef_ = coef
        self.trainer.model.intercept_ = inter

    # ---- metric / rollback helpers ----
    def _primary_metric_name(self) -> str:
        return getattr(self.cfg, "primary_metric", "f1_val")

    def _get_metric(self, ev: dict) -> float:
        try_keys = [self._primary_metric_name(), "f1_val", "f1", "auprc", "accuracy"]
        for k in try_keys:
            if isinstance(ev, dict) and k in ev:
                try:
                    return float(ev[k])
                except Exception:
                    pass
        return float("nan")

    def _rollback_tols(self) -> tuple[float, float]:
        tol_abs = float(getattr(self.cfg, "rollback_tol", 0.002))
        tol_rel = float(getattr(self.cfg, "rollback_rel", 0.0))
        return tol_abs, tol_rel
