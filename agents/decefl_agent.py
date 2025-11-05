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
    # --- NEW: optional gating / trimming knobs (can be wired from YAML later) ---
    cos_gate_thresh: float = float("nan")  # NaN → disabled; else keep only if cos >= thresh
    trim_p: float = 0.0                    # 0.0 → no trimming; else [0..0.49] trim at both tails

def coord_median(stack: np.ndarray) -> np.ndarray:
    return np.median(stack, axis=0)

def weighted_mean(stack: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    if w is None:  # simple mean by default
        return np.mean(stack, axis=0)
    w = np.asarray(w, dtype=np.float64)
    w = w / (np.sum(w) + 1e-12)
    return (stack * w[:, None]).sum(axis=0)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a); bn = np.linalg.norm(b)
    if an == 0.0 or bn == 0.0: return 0.0
    return float(np.dot(a, b) / (an * bn))

class DeceFLAgent:
    """
    Decentralized, round-synchronous DeceFL with telemetry:
    - Orchestrator calls run_one_round() (no internal while-loop).
    - Each round: local fit -> send to K peers (log_edge: send) ->
      wait -> receive (log_edge: recv) -> classify each delta (log_agg_decision) -> aggregate kept deltas only.
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

        # peers: top-K neighborhood (id -> (host,port))
        self.peers: Dict[str, Tuple[str,int]] = peers
        self.round = 0
        self.bytes_sent = 0

        # inbox: round_id -> List[Tuple[from_id, vec, bytes]]
        self.inbox: Dict[int, List[Tuple[str, np.ndarray, int]]] = {}

        self.node.route("ModelDeltaMsg", self._on_model_delta)
        self.node.route("Join", self._on_join)
        self.node.route("Welcome", self._on_welcome)

    async def start(self):
        await self.node.start()
        # handshake
        j = Join(agent_id=self.id, host=self.cfg.host, port=self.cfg.port, capability="DeceFL")
        for pid, (h, p) in self.peers.items():
            await self.node.send(h, p, j)

    async def shutdown(self):
        stop = getattr(self.node, "stop", None)
        if callable(stop): await stop()

    async def run_forever(self):
        """Deprecated: kept for compatibility; no loop here."""
        await self.start()

    def _desired_min_incoming(self) -> int:
        # Aim to use ALL available neighbors this round (K/K if they arrive in time)
        return max(1, len(self.peers))

    async def run_one_round(self, local_epochs: int = 5, wait_timeout_s: float = 1.0, grace_s: float = 0.15):
        t0 = time.time()

        # 1) Local step
        self.trainer.fit_local(local_epochs)
        metrics = self.trainer.eval()

        # 2) Send head to peers (+ edge log)
        weights, size = self._get_weights()
        if weights is not None and self.peers:
            payload = {"weights": weights.tolist()}
            msg = ModelDeltaMsg(agent_id=self.id, t=self.round, model_id=self.cfg.model_id,
                                strategy="decefl", payload=payload, bytes_size=int(size))
            for pid, (h, p) in self.peers.items():
                await self.node.send(h, p, msg)
                self.bytes_sent += int(size)
                # ---- log edge: SEND ----
                self.monitor.log_edge(t=self.round, src=self.id, dst=pid, bytes_sz=int(size),
                                      kind="model", action="send", cos=None, reason="head")
            print(f"[{self.id}] SENT r={self.round} -> K={len(self.peers)} size={size} norm={np.linalg.norm(weights):.6f}", flush=True)

        # 3) Wait for ALL peers or until timeout + small grace to catch stragglers
        target = self._desired_min_incoming()
        await self._await_round_updates(self.round, min_count=target, timeout_s=wait_timeout_s)
        if grace_s > 0:
            await asyncio.sleep(grace_s)

        # 4) Aggregate THIS round with per-delta decisions
        buf = self.inbox.pop(self.round, [])
        got = len(buf)
        k = len(self.peers)

        if got == 0:
            print(f"[{self.id}] WARN: no peer updates for r={self.round}", flush=True)
        else:
            # classify each received delta vs. current head (before aggregation)
            before_vec, _ = self._get_weights()
            kept: List[np.ndarray] = []
            late = len(self.inbox.get(self.round, []))  # anything that snuck in after pop

            for from_id, vec, bsz in buf:
                cos = _cosine(before_vec, vec) if (before_vec is not None and vec is not None) else 0.0
                keep = True
                reason = "ok"

                # cosine gating if enabled (threshold is a number, not NaN)
                if not np.isnan(self.cfg.cos_gate_thresh):
                    if cos < float(self.cfg.cos_gate_thresh):
                        keep = False
                        reason = f"cos<{self.cfg.cos_gate_thresh}"

                # log per-delta decision
                self.monitor.log_agg_decision(
                    t=self.round, agent=self.id, from_id=from_id, bytes_sz=int(bsz),
                    decision=("keep" if keep else "drop"),
                    cos_thresh=float(self.cfg.cos_gate_thresh) if not np.isnan(self.cfg.cos_gate_thresh) else float("nan"),
                    mode=self.cfg.agg_mode, trim_p=float(self.cfg.trim_p), rolled_back=0
                )
                # also mirror to edges.csv with the cosine observed on receive path
                # (denote the decision in "reason")
                self.monitor.log_edge(
                    t=self.round, src=from_id, dst=self.id, bytes_sz=int(bsz),
                    kind="model", action=("recv_keep" if keep else "recv_drop"),
                    cos=cos, reason=reason
                )

                if keep:
                    kept.append(vec)

            # optional trimming (if user later sets trim_p>0)
            stack = None
            if kept:
                stack = np.stack(kept, axis=0)
                if 0.0 < self.cfg.trim_p < 0.5:
                    m = stack.shape[0]
                    if m >= 3:
                        # trim equally at both tails by L2 norm rank
                        norms = np.linalg.norm(stack, axis=1)
                        order = np.argsort(norms)
                        trim = int(np.floor(self.cfg.trim_p * m))
                        sel = order[trim: m - trim] if (m - 2*trim) >= 1 else order
                        stack = stack[sel]

            if stack is not None and stack.size > 0:
                agg = coord_median(stack) if self.cfg.agg_mode.lower() == "median" else weighted_mean(stack)
                before_norm = np.linalg.norm(before_vec) if before_vec is not None else 0.0
                self._set_weights(agg)
                after_vec, _ = self._get_weights()
                after_norm = np.linalg.norm(after_vec) if after_vec is not None else 0.0
                delta_norm = (np.linalg.norm(after_vec - before_vec)
                              if (before_vec is not None and after_vec is not None) else 0.0)
                kept_norms = [float(np.linalg.norm(u)) for u in kept]
                print(f"[{self.id}] AGG r={self.round} kept={len(kept)}/{got} of K={k} "
                      f"norms={kept_norms} mode={self.cfg.agg_mode} Δ={delta_norm:.6f}", flush=True)
            else:
                print(f"[{self.id}] AGG r={self.round} kept=0/{got} of K={k} → skip apply", flush=True)

            # late arrivals after grace → mark explicitly
            if late:
                # re-check and purge any lingering entries for this round
                extra = self.inbox.pop(self.round, [])
                for from_id, vec, bsz in extra:
                    self.monitor.log_agg_decision(
                        t=self.round, agent=self.id, from_id=from_id, bytes_sz=int(bsz),
                        decision="late_drop",
                        cos_thresh=float(self.cfg.cos_gate_thresh) if not np.isnan(self.cfg.cos_gate_thresh) else float("nan"),
                        mode=self.cfg.agg_mode, trim_p=float(self.cfg.trim_p), rolled_back=0
                    )
                print(f"[{self.id}] NOTE: dropping {len(extra)} late arrivals for r={self.round}", flush=True)

        # 5) Per-round metrics & advance
        self.monitor.log(self.round, self.bytes_sent, 0.0, 0.0, metrics.get("f1_val", 0.0))
        self.round += 1

        # 6) Pacing
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
        bsz = int(getattr(msg, "bytes_size", 0) or 0)
        self.inbox.setdefault(r, []).append((msg.agent_id, arr, bsz))
        # ---- log edge: RECV (raw arrival; final keep/drop is decided at agg time) ----
        self.monitor.log_edge(t=r, src=msg.agent_id, dst=self.id, bytes_sz=bsz,
                              kind="model", action="recv", cos=None, reason=None)
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
                    if isinstance(m, nn.Linear): last = sub
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
