import os
import asyncio, time, numpy as np
from typing import Dict, Tuple, List, DefaultDict, Optional
from dataclasses import dataclass
from collections import defaultdict

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


# -------------------------
# Shared config dataclasses
# -------------------------
@dataclass
class FedAvgClientConfig:
    agent_id: str
    host: str
    port: int
    model_id: str
    round_time_s: float = 0.1
    max_rounds: int = 50
    client_barrier_timeout_s: float = 5.0  # wait for orchestrator agg(r)


@dataclass
class FedAvgOrchestratorConfig:
    agent_id: str
    host: str
    port: int
    model_id: str
    round_time_s: float = 0.1
    max_rounds: int = 50  # for logging only


# --------------------------------
# FedAvg Client (trains + waits)
# --------------------------------
class FedAvgClientAgent:
    """
    Per-round client behavior:
      1) fit_local -> eval
      2) send local weights to orchestrator (strategy='fedavg-local')
      3) WAIT for orchestrator broadcast agg(r) (strategy='fedavg-agg')
      4) apply global weights, finish round
    """
    def __init__(self, cfg: FedAvgClientConfig, trainer: LocalTrainer, orchestrator: Tuple[str, Tuple[str, int]]):
        self.cfg = cfg
        self.trainer = trainer
        self.node = P2PNode(cfg.host, cfg.port)
        self.id = cfg.agent_id
        self.model_id = cfg.model_id
        self.bytes_sent = 0
        exp_name = os.environ.get("EXP_NAME", "fedavg")
        self.monitor = CSVMonitor(cfg.agent_id, exp_name)
        self._f1_streak_hi = 0  # >>> sanity-add
        # orchestrator info
        self._orch_id, (self._orch_host, self._orch_port) = orchestrator

        # barrier: round -> event set when agg(r) applied
        self._agg_events: Dict[int, asyncio.Event] = {}

        # routes
        self.node.route("ModelDeltaMsg", self._on_model_delta)
        self.node.route("Welcome", self._on_welcome)

    async def start(self):
        await self.node.start()
        # join the orchestrator
        j = Join(agent_id=self.id, host=self.cfg.host, port=self.cfg.port, capability="FedAvgClient")
        await self.node.send(self._orch_host, self._orch_port, j)

    async def run_round(self, round_idx: int):
        t0 = time.time()

        # local train + eval
        self.trainer.fit_local(5)
        metrics = self.trainer.eval()
        f1 = float(metrics.get("f1_val", 0.0))

        # >>> sanity-add: simple streak tracker and optional deep probe
        if f1 >= 0.99:
            self._f1_streak_hi += 1
        else:
            self._f1_streak_hi = 0
        if self._f1_streak_hi >= 3 and round_idx <= 5:
            await self._sanity_probe(round_idx, f1)

        # send local weights
        weights, size = self._get_weights()
        if weights is not None:
            payload = {"weights": weights.tolist()}
            msg = ModelDeltaMsg(
                agent_id=self.id, t=round_idx, model_id=self.model_id,
                strategy="fedavg-local", payload=payload, bytes_size=size
            )
            await self.node.send(self._orch_host, self._orch_port, msg)
            self.bytes_sent += size
            print(f"[{self.id}] SENT local → {self._orch_id}@{self._orch_host}:{self._orch_port} "
                  f"size={size}, norm={np.linalg.norm(weights):.6f}, f1={f1:.4f}, r={round_idx}", flush=True)

        # wait for orchestrator broadcast of agg(r)
        evt = self._agg_events.setdefault(round_idx, asyncio.Event())
        try:
            await asyncio.wait_for(evt.wait(), timeout=self.cfg.client_barrier_timeout_s)
            print(f"[{self.id}] BARRIER OK r={round_idx} (applied agg)", flush=True)
        except asyncio.TimeoutError:
            print(f"[{self.id}] BARRIER TIMEOUT r={round_idx} (may apply late)", flush=True)

        # monitor + pacing
        self.monitor.log(round_idx, self.bytes_sent, 0.0, 0.0, f1)
        dt = self.cfg.round_time_s - (time.time() - t0)
        if dt > 0:
            await asyncio.sleep(dt)

    # ---------- handlers ----------
    async def _on_model_delta(self, msg: ModelDeltaMsg):
        if msg.strategy != "fedavg-agg":
            return
        # apply global
        w = msg.payload.get("weights")
        if w is None:
            return
        arr = np.asarray(w, dtype=np.float32)
        self._set_weights(arr)
        print(f"[{self.id}] APPLIED agg r={msg.t} (size={arr.size}, norm={np.linalg.norm(arr):.6f})", flush=True)
        # release barrier
        evt = self._agg_events.setdefault(msg.t, asyncio.Event())
        if not evt.is_set():
            evt.set()

    async def _on_welcome(self, msg: Welcome):
        # no-op; we already know the orchestrator
        return

    # ---------- model helpers ----------
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

    def _get_weights(self):
        # sklearn-like
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
        # sklearn-like
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
    # >>> sanity-add
    async def _sanity_probe(self, round_idx: int, f1: float):
        """
        Try to detect common causes of 'too-good' validation:
          - val labels single-class
          - predictions single-class
          - eval likely using train split
        This is best-effort and fully optional; does not affect training.
        """
        try:
            # 1) Attempt to access a validation set
            X_val = getattr(self.trainer, "X_val", None)
            y_val = getattr(self.trainer, "y_val", None)
            val_loader = getattr(self.trainer, "val_loader", None)
            predict_fn = getattr(self.trainer, "predict", None)

            y_true = None
            y_pred = None

            if X_val is not None and y_val is not None and callable(predict_fn):
                # sklearn-like path
                y_true = np.asarray(y_val)
                y_pred = np.asarray(predict_fn(X_val))
            elif val_loader is not None and hasattr(self.trainer, "model"):
                # torch-like path: run a tiny loop
                try:
                    self.trainer.model.eval()
                except Exception:
                    pass
                ys, ps = [], []
                count = 0
                for batch in val_loader:
                    # very defensive unpacking
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        xb, yb = batch[0], batch[1]
                    else:
                        continue
                    # basic forward
                    try:
                        import torch
                        with torch.no_grad():
                            logits = self.trainer.model(xb)
                            preds = torch.argmax(logits, dim=-1).cpu().numpy()
                            ys.append(yb.cpu().numpy() if hasattr(yb, "cpu") else np.asarray(yb))
                            ps.append(preds)
                            count += len(preds)
                            if count >= 256:  # small probe
                                break
                    except Exception:
                        break
                if ys and ps:
                    y_true = np.concatenate(ys)
                    y_pred = np.concatenate(ps)

            if y_true is None or y_pred is None:
                print(f"[{self.id}] SANITY r={round_idx}: cannot access val set; "
                      f"f1={f1:.4f}. Check that trainer.eval() uses held-out data.", flush=True)
                return

            # 2) Basic distributions
            uniq_true, cnt_true = np.unique(y_true, return_counts=True)
            uniq_pred, cnt_pred = np.unique(y_pred, return_counts=True)

            msg = (f"[{self.id}] SANITY r={round_idx}: f1={f1:.4f} "
                   f"| y_val classes={list(zip(uniq_true.tolist(), cnt_true.tolist()))} "
                   f"| y_pred classes={list(zip(uniq_pred.tolist(), cnt_pred.tolist()))}")
            print(msg, flush=True)

            # 3) Heuristics & warnings
            if len(uniq_true) == 1:
                print(f"[{self.id}] WARN r={round_idx}: validation labels are single-class → "
                      f"macro/micro F1 can saturate at 1.0. Use stratified split per client.", flush=True)
            if len(uniq_pred) == 1 and uniq_pred[0] in uniq_true and cnt_pred[0] == len(y_pred):
                print(f"[{self.id}] WARN r={round_idx}: model predicts a single class on val. "
                      f"If val is also single-class, F1=1.0 is trivial; verify split.", flush=True)

        except Exception as e:
            print(f"[{self.id}] SANITY r={round_idx}: probe failed ({e}).", flush=True)



# ----------------------------------------
# FedAvg Orchestrator (aggregate + broadcast)
# ----------------------------------------
class FedAvgOrchestrator:
    """
    Collects client local weights per round, FedAvg (mean), and broadcasts the global vector.
    No local training. Maintains a 'current_global' vector for diagnostics/broadcast.
    """
    def __init__(self, cfg: FedAvgOrchestratorConfig):
        self.cfg = cfg
        self.node = P2PNode(cfg.host, cfg.port)
        self.id = cfg.agent_id
        self.model_id = cfg.model_id
        self.round = 0
        self.clients: set[str] = set()
        self.client_endpoints: Dict[str, Tuple[str, int]] = {}
        self._updates_by_round: DefaultDict[int, List[np.ndarray]] = defaultdict(list)
        self.current_global: Optional[np.ndarray] = None

        # routes
        self.node.route("ModelDeltaMsg", self._on_model_delta)
        self.node.route("Join", self._on_join)

    async def start(self):
        await self.node.start()

    async def run_round(self, round_idx: int):
        self.round = round_idx
        t0 = time.time()

        # wait window (use most of round time)
        expected = len(self.clients)
        wait_budget = max(0.0, self.cfg.round_time_s * 0.8)
        poll_sleep = 0.01
        waited = 0.0

        while waited < wait_budget:
            got = len(self._updates_by_round.get(round_idx, []))
            if got >= expected and expected > 0:
                break
            await asyncio.sleep(poll_sleep)
            waited += poll_sleep

        # aggregate
        updates = self._updates_by_round.pop(round_idx, [])
        if updates:
            stack = np.stack(updates, axis=0)
            norms = [float(np.linalg.norm(u)) for u in stack]
            print(f"[{self.id}] AGG r={round_idx} got={len(updates)}/{max(expected,1)} "
                  f"shape={stack.shape} norms={norms}", flush=True)
            agg = np.mean(stack, axis=0).astype(np.float32, copy=False)
            agg_norm = float(np.linalg.norm(agg))
            prev_norm = float(np.linalg.norm(self.current_global)) if self.current_global is not None else 0.0
            self.current_global = agg
            print(f"[{self.id}] AGG r={round_idx} mean_norm={agg_norm:.6f} (prev={prev_norm:.6f})", flush=True)

            # broadcast global
            payload = {"weights": self.current_global.tolist()}
            bmsg = ModelDeltaMsg(
                agent_id=self.id, t=round_idx, model_id=self.model_id,
                strategy="fedavg-agg", payload=payload, bytes_size=int(self.current_global.nbytes)
            )
            for cid in sorted(self.clients):
                host, port = self.client_endpoints.get(cid, (None, None))
                if host is None or port is None:
                    print(f"[{self.id}] WARN missing endpoint for {cid}; skip", flush=True)
                    continue
                await self.node.send(host, port, bmsg)
                print(f"[{self.id}] BROADCAST agg → {cid}@{host}:{port} "
                      f"size={bmsg.bytes_size}, norm={agg_norm:.6f}, r={round_idx}", flush=True)
        else:
            print(f"[{self.id}] WARN r={round_idx} received 0 updates (expected ~{expected}).", flush=True)

        # pace
        dt = self.cfg.round_time_s - (time.time() - t0)
        if dt > 0:
            await asyncio.sleep(dt)

    # ---------- handlers ----------
    async def _on_model_delta(self, msg: ModelDeltaMsg):
        if msg.strategy != "fedavg-local":
            return
        w = msg.payload.get("weights", None)
        if w is None:
            return
        arr = np.asarray(w, dtype=np.float32)
        self._updates_by_round[msg.t].append(arr)
        print(f"[{self.id}] RECV local r={msg.t} from {msg.agent_id} "
              f"(size={arr.size}, norm={np.linalg.norm(arr):.6f}, "
              f"count={len(self._updates_by_round[msg.t])})", flush=True)

    async def _on_join(self, msg: Join):
        self.clients.add(msg.agent_id)
        self.client_endpoints[msg.agent_id] = (msg.host, msg.port)
        # You can choose to send a Welcome; clients don't require it, but it's OK.
        await self.node.send(msg.host, msg.port, Welcome(peers=[(self.id, self.cfg.host, self.cfg.port, "orchestrator")]))
        print(f"[{self.id}] JOIN {msg.agent_id} endpoint={self.client_endpoints[msg.agent_id]} "
              f"clients={sorted(self.clients)}", flush=True)
        
