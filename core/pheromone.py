# Asilo_1/core/pheromone.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
import random

@dataclass
class PheromoneConfig:
    rho: float = 0.005               # evaporation per round (on peer taus)
    eta: float = 2.0                # deposition scale
    tau0: float = 0.01              # initial pheromone for unknown peers
    temp: float = 0.5               # softmax temperature (if you want sampling)
    eps_greedy: float = 0.15        # exploration probability
    u_max: float = 0.5              # clip ΔU in [0, u_max]
    beta_ucb: float = 0.6           # UCB exploration weight (a bit larger)
    reputation_kappa: float = 0.5   # weight for (1 - kappa*b_ij)

class PeerState:
    __slots__ = ("tau", "contacts", "reputation")
    def __init__(self, tau: float):
        self.tau: float = tau
        self.contacts: int = 0
        self.reputation: float = 0.0  # 0=good, 1=bad

class PheromoneTable:
    """
    Tracks pheromone for peers (tau[peer]) and the agent's own broadcast
    pheromone (self_p). The 'self_p' is what you actually announce and
    gate on; tau entries are what you maintain ABOUT other peers.
    """
    def __init__(self, cfg: PheromoneConfig, self_id: Optional[str] = None):
        self.cfg = cfg
        self.self_id = self_id
        self.table: Dict[str, PeerState] = {}
        self.self_p: float = cfg.tau0  # what this agent broadcasts (set from Agent)

    # ---------- housekeeping ----------
    def ensure(self, peer_id: str):
        if peer_id not in self.table:
            self.table[peer_id] = PeerState(tau=self.cfg.tau0)

    def evaporate(self):
        """Evaporate peer taus. Do NOT evaporate self broadcast pheromone."""
        for pid, st in self.table.items():
            if self.self_id is not None and pid == self.self_id:
                # self entry (if ever present) should not drift to zero;
                # we use self_p for logging/broadcasting anyway.
                continue
            st.tau *= (1.0 - self.cfg.rho)

    # ---------- updates ----------
    def update_self(self, p_local: float):
        """Keep the broadcast value that the agent will announce."""
        self.self_p = max(0.0, float(p_local))

    def deposit(self, peer_id: str, delta_u: float, bytes_sent: int,
                reputation_badness: float = 0.0):
        """
        Reward a peer when its message improved us.
        Scale by KB, not raw bytes, to avoid making deposits microscopic.
        """
        du = max(0.0, min(self.cfg.u_max, float(delta_u)))
        kb = max(1.0, float(bytes_sent) / 1024.0)  # per-KB normalization
        self.ensure(peer_id)
        st = self.table[peer_id]
        rep = max(0.0, min(1.0, float(reputation_badness)))
        rep_factor = (1.0 - self.cfg.reputation_kappa * rep)
        st.tau += self.cfg.eta * rep_factor * (du / kb)
        st.contacts += 1
        st.reputation = rep

    # ---------- scoring & selection ----------
    def _ucb_bonus(self, st: PeerState) -> float:
        # Standard-ish UCB: sqrt(log(n+2)/(n+1))
        return self.cfg.beta_ucb * math.sqrt(
            math.log(st.contacts + 2.0) / (st.contacts + 1.0)
        )

    def _scores(self) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        for pid, st in self.table.items():
            out.append((pid, st.tau + self._ucb_bonus(st)))
        return out

    def choose_peers(self, k: int) -> List[str]:
        if not self.table:
            return []
        scores = self._scores()

        # eps-greedy exploration
        if random.random() < self.cfg.eps_greedy:
            random.shuffle(scores)
        else:
            scores.sort(key=lambda x: -x[1])

        return [pid for pid, _ in scores[:k]]

    # alias for legacy callers
    def topk(self, k: int) -> List[str]:
        return self.choose_peers(k)

    # ---------- accessors ----------
    def get_tau(self, peer_id: str) -> float:
        st = self.table.get(peer_id)
        return st.tau if st else self.cfg.tau0

    def get_self(self) -> float:
        """Value you broadcast (what the agent gates on)."""
        return self.self_p

    def score_of(self, peer_id: str) -> float:
        st = self.table.get(peer_id)
        if not st:
            return self.cfg.tau0
        return st.tau + self._ucb_bonus(st)
