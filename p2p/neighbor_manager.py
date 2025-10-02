# Asilo_1/p2p/neighbor_manager.py
import time
from dataclasses import dataclass
import random
from typing import Dict, List, Tuple

@dataclass
class PeerInfo:
    id: str
    host: str
    port: int
    capability: str | None = None
    last_seen: float = 0.0
    pheromone: float = 0.0
    # NEW: require multiple consecutive misses before eviction
    miss_count: int = 0

class NeighborManager:
    def __init__(self, peers: Dict[str, Tuple[str, int]], k_default: int, ttl_seconds: float = 20.0):
        self.k = k_default
        self.ttl = ttl_seconds
        self.members: Dict[str, PeerInfo] = {
            aid: PeerInfo(aid, host, port, None, time.time(), 0.0, 0)
            for aid, (host, port) in peers.items()
        }

    def add_or_update(self, aid: str, host: str, port: int, capability: str | None = None):
        pi = self.members.get(aid)
        now = time.time()
        if pi is None:
            self.members[aid] = PeerInfo(aid, host, port, capability, now, 0.0, 0)
        else:
            pi.host, pi.port = host, port
            if capability:
                pi.capability = capability
            pi.last_seen = now
            pi.miss_count = 0  # RESET on any contact

    def update_pheromone(self, aid: str, p: float):
        pi = self.members.get(aid)
        if pi is None:
            return
        pi.pheromone = p
        pi.last_seen = time.time()
        pi.miss_count = 0  # RESET on heartbeat/contact

    def remove(self, aid: str):
        self.members.pop(aid, None)

    def sweep_dead(self) -> List[str]:
        """
        Mark peers missed past TTL; only evict after 2 consecutive misses.
        Returns the list of actually-removed peers.
        """
        now = time.time()
        removed: List[str] = []
        for aid, pi in list(self.members.items()):
            if (now - pi.last_seen) > self.ttl:
                pi.miss_count += 1
                if pi.miss_count >= 2:  # <-- grace period
                    self.members.pop(aid, None)
                    removed.append(aid)
        return removed

    def _live_items(self) -> List[Tuple[str, PeerInfo]]:
        """Peers within TTL are considered live (even if not top pheromone)."""
        now = time.time()
        return [(aid, pi) for aid, pi in self.members.items() if (now - pi.last_seen) <= self.ttl]

    def topk(self, k: int | None = None) -> List[Tuple[str, PeerInfo]]:
        """
        Return up to k live peers, sorted by decreasing pheromone.
        Peers past TTL are excluded even if not yet evicted (grace period).
        """
        kk = k or self.k
        live = self._live_items()
        live.sort(key=lambda kv: -kv[1].pheromone)
        return live[:kk]

    def all_peers(self) -> List[Tuple[str, PeerInfo]]:
        return list(self.members.items())
    

    def choose_k(self, k: int | None = None, epsilon: float = 0.05) -> List[Tuple[str, PeerInfo]]:
        """epsilon-greedy: with prob eps pick random live peers, else use topk."""
        kk = k or self.k
        live = self._live_items()
        if not live:
            return []
        if random.random() < epsilon and len(live) > kk:
            return random.sample(live, kk)
        return self.topk(kk)

