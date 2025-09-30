# Asilo_1/p2p/neighbor_manager.py
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class PeerInfo:
    id: str
    host: str
    port: int
    capability: str | None = None
    last_seen: float = 0.0
    pheromone: float = 0.0

class NeighborManager:
    def __init__(self, peers: Dict[str, Tuple[str, int]], k_default: int, ttl_seconds: float = 20.0):
        self.k = k_default
        self.ttl = ttl_seconds
        self.members: Dict[str, PeerInfo] = {
            aid: PeerInfo(aid, host, port, None, time.time(), 0.0)
            for aid, (host, port) in peers.items()
        }

    def add_or_update(self, aid: str, host: str, port: int, capability: str | None = None):
        pi = self.members.get(aid)
        if pi is None:
            self.members[aid] = PeerInfo(aid, host, port, capability, time.time(), 0.0)
        else:
            pi.host, pi.port = host, port
            if capability: pi.capability = capability
            pi.last_seen = time.time()

    def update_pheromone(self, aid: str, p: float):
        pi = self.members.get(aid)
        if pi is None:
            return
        pi.pheromone = p
        pi.last_seen = time.time()

    def remove(self, aid: str):
        self.members.pop(aid, None)

    def sweep_dead(self) -> List[str]:
        now = time.time()
        dead = [aid for aid, pi in self.members.items() if (now - pi.last_seen) > self.ttl]
        for aid in dead:
            self.members.pop(aid, None)
        return dead

    def topk(self, k: int | None = None) -> List[Tuple[str, PeerInfo]]:
        kk = k or self.k
        live = list(self.members.items())
        live.sort(key=lambda kv: -kv[1].pheromone)
        return live[:kk]

    def all_peers(self) -> List[Tuple[str, PeerInfo]]:
        return list(self.members.items())
