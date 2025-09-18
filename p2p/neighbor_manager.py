from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class PeerInfo:
    host: str
    port: int
    pheromone: float = 0.0

class NeighborManager:
    def __init__(self, initial_peers: Dict[str, Tuple[str, int]], k_default: int = 2):
        self.peers: Dict[str, PeerInfo] = {
            aid: PeerInfo(host=h, port=p) for aid, (h, p) in initial_peers.items()
        }
        self.k_default = k_default

    def update_pheromone(self, agent_id: str, p: float):
        if agent_id in self.peers:
            self.peers[agent_id].pheromone = p

    def topk(self, k: int | None = None) -> List[Tuple[str, PeerInfo]]:
        kk = k if k is not None else self.k_default
        return sorted(self.peers.items(), key=lambda kv: kv[1].pheromone, reverse=True)[:kk]

    def endpoints(self) -> Dict[str, Tuple[str, int]]:
        return {aid: (info.host, info.port) for aid, info in self.peers.items()}