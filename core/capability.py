# Asilo_1/core/capability.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class CapabilityProfile:
    name: str
    width: float        # 0.25/0.5/0.75/1.0
    local_batches: int  # steps per round
    k_peers: int
    max_bytes_round: int

FAST = CapabilityProfile("FAST", width=1.0,  local_batches=8,  k_peers=3, max_bytes_round=200_000)
MID  = CapabilityProfile("MID",  width=0.75, local_batches=6,  k_peers=3, max_bytes_round=120_000)
SLOW = CapabilityProfile("SLOW", width=0.5,  local_batches=4,  k_peers=2, max_bytes_round=60_000)
