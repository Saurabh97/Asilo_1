from dataclasses import dataclass

@dataclass
class CapabilityProfile:
    local_batches: int
    k_peers: int
    delta: str            # 'head' or 'proto'
    max_bytes_round: int

class PolicyManager:
    def __init__(self, profiles: dict[str, CapabilityProfile], assigned: str):
        if assigned not in profiles:
            raise KeyError(f"Capability profile '{assigned}' not found")
        self.profiles = profiles
        self.assigned = assigned

    @property
    def profile(self) -> CapabilityProfile:
        return self.profiles[self.assigned]
