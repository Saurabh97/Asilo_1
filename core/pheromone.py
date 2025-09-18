from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PheromoneConfig:
    alpha_decay: float = 0.05 # evaporation per round
    beta_gain: float = 1.0 # scale on utility
    share_threshold: float = 0.35
    evaporation_floor: float = 0.0


class PheromoneManager:
    def __init__(self, cfg: PheromoneConfig):
        self.cfg = cfg
        self.p_value: float = 0.0


    def update_with_utility(self, u: float) -> float:
        u = max(0.0, min(1.0, float(u)))
        self.p_value = (1.0 - self.cfg.alpha_decay) * self.p_value + self.cfg.beta_gain * u
        if self.cfg.evaporation_floor is not None:
            self.p_value = max(self.cfg.evaporation_floor, self.p_value)
        return self.p_value 


    def evaporate(self) -> float:
        self.p_value = (1.0 - self.cfg.alpha_decay) * self.p_value
        if self.cfg.evaporation_floor is not None:
            self.p_value = max(self.cfg.evaporation_floor, self.p_value)
        return self.p_value


    def should_share(self) -> bool:
        return self.p_value >= self.cfg.share_threshold
