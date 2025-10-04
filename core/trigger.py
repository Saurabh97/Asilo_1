# Asilo_1/core/trigger.py
from __future__ import annotations
from dataclasses import dataclass
import time

@dataclass
class TriggerConfig:
    theta_high: float = 0.7
    theta_low: float = 0.5
    cooldown_s: float = 10.0
    bucket_capacity: int = 400_000   # bytes
    bucket_refill_per_s: int = 40_000

class TokenBucket:
    def __init__(self, capacity: int, refill_per_s: int):
        self.capacity = int(capacity)
        self.tokens = int(capacity)
        self.refill_per_s = int(refill_per_s)
        self._t = time.time()
    def _refill(self):
        now = time.time()
        dt = now - self._t
        self._t = now
        self.tokens = min(self.capacity, self.tokens + int(dt * self.refill_per_s))
    def try_consume(self, n: int) -> bool:
        self._refill()
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

class Trigger:
    def __init__(self, cfg: TriggerConfig):
        self.cfg = cfg
        self.last_send = 0.0
        self.active = False
        self.bucket = TokenBucket(cfg.bucket_capacity, cfg.bucket_refill_per_s)
        self.cooldown_s = cfg.cooldown_s
    def should_send(self, psi: float) -> bool:
        now = time.time()
        # hysteresis
        if self.active:
            if psi < self.cfg.theta_low:
                self.active = False
        else:
            if psi >= self.cfg.theta_high:
                self.active = True
        # cooldown + tokens
        cooldown_ok = (now - self.last_send) >= self.cfg.cooldown_s
        return self.active and cooldown_ok and self.bucket.tokens > 0
    def on_send(self, bytes_sent: int):
        self.last_send = time.time()
        self.bucket.try_consume(int(bytes_sent))
