from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class LocalTrainer(ABC):
    @abstractmethod
    def fit_local(self, batches: int) -> Dict[str, float]:
        """Run a chunk of local training and return metrics (e.g., loss)."""


    @abstractmethod
    def eval(self) -> Dict[str, float]:
        """Return validation metrics including 'f1_val' if possible."""


    @abstractmethod
    def compute_utility(self, prev_metrics: Dict[str, float], curr_metrics: Dict[str, float]) -> float:
        """Map Δmetrics to a bounded utility in [0, 1]."""


    @abstractmethod
    def make_delta(self, strategy: str) -> Tuple[Dict[str, Any], int]:
        """Return (payload, bytes_size)."""


    @abstractmethod
    def apply_delta(self, payload: Dict[str, Any], strategy: str) -> None:
        """Apply received delta to local model."""