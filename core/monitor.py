# asilo/core/monitor.py
import csv, os, time
from typing import Optional

class CSVMonitor:
    """
    Lightweight CSV logger with multiple streams:
      - <agent>.csv          → per-agent round metrics (existing)
      - edges.csv            → per-edge communication events (global across agents)
      - pheromone.csv        → per-round pheromone snapshot (agent's view of each neighbor)
      - agg_decisions.csv    → aggregation keep/drop decisions per received delta
    All files live under logs/<exp_name>/.
    """
    def __init__(self, agent_id: str, exp_name: str = "default"):
        self.agent_id = agent_id
        self.log_dir = os.path.join("logs", exp_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # --- per-agent metrics (unchanged schema for backward compatibility) ---
        self.path = os.path.join(self.log_dir, f"{agent_id}.csv")
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "t","bytes_sent","utility","pheromone","f1_val","agg_count","rollback"
                ])

        # --- global edges log ---
        self.edges_path = os.path.join(self.log_dir, "edges.csv")
        if not os.path.exists(self.edges_path):
            with open(self.edges_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    # one row per event
                    "t","src","dst","bytes","kind","action","cos","reason"
                ])

        # --- pheromone snapshot log ---
        self.phero_path = os.path.join(self.log_dir, "pheromone.csv")
        if not os.path.exists(self.phero_path):
            with open(self.phero_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    # one row per (agent, neighbor) per round
                    "t","agent","neighbor","tau","score","contacts","reputation","self_p"
                ])

        # --- aggregation decision log ---
        self.agg_path = os.path.join(self.log_dir, "agg_decisions.csv")
        if not os.path.exists(self.agg_path):
            with open(self.agg_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    # one row per received delta classified at aggregation time
                    "t","agent","from","bytes","decision","cos_thresh","mode","trim_p","rolled_back"
                ])

    # ===== existing per-round metrics =====
    def log(self, t, bytes_sent, utility, pheromone, f1_val, agg_count: int = 0, rollback: int = 0):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([t, bytes_sent, utility, pheromone, f1_val, agg_count, rollback])

    # ===== new: per-edge comm events =====
    def log_edge(self, t: int, src: str, dst: str, bytes_sz: int, kind: str,
                 action: str, cos: Optional[float] = None, reason: Optional[str] = None):
        try:
            with open(self.edges_path, "a", newline="") as f:
                csv.writer(f).writerow([t, src, dst, int(bytes_sz or 0), kind, action,
                                        ("" if cos is None else float(cos)),
                                        ("" if reason is None else str(reason))])
        except Exception:
            # never crash caller due to telemetry
            pass

    # ===== new: pheromone snapshot =====
    def log_pheromone(self, t: int, agent: str, neighbor: str,
                      tau: float, score: float, contacts: int, reputation: float, self_p: float):
        try:
            with open(self.phero_path, "a", newline="") as f:
                csv.writer(f).writerow([t, agent, neighbor,
                                        float(tau), float(score), int(contacts), float(reputation), float(self_p)])
        except Exception:
            pass

    # ===== new: aggregation keep/drop classification =====
    def log_agg_decision(self, t: int, agent: str, from_id: str, bytes_sz: int,
                         decision: str, cos_thresh: float, mode: str, trim_p: float, rolled_back: int):
        try:
            with open(self.agg_path, "a", newline="") as f:
                csv.writer(f).writerow([t, agent, from_id, int(bytes_sz or 0),
                                        decision, float(cos_thresh), mode, float(trim_p), int(rolled_back or 0)])
        except Exception:
            pass
