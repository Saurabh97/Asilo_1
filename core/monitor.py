# asilo/core/monitor.py
import csv, os, time
class CSVMonitor:
    def __init__(self, agent_id, exp_name="default"):
        log_dir = os.path.join("logs", exp_name)
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"{agent_id}.csv")
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(["t","bytes_sent","utility","pheromone","f1_val","agg_count", "rollback"])
    def log(self, t, bytes_sent, utility, pheromone, f1_val,agg_count: int = 0, rollback: int = 0):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([t, bytes_sent, utility, pheromone, f1_val, agg_count, rollback])
