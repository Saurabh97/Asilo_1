# asilo/core/monitor.py
import csv, os, time
class CSVMonitor:
    def __init__(self, agent_id: str, outdir: str = "logs"):
        os.makedirs(outdir, exist_ok=True)
        self.path = os.path.join(outdir, f"{agent_id}.csv")
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(["t","bytes_sent","utility","pheromone","f1_val"])
    def log(self, t, bytes_sent, utility, pheromone, f1_val):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([t, bytes_sent, utility, pheromone, f1_val])
