import os, glob
import pandas as pd
import matplotlib.pyplot as plt

LOG_DIRS = {
    "ASILO": "logs/asilo",
    "FedAvg": "logs/fedavg",
    "DFedSAM": "logs/dfedsam",
    "DeceFL": "logs/decefl"
}
colors = {"ASILO":"blue", "FedAvg":"orange", "DFedSAM":"green", "DeceFL":"red"}

out_dir = "artifacts"
os.makedirs(out_dir, exist_ok=True)

def load_logs(log_dir):
    rows = []
    for path in glob.glob(os.path.join(log_dir, "*.csv")):
        try:
            df = pd.read_csv(path)
            df["agent"] = os.path.splitext(os.path.basename(path))[0]
            rows.append(df)
        except Exception:
            pass
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def smooth(series, w=15):
    return series.rolling(window=w, min_periods=1).mean()

def plot_metric(metric_name, ylabel, filename):
    plt.figure(figsize=(18,5))

    # 1) Metric vs Rounds
    plt.subplot(1,3,1)
    for algo, path in LOG_DIRS.items():
        df = load_logs(path)
        if df.empty: continue
        t_col = pick_col(df, ["t","round","r"])
        m_col = pick_col(df, [metric_name])
        if t_col is None or m_col is None: continue
        df_group = df.groupby(t_col)[m_col].agg("mean")
        plt.plot(df_group.index, smooth(df_group), color=colors[algo], label=algo, linewidth=2.5)
    plt.xlabel("Rounds"); plt.ylabel(ylabel); plt.title(f"{ylabel} vs Training Rounds")

    # 2) Metric vs Normalized Communication
    plt.subplot(1,3,2)
    for algo, path in LOG_DIRS.items():
        df = load_logs(path)
        if df.empty: continue
        bytes_col = pick_col(df, ["bytes_sent","bytes"])
        m_col = pick_col(df, [metric_name])
        if bytes_col is None or m_col is None: continue
        df["cum_bytes"] = df.groupby("agent")[bytes_col].cumsum()
        df["cum_norm"] = df["cum_bytes"] / df["cum_bytes"].max()
        df_group = df.groupby("cum_norm")[m_col].mean()
        plt.plot(df_group.index, smooth(df_group), color=colors[algo], label=algo, linewidth=2.5, marker="o", markevery=20)
    plt.xlabel("Normalized Communication"); plt.ylabel(ylabel); plt.title(f"{ylabel} vs Communication Cost")

    # 3) Pheromone only if ASILO and metric == "f1_val"
    plt.subplot(1,3,3)
    if metric_name == "f1_val":
        df = load_logs(LOG_DIRS["ASILO"])
        if not df.empty:
            t_col = pick_col(df, ["t","round","r"])
            phero_col = pick_col(df, ["pheromone","p_local"])
            if t_col and phero_col:
                df_group = df.groupby(t_col)[phero_col].agg(["mean","std"])
                plt.plot(df_group.index, smooth(df_group["mean"]), color="blue", label="ASILO", linewidth=2.5)
                plt.fill_between(df_group.index,
                                 smooth(df_group["mean"]-df_group["std"]),
                                 smooth(df_group["mean"]+df_group["std"]),
                                 color="blue", alpha=0.2)
        plt.xlabel("Rounds"); plt.ylabel("Pheromone"); plt.title("Pheromone Dynamics (ASILO)")
    else:
        plt.axis("off")  # empty for AUPRC

    for j in range(1,3): plt.subplot(1,3,j).legend(loc="lower right")
    plt.tight_layout()
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path)
    print(f"[done] wrote {out_path}")

if __name__ == "__main__":
    plot_metric("f1_val", "F1", "comparison_f1.png")
    plot_metric("auprc", "AUPRC", "comparison_auprc.png")
