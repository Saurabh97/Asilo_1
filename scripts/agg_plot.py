import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOG_DIRS = {
    "ASILO":  "logs/asilo",
    "FedAvg": "logs/fedavg",
    "DFedSAM":"logs/dfedsam",
    "DeceFL": "logs/decefl",
}
colors = {"ASILO":"blue", "FedAvg":"orange", "DFedSAM":"green", "DeceFL":"red"}

out_dir = "artifacts"
os.makedirs(out_dir, exist_ok=True)

# ---------- io helpers ----------
def load_logs(log_dir: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(log_dir, "*.csv")):
        try:
            df = pd.read_csv(path)
            df["agent"] = os.path.splitext(os.path.basename(path))[0]
            rows.append(df)
        except Exception:
            pass
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def smooth(series: pd.Series, w=15):
    return series.rolling(window=w, min_periods=1).mean()

# ---------- plotting ----------
def plot_metric(metric_name, ylabel, filename, bins=200, smooth_w=15):
    # preload all logs once
    dfs = {algo: load_logs(path) for algo, path in LOG_DIRS.items()}
    # drop empties
    dfs = {k:v for k,v in dfs.items() if not v.empty}

    plt.figure(figsize=(18,5))

    # 1) Metric vs Rounds (macro over agents)
    plt.subplot(1,3,1)
    for algo, df in dfs.items():
        t_col = pick_col(df, ["t","round","r"])
        m_col = pick_col(df, [metric_name])
        if t_col is None or m_col is None: 
            continue
        df_group = df.groupby(t_col, as_index=True)[m_col].mean().sort_index()
        plt.plot(df_group.index, smooth(df_group, smooth_w), color=colors[algo], label=algo, linewidth=2.5)
    plt.xlabel("Rounds"); plt.ylabel(ylabel); plt.title(f"{ylabel} vs Training Rounds")

    # 2) Metric vs Normalized Communication (global normalization + uniform bins)
    plt.subplot(1,3,2)

    # find a single global max bytes across all algos & agents
    global_max_bytes = 0.0
    for df in dfs.values():
        bytes_col = pick_col(df, ["bytes_sent","bytes"])
        if bytes_col is None: 
            continue
        tmp = df.copy()
        tmp["cum_bytes"] = tmp.groupby("agent")[bytes_col].cumsum()
        # last value per agent
        last_per_agent = tmp.groupby("agent", as_index=False)["cum_bytes"].last()["cum_bytes"].max()
        global_max_bytes = max(global_max_bytes, float(last_per_agent) if pd.notna(last_per_agent) else 0.0)
    if global_max_bytes <= 0:
        global_max_bytes = 1.0  # avoid div by zero; will plot at x≈0

    # plot each algo against the same x normalization
    for algo, df in dfs.items():
        bytes_col = pick_col(df, ["bytes_sent","bytes"])
        m_col     = pick_col(df, [metric_name])
        t_col     = pick_col(df, ["t","round","r"])
        if bytes_col is None or m_col is None or t_col is None:
            continue

        tmp = df[[t_col, "agent", bytes_col, m_col]].copy()
        tmp["cum_bytes"] = tmp.groupby("agent")[bytes_col].cumsum()
        tmp["x"] = tmp["cum_bytes"] / (global_max_bytes + 1e-9)
        tmp = tmp.sort_values("x")

        # bin to uniform x grid to avoid noisy aggregation by float keys
        grid = np.linspace(0.0, 1.0, bins)
        # for each bin, take mean metric of rows that fall into the bin
        inds = np.digitize(tmp["x"].to_numpy(), grid, right=True)
        yb = pd.Series(index=range(len(grid)), dtype=float)
        for i in range(len(grid)):
            sel = tmp.loc[inds == i, m_col]
            if len(sel) > 0:
                yb.iloc[i] = sel.mean()

        # drop empty leading/trailing NaNs and smooth
        yb = yb.interpolate(limit_direction="both")
        plt.plot(grid, smooth(yb, max(3, smooth_w//2)), color=colors[algo], label=algo, linewidth=2.5)

    plt.xlabel("Normalized Communication") 
    plt.ylabel(ylabel) 
    plt.title(f"{ylabel} vs Communication Cost")

    # 3) Pheromone (ASILO only)
    plt.subplot(1,3,3)
    if metric_name == "f1_val" and "ASILO" in dfs:
        df = dfs["ASILO"]
        t_col = pick_col(df, ["t","round","r"])
        phero_col = pick_col(df, ["pheromone","p_local"])
        if t_col and phero_col:
            grp = df.groupby(t_col)[phero_col].agg(["mean","std"]).sort_index()
            mu, sd = grp["mean"], grp["std"]
            lo = (mu - sd).clip(lower=0)   # pheromone is >= 0; clip for display
            hi = (mu + sd)
            plt.fill_between(grp.index, smooth(lo, smooth_w), smooth(hi, smooth_w), color="blue", alpha=0.2)
            plt.plot(grp.index, smooth(mu, smooth_w), color="blue", label="ASILO", linewidth=2.5)
        plt.xlabel("Rounds"); plt.ylabel("Pheromone"); plt.title("Pheromone Dynamics (ASILO)")
    else:
        plt.axis("off")

    for j in range(1,3):
        plt.subplot(1,3,j).legend(loc="lower right")
    plt.tight_layout()
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[done] wrote {out_path}")

if __name__ == "__main__":
    plot_metric("f1_val", "F1", "comparison_f1.png")
    plot_metric("auprc",  "AUPRC", "comparison_auprc.png")
