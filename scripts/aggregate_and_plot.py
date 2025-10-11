#!/usr/bin/env python3
import glob, os
import pandas as pd
import matplotlib.pyplot as plt

LOG_DIR = "logs/asilo"
OUT_DIR = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

ROLL = 10  # smoothing window for rolling average

def load_logs():
    rows = []
    for path in glob.glob(os.path.join(LOG_DIR, "*.csv")):
        agent = os.path.splitext(os.path.basename(path))[0]
        try:
            df = pd.read_csv(path)
            df["agent"] = agent
            rows.append(df)
        except Exception as e:
            print(f"[warn] failed to read {path}: {e}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def pick_col(df, candidates):
    """Return the first matching column name from candidates, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def smooth(series):
    """Apply rolling average smoothing for cleaner plots."""
    return series.rolling(ROLL, min_periods=1).mean()

def main():
    df = load_logs()
    if df.empty:
        print("[aggregate] no logs found in 'logs/'. Run the experiment first.")
        return

    # ---- detect column names ----
    t_col     = pick_col(df, ["t", "round", "r", "step"])
    f1_col    = pick_col(df, ["f1_val", "f1", "val_f1", "f1_valid"])
    bytes_col = pick_col(df, ["bytes_sent", "bytes", "tx_bytes"])
    phero_col = pick_col(df, ["pheromone", "p_local", "p", "tau"])

    if t_col is None or f1_col is None:
        raise ValueError(f"Missing time or F1 column (have: {list(df.columns)})")

    # ensure sorted & drop duplicates
    df = df.sort_values(["agent", t_col])
    df = df.drop_duplicates(subset=["agent", t_col], keep="last")

    # ---------- 1) F1 vs rounds ----------
    plt.figure(figsize=(10,6))
    for agent, dfa in df.groupby("agent"):
        plt.plot(dfa[t_col], smooth(dfa[f1_col]), label=agent)
    plt.xlabel("Round")
    plt.ylabel("Validation F1")
    plt.title("Per-Agent F1 vs Rounds (smoothed)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "f1_vs_rounds.png"))

    # ---------- 2) Accuracy vs communication ----------
    if bytes_col is not None:
        plt.figure(figsize=(10,6))
        for agent, dfa in df.groupby("agent"):
            b_cum = dfa[bytes_col].fillna(0).clip(lower=0).cumsum()
            plt.plot(smooth(b_cum), smooth(dfa[f1_col]), label=agent)
        plt.xlabel("Cumulative Bytes Sent")
        plt.ylabel("Validation F1")
        plt.title("Accuracy vs Communication (smoothed)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "f1_vs_bytes.png"))
    else:
        print("[aggregate] bytes column not found; skipped comms plot.")

    # ---------- 3) Pheromone vs rounds ----------
    if phero_col is not None:
        plt.figure(figsize=(10,6))
        for agent, dfa in df.groupby("agent"):
            plt.plot(dfa[t_col], smooth(dfa[phero_col]), label=agent)
        plt.xlabel("Round")
        plt.ylabel("Pheromone Level")
        plt.title("Pheromone over Time (smoothed)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "pheromone_vs_rounds.png"))
    else:
        print("[aggregate] pheromone column not found; skipped pheromone plot.")

    print("[aggregate] wrote plots to", OUT_DIR)

if __name__ == "__main__":
    main()
