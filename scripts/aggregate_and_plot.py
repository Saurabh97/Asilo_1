#!/usr/bin/env python3
import glob, os
import pandas as pd
import matplotlib.pyplot as plt

LOG_DIR = os.path.join("logs")
out_dir = os.path.join("artifacts"); os.makedirs(out_dir, exist_ok=True)

def load_logs():
    rows = []
    for path in glob.glob(os.path.join(LOG_DIR, "*.csv")):
        agent = os.path.splitext(os.path.basename(path))[0]
        try:
            df = pd.read_csv(path)
            df["agent"] = agent
            rows.append(df)
        except Exception:
            pass
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def pick_col(df, candidates):
    """Return the first column name that exists in df from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    df = load_logs()
    if df.empty:
        print("[aggregate] no logs found in 'logs/'. run the experiment first.")
        return

    # ---- column detection / compatibility ----
    t_col       = pick_col(df, ["t", "round", "r", "step"])
    f1_col      = pick_col(df, ["f1_val", "f1", "val_f1", "f1_valid"])
    bytes_col   = pick_col(df, ["bytes_sent", "bytes", "tx_bytes"])
    phero_col   = pick_col(df, ["pheromone", "p_local", "p", "tau"])

    if t_col is None or f1_col is None:
        raise ValueError(f"Missing time or F1 column (have columns: {list(df.columns)})")

    # ensure sorted & last value per round if duplicates
    df = df.sort_values([ "agent", t_col ])
    df = df.drop_duplicates(subset=["agent", t_col], keep="last")

    # ---------------- 1) F1 vs rounds ----------------
    plt.figure()
    for agent, dfa in df.groupby("agent"):
        plt.plot(dfa[t_col], dfa[f1_col], label=agent)
    plt.xlabel("round")
    plt.ylabel("F1 (val)")
    plt.title("Per-agent F1 vs rounds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "f1_vs_rounds.png"))

    # ---------------- 2) Accuracy vs communication (cumulative) ----------------
    if bytes_col is not None:
        plt.figure()
        for agent, dfa in df.groupby("agent"):
            # use cumulative bytes to avoid the saw-tooth
            b = dfa[bytes_col].fillna(0).clip(lower=0)
            b_cum = b.cumsum()
            plt.plot(b_cum, dfa[f1_col], label=agent)
        plt.xlabel("cumulative bytes sent")
        plt.ylabel("F1 (val)")
        plt.title("Accuracy vs communication")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "f1_vs_bytes.png"))
    else:
        print("[aggregate] bytes column not found; skipped comms plot.")

    # ---------------- 3) Pheromone over time ----------------
    if phero_col is not None:
        plt.figure()
        for agent, dfa in df.groupby("agent"):
            plt.plot(dfa[t_col], dfa[phero_col], label=agent)
        plt.xlabel("round")
        plt.ylabel(phero_col)
        plt.title("Pheromone over time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "pheromone_vs_rounds.png"))
    else:
        print("[aggregate] pheromone column not found; skipped pheromone plot.")

    print("[aggregate] wrote:",
    "artifacts/f1_vs_rounds.png",
    "artifacts/f1_vs_bytes.png" if bytes_col else "(no bytes plot)",
    "artifacts/pheromone_vs_rounds.png" if phero_col else "(no pheromone plot)")

if __name__ == "__main__":
    main()
