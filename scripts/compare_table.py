#!/usr/bin/env python3
"""
Summarize per-subject F1 performance comparison table:
ASILO vs FedAvg vs DFedSAM vs DeceFL

It scans the same CSV inputs as your plotting script and outputs:
  - a CSV table with mean and max F1 per subject per model
  - a formatted Markdown/console table for quick comparison
"""

import pandas as pd
import argparse, os, sys, glob
from plot_subject_compare import assemble_dataset, DEFAULT_PATTERNS, parse_kv_pairs

def summarize_table(df: pd.DataFrame, out_csv: str):
    # Compute mean and max F1 per subject per model
    summary = (
        df.groupby(["subject", "model"])["f1"]
        .agg(["mean", "max"])
        .reset_index()
        .pivot(index="subject", columns="model", values=["mean", "max"])
    )

    # Flatten multi-index columns
    summary.columns = [f"{stat}_{model}" for stat, model in summary.columns]
    summary = summary.reset_index()

    # Compute overall averages (across subjects)
    avg_row = summary.mean(numeric_only=True)
    avg_row["subject"] = "Average"
    summary = pd.concat([summary, pd.DataFrame([avg_row])], ignore_index=True)

    # Save CSV
    summary.to_csv(out_csv, index=False)
    print(f"\n✅ Comparison table saved to: {out_csv}\n")

    # Print clean summary to console
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(summary.round(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="Root directory for model folders (same as plot_subject_compare.py)")
    ap.add_argument("--pattern", action="append", default=[],
                    help='Override/add a search pattern: MODEL=GLOB (e.g., ASILO="logs/asilo/*.csv")')
    ap.add_argument("--out", default="comparison_summary.csv",
                    help="Output CSV file path")
    args = ap.parse_args()

    # Build patterns
    patterns = dict(DEFAULT_PATTERNS)
    if args.root:
        patterns = {
            "ASILO":   os.path.join(args.root, "asilo",   "**", "*S??*.csv"),
            "FedAvg":  os.path.join(args.root, "fedavg",  "**", "*S??*.csv"),
            "DFedSAM": os.path.join(args.root, "dfedsam", "**", "*S??*.csv"),
            "DeceFL":  os.path.join(args.root, "decefl",  "**", "*S??*.csv"),
        }
    if args.pattern:
        patterns.update(parse_kv_pairs(args.pattern))

    # Load all data
    df = assemble_dataset(patterns)
    if df.empty:
        print("❌ No data found. Please check your paths or patterns.")
        sys.exit(1)

    summarize_table(df, args.out)

if __name__ == "__main__":
    main()
