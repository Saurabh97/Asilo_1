#!/usr/bin/env python3
"""
Plot per-subject model comparison: ASILO vs FedAvg vs DFedSAM vs DeceFL.

It searches your result CSVs by glob patterns, loads them, normalizes
column names (round/F1), and produces:
  - one PNG per subject with all 4 models overlaid (F1 vs Rounds)
  - an optional combined grid (if --grid is passed)

Assumptions (but configurable via --pattern):
  outputs/asilo/*S??*.csv
  outputs/fedavg/*S??*.csv
  outputs/dfedsam/*S??*.csv
  outputs/decefl/*S??*.csv

Each CSV should have, at minimum:
  - a round-like column: one of ['round','r','t','epoch']
  - an F1-like column: one of ['f1','F1','f1_val','f1_score']

If a 'subject' column exists in the CSV, it is used; otherwise the script
extracts Sxx from the filename (e.g., ".../S02_metrics.csv" -> S02).

Usage examples:
  python plot_subject_compare.py --root outputs --grid
  python plot_subject_compare.py --pattern ASILO="logs/asilo/*_S??.csv" --pattern FedAvg="logs/fedavg/*.csv"

"""

import argparse, re, os, sys, glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# ----- helpers -----

ROUND_CANDIDATES = ["round", "r", "t", "epoch"]
F1_CANDIDATES    = ["f1", "F1", "f1_val", "f1_score", "val_f1", "macro_f1"]

SUBJECT_RE = re.compile(r"(S\d{2})", re.IGNORECASE)

DEFAULT_PATTERNS = {
    "ASILO":   "outputs/asilo/**/*S??*.csv",
    "FedAvg":  "outputs/fedavg/**/*S??*.csv",
    "DFedSAM": "outputs/dfedsam/**/*S??*.csv",
    "DeceFL":  "outputs/decefl/**/*S??*.csv",
}

MODEL_COLORS = {
    "ASILO": "#1f77b4",   # blue
    "FedAvg": "#ff7f0e",  # orange
    "DFedSAM": "#2ca02c", # green
    "DeceFL": "#d62728",  # red
}

def find_first_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
        # try case-insensitive
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    raise KeyError(f"None of the columns {candidates} found in {list(df.columns)}")

def infer_subject_from_path(path: str) -> str:
    m = SUBJECT_RE.search(os.path.basename(path))
    if m:
        return m.group(1).upper()
    # try parent dir
    m = SUBJECT_RE.search(os.path.dirname(path))
    if m:
        return m.group(1).upper()
    return "UNK"

def load_frames_for_model(pattern: str, model_name: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern, recursive=True))
    rows = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            # try TSV
            try:
                df = pd.read_csv(fp, sep="\t")
            except Exception:
                continue

        # pick columns
        try:
            round_col = find_first_col(df, ROUND_CANDIDATES)
            f1_col    = find_first_col(df, F1_CANDIDATES)
        except KeyError:
            continue

        # subject column or from path
        if "subject" in df.columns:
            subj = str(df["subject"].iloc[0]).upper()
        else:
            subj = infer_subject_from_path(fp)

        tmp = df[[round_col, f1_col]].copy()
        tmp.rename(columns={round_col: "round", f1_col: "f1"}, inplace=True)
        tmp["subject"] = subj
        tmp["model"]   = model_name
        # drop NaNs & keep monotonic rounds
        tmp = tmp.dropna(subset=["round","f1"])
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["round","f1","subject","model"])
    out = pd.concat(rows, ignore_index=True)
    # enforce numeric round/f1
    out["round"] = pd.to_numeric(out["round"], errors="coerce")
    out["f1"]    = pd.to_numeric(out["f1"], errors="coerce")
    out = out.dropna(subset=["round","f1"])
    return out

def assemble_dataset(patterns: Dict[str,str]) -> pd.DataFrame:
    frames = []
    for model, pat in patterns.items():
        df = load_frames_for_model(pat, model)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["round","f1","subject","model"])
    all_df = pd.concat(frames, ignore_index=True)
    # Keep only known models & nice order
    all_df["model"] = pd.Categorical(
        all_df["model"],
        categories=["ASILO","FedAvg","DFedSAM","DeceFL"],
        ordered=True
    )
    return all_df.sort_values(["subject","model","round"])

def plot_subject_panels(df: pd.DataFrame, outdir: str, grid: bool=False, smooth:int=1):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    subjects = sorted(df["subject"].dropna().unique())

    def maybe_smooth(x: pd.Series, w:int) -> pd.Series:
        if w <= 1: return x
        return x.rolling(window=w, min_periods=1, center=False).mean()

    # One PNG per subject
    for subj in subjects:
        dsub = df[df["subject"] == subj]
        if dsub.empty: 
            continue
        plt.figure(figsize=(8,5))
        for model in ["ASILO","FedAvg","DFedSAM","DeceFL"]:
            dsm = dsub[dsub["model"] == model]
            if dsm.empty: 
                continue
            rr = dsm["round"].values
            f1 = maybe_smooth(dsm["f1"], smooth).values
            plt.plot(rr, f1, label=model, linewidth=2.0, color=MODEL_COLORS.get(model, None))
        plt.title(f"{subj}: F1 vs Rounds")
        plt.xlabel("Rounds"); plt.ylabel("F1")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{subj}_compare.png"), dpi=140)
        plt.close()

    # Optional: combined grid of all subjects
    if grid and len(subjects) > 0:
        n = len(subjects)
        cols = 3
        rows = (n + cols - 1)//cols
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows), squeeze=False)
        axit = iter(axes.flatten())
        for subj in subjects:
            ax = next(axit, None)
            if ax is None: break
            dsub = df[df["subject"] == subj]
            for model in ["ASILO","FedAvg","DFedSAM","DeceFL"]:
                dsm = dsub[dsub["model"] == model]
                if dsm.empty: 
                    continue
                rr = dsm["round"].values
                f1 = maybe_smooth(dsm["f1"], smooth).values
                ax.plot(rr, f1, label=model, linewidth=2.0, color=MODEL_COLORS.get(model, None))
            ax.set_title(subj)
            ax.set_xlabel("Rounds"); ax.set_ylabel("F1")
            ax.grid(True, alpha=0.25)
        # tidy legends: one global legend
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        plt.tight_layout(rect=[0,0.04,1,1])
        plt.savefig(os.path.join(outdir, f"ALL_subjects_compare_grid.png"), dpi=150)
        plt.close()


def parse_kv_pairs(pairs: List[str]) -> Dict[str,str]:
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--pattern must be MODEL=GLOB, got {p}")
        k,v = p.split("=",1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="If given, build default patterns under this root (e.g., --root outputs)")
    ap.add_argument("--pattern", action="append", default=[],
                    help='Override/add a search pattern: MODEL=GLOB (e.g., ASILO="logs/asilo/*.csv")')
    ap.add_argument("--outdir", default="plots/subject_compare",
                    help="Output directory for PNGs")
    ap.add_argument("--grid", action="store_true",
                    help="Also make one combined grid figure for all subjects")
    ap.add_argument("--smooth", type=int, default=1,
                    help="Rolling window to smooth F1 curves (>=1; 1 = no smoothing)")
    args = ap.parse_args()

    patterns = dict(DEFAULT_PATTERNS)
    if args.root:
        # rebuild defaults under a custom root
        patterns = {
            "ASILO":   os.path.join(args.root, "asilo",   "**", "*S??*.csv"),
            "FedAvg":  os.path.join(args.root, "fedavg",  "**", "*S??*.csv"),
            "DFedSAM": os.path.join(args.root, "dfedsam", "**", "*S??*.csv"),
            "DeceFL":  os.path.join(args.root, "decefl",  "**", "*S??*.csv"),
        }
    if args.pattern:
        patterns.update(parse_kv_pairs(args.pattern))

    df = assemble_dataset(patterns)
    if df.empty:
        print("No data found. Check your --pattern globs or file columns.")
        sys.exit(1)

    plot_subject_panels(df, args.outdir, grid=args.grid, smooth=args.smooth)
    print(f"Done. PNGs saved under: {args.outdir}")

if __name__ == "__main__":
    main()
