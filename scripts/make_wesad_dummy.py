#!/usr/bin/env python3
"""
Generate dummy WESAD wrist CSVs per subject so the wearables demo can run end-to-end.

Schema per CSV:
  ts,label,feat_0,feat_1,...,feat_{D-1}
Where:
  - ts    : integer timestamp (window index)
  - label : 0=baseline/amusement, 1=stress
  - feat_*: synthetic features (floats)

Non-IID flavor:
  - Each subject has a unique feature shift (mean vector).
  - Class-1 (stress) has a global shift to make the task learnable.
  - Optional per-subject positive rate variation.

Usage example (PowerShell / cmd):
  python Asilo_1\\scripts\\make_wesad_dummy.py ^
      --out Asilo_1\\data\\processed\\WESAD_wrist ^
      --subjects S02 S05 S07 S08 S11 S14 ^
      --n 3000 --features 16 --pos-rate 0.30 --seed 1337
"""

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd


def subject_seed(global_seed: int, subject: str) -> int:
    h = hashlib.sha1((subject + str(global_seed)).encode()).hexdigest()
    # Take last 8 hex chars to form a stable 32-bit seed
    return int(h[-8:], 16)


def make_subject_df(
    subject: str,
    n_samples: int,
    d_features: int,
    base_pos_rate: float,
    global_seed: int,
) -> pd.DataFrame:
    """
    Build a synthetic per-subject dataset with:
      - subject-specific mean shift (non-IID),
      - separable classes via a global stress shift,
      - slight per-subject variation in positive rate.
    """
    rng = np.random.default_rng(subject_seed(global_seed, subject))

    # Per-subject non-IID mean shift
    subj_shift = rng.normal(loc=0.0, scale=0.8, size=d_features)

    # Class-1 (stress) global shift (e.g., higher arousal features)
    stress_shift = np.concatenate(
        [
            np.full(d_features // 3, 0.8),   # some features trend up
            np.full(d_features // 3, 0.2),   # some mildly up
            np.full(d_features - 2 * (d_features // 3), -0.1),  # some down/noisy
        ]
    )

    # Slight per-subject variation in class balance
    pos_rate = float(
        np.clip(
            rng.normal(loc=base_pos_rate, scale=0.05),
            0.05,
            0.95,
        )
    )

    # Labels
    y = (rng.random(n_samples) < pos_rate).astype(int)

    # Base features
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, d_features))

    # Apply subject shift to everyone
    X = X + subj_shift

    # Apply stress shift to class-1 windows
    X[y == 1] = X[y == 1] + stress_shift

    # Small correlated noise to simulate sensor coupling
    cov = 0.1 * np.ones((d_features, d_features)) + 0.9 * np.eye(d_features)
    L = np.linalg.cholesky(cov)
    X = (X @ L)

    # Shuffle rows with a stable permutation
    perm = rng.permutation(n_samples)
    X, y = X[perm], y[perm]

    # Build DataFrame (simple integer window index as ts)
    df = pd.DataFrame(
        X,
        columns=[f"feat_{i}" for i in range(d_features)],
    )
    df.insert(0, "label", y.astype(int))
    df.insert(0, "ts", np.arange(n_samples, dtype=int))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=str,
        default=str(Path("Asilo_1") / "data" / "processed" / "WESAD_wrist"),
        help="Output directory for per-subject CSVs",
    )
    ap.add_argument(
        "--subjects",
        nargs="+",
        default=["S02", "S05", "S07", "S08", "S11", "S14"],
        help="Subject IDs to generate",
    )
    ap.add_argument("--n", type=int, default=3000, help="Rows per subject CSV")
    ap.add_argument("--features", type=int, default=16, help="Number of features")
    ap.add_argument(
        "--pos-rate",
        type=float,
        default=0.30,
        help="Baseline positive (stress) rate per subject (will vary ±0.05)",
    )
    ap.add_argument("--seed", type=int, default=1337, help="Global seed")
    args = ap.parse_args()

    out_dir = Path(args.out)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[make_wesad_dummy] Writing CSVs to: {out_dir.resolve()}")
    print(
        f"[make_wesad_dummy] subjects={args.subjects}  n={args.n}  d={args.features}  base_pos_rate={args.pos_rate}"
    )

    for sid in args.subjects:
        df = make_subject_df(
            subject=sid,
            n_samples=args.n,
            d_features=args.features,
            base_pos_rate=args.pos_rate,
            global_seed=args.seed,
        )
        out_path = out_dir / f"{sid}.csv"
        df.to_csv(out_path, index=False)
        # quick stats
        pos = int(df["label"].sum())
        print(f"  - {sid}: {len(df)} rows  pos={pos} ({pos/len(df):.1%})  -> {out_path.name}")

    print("[make_wesad_dummy] Done.")


if __name__ == "__main__":
    main()
