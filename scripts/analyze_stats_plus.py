#!/usr/bin/env python
# Extended analysis: robust log discovery + F1 stats (mean/std/variance/CI),
# F1 vs rounds, F1 vs normalized communication, pheromone decay, and per-round boxplots.
#
# Usage:
#   python analysis/analyze_stats_plus.py --repo-root <path> --run-index <run_index.json> --out <out_dir> \
#       --norm-ref FedAvg --bytes-cols bytes,bytes_sent,tx_bytes --pheromone-cols pheromone,tau,avg_tau

import os, sys, argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Robust log discovery (tolerates your naming) ----------------
# Handles:
#  - exact match logs/<exp_name>
#  - duplicate seed suffix: *_seedN_seedN
#  - DeceFL-style: *_seedN <-> *_N
#  - homogenous vs homogeneous typo
#  - fallback: best fuzzy/substring match under logs/

SEED_DUP_RE = re.compile(r"(.*?_seed)(\d+)(?:_seed\2)?$", re.IGNORECASE)

def _variants_for_name(name: str):
    vs = set()
    vs.add(name)

    # homogenous / homogeneous swap
    vs.add(name.replace("homogenous", "homogeneous"))
    vs.add(name.replace("homogeneous", "homogenous"))

    # if ends with _seedN_seedN -> also try _seedN plus DeceFL-style _N
    m = SEED_DUP_RE.match(name)
    if m:
        base, n = m.group(1), m.group(2)
        vs.add(f"{base}{n}")            # drop duplicate
        vs.add(f"{base}{n}_seed{n}")    # original duplicate (belt & suspenders)
        vs.add(f"{base[:-5]}{n}")       # drop "_seed" → "..._<n>"

    # if ends with _seedN → also try _N
    m2 = re.search(r"(.*)_seed(\d+)$", name, re.IGNORECASE)
    if m2:
        vs.add(f"{m2.group(1)}_{m2.group(2)}")

    # if ends with _N (no 'seed') → also try _seedN
    m3 = re.search(r"(.*)_(\d+)$", name)
    if m3 and not m2:
        vs.add(f"{m3.group(1)}_seed{m3.group(2)}")

    return list(vs)

def _best_fuzzy_dir(logs_root: Path, target: str) -> Path | None:
    if not logs_root.exists():
        return None
    candidates = [p for p in logs_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    # prefer startswith, then substring score
    starts = [p for p in candidates if p.name.lower().startswith(target.lower())]
    if starts:
        return sorted(starts, key=lambda p: len(p.name), reverse=True)[0]
    subs = [p for p in candidates if target.lower() in p.name.lower()]
    if subs:
        def score(p):
            s = p.name.lower()
            t = target.lower()
            lcs = 0
            for i in range(len(t)):
                for j in range(i+1, len(t)+1):
                    if t[i:j] in s:
                        lcs = max(lcs, j-i)
            return (lcs, -abs(len(s)-len(t)))
        return sorted(subs, key=score, reverse=True)[0]
    return None

def discover_log_files(repo_root: Path, exp_name: str):
    logs_root = repo_root / "logs"

    d = logs_root / exp_name
    if d.exists():
        return list(d.glob("*.csv"))

    for v in _variants_for_name(exp_name):
        dv = logs_root / v
        if dv.exists():
            return list(dv.glob("*.csv"))

    fuzzy = _best_fuzzy_dir(logs_root, exp_name)
    if fuzzy:
        return list(fuzzy.glob("*.csv"))

    return []

# --------------------- CSV parsing and aggregation ---------------------
def parse_csv_numeric(path, f1_candidates, round_candidates, bytes_candidates, pheromone_candidates):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    def pick(cands):
        for c in cands:
            if c.lower() in cols:
                return cols[c.lower()]
        return None

    round_col = pick(round_candidates) or list(df.columns)[0]
    f1_col    = pick(f1_candidates)
    bytes_col = pick(bytes_candidates)
    pher_col  = pick(pheromone_candidates)
    return df, round_col, f1_col, bytes_col, pher_col

def load_runs(run_index_json):
    with open(run_index_json, 'r', encoding='utf-8') as f:
        return json.load(f)

def aggregate_per_run(repo_root, runs, f1_candidates, round_candidates, bytes_candidates, pheromone_candidates):
    rows = []
    for r in runs:
        exp = r['exp_name']
        files = discover_log_files(repo_root, exp)
        if not files:
            print(f"[WARN] No logs for {exp}", file=sys.stderr); continue

        per_agent = []
        for fp in files:
            try:
                df, rcol, fcol, bcol, pcol = parse_csv_numeric(fp, f1_candidates, round_candidates, bytes_candidates, pheromone_candidates)
            except Exception as e:
                print(f"[WARN] Parse fail {fp}: {e}", file=sys.stderr); continue

            keep = {'round': pd.to_numeric(df[rcol], errors='coerce').astype('Int64')}
            if fcol is not None: keep['f1'] = pd.to_numeric(df[fcol], errors='coerce')
            if bcol is not None: keep['bytes'] = pd.to_numeric(df[bcol], errors='coerce')
            if pcol is not None: keep['pheromone'] = pd.to_numeric(df[pcol], errors='coerce')
            per_agent.append(pd.DataFrame(keep))

        if not per_agent:
            continue

        df_all = per_agent[0]
        for k in per_agent[1:]:
            df_all = df_all.merge(k, on='round', how='outer', suffixes=(None, None))

        f_cols = [c for c in df_all.columns if str(c).startswith('f1')]
        b_cols = [c for c in df_all.columns if str(c).startswith('bytes')]
        p_cols = [c for c in df_all.columns if str(c).startswith('pheromone')]

        df_all = df_all.sort_values('round')
        df_run = pd.DataFrame({'round': df_all['round'].astype('Int64')})
        if f_cols: df_run['f1_run'] = df_all[f_cols].mean(axis=1, skipna=True)
        if b_cols:
            df_run['bytes_round_run'] = df_all[b_cols].sum(axis=1, skipna=True)
            df_run['bytes_cum_run'] = df_run['bytes_round_run'].fillna(0).cumsum()
        if p_cols:
            df_run['pheromone_mean_run'] = df_all[p_cols].mean(axis=1, skipna=True)

        for _, row in df_run.iterrows():
            rows.append({
                'block': r.get('block'), 'label': r.get('label'), 'method': r.get('method'),
                'seed': r.get('seed'),
                'round': int(row['round']) if pd.notna(row['round']) else None,
                'f1_run': float(row['f1_run']) if 'f1_run' in df_run and pd.notna(row['f1_run']) else np.nan,
                'bytes_round_run': float(row['bytes_round_run']) if 'bytes_round_run' in df_run and pd.notna(row.get('bytes_round_run')) else np.nan,
                'bytes_cum_run': float(row['bytes_cum_run']) if 'bytes_cum_run' in df_run and pd.notna(row.get('bytes_cum_run')) else np.nan,
                'pheromone_mean_run': float(row['pheromone_mean_run']) if 'pheromone_mean_run' in df_run and pd.notna(row.get('pheromone_mean_run')) else np.nan
            })
    return pd.DataFrame(rows)

def aggregate_across_seeds(df_runs, norm_ref_method=None):
    stats = []
    if df_runs.empty:
        return pd.DataFrame()
    for (b,l,m,r), grp in df_runs.groupby(['block','label','method','round']):
        vals = grp['f1_run'].dropna().values
        n = len(vals)
        mu  = float(np.mean(vals)) if n else np.nan
        sd  = float(np.std(vals, ddof=1)) if n>1 else 0.0
        var = float(sd**2)
        sem = sd / np.sqrt(n) if n>1 else 0.0
        ci_low, ci_high = mu - 1.96*sem, mu + 1.96*sem

        comm_vals = grp['bytes_cum_run'].dropna().values
        comm_mu = float(np.mean(comm_vals)) if len(comm_vals) else np.nan

        stats.append({
            'block': b, 'label': l, 'method': m, 'round': r, 'n': n,
            'f1_mean': mu, 'f1_std': sd, 'f1_var': var, 'f1_sem': sem,
            'ci_low': ci_low, 'ci_high': ci_high,
            'bytes_cum_mean': comm_mu
        })
    df_stats = pd.DataFrame(stats)
    if norm_ref_method is not None and not df_stats.empty:
        ref = df_stats[df_stats['method']==norm_ref_method][['block','label','round','bytes_cum_mean']] \
                .rename(columns={'bytes_cum_mean':'ref_bytes'})
        df_stats = df_stats.merge(ref, on=['block','label','round'], how='left')
        df_stats['comm_norm'] = df_stats['bytes_cum_mean'] / df_stats['ref_bytes']
    return df_stats.sort_values(['block','label','method','round'])

# ----------------------------- Plot helpers -----------------------------
def plot_f1_vs_norm_comm(df_stats, out_png):
    if df_stats.empty or 'comm_norm' not in df_stats.columns or df_stats['comm_norm'].isna().all():
        print('[WARN] No normalized communication available; skip plot.')
        return
    plt.figure()
    for (b,l,m), grp in df_stats.groupby(['block','label','method']):
        grp = grp.dropna(subset=['comm_norm','f1_mean']).sort_values('round')
        if grp.empty: continue
        plt.plot(grp['comm_norm'].values, grp['f1_mean'].values, marker='o', linewidth=1, label=f'{b}-{l}-{m}')
    plt.xlabel('Normalized Communication (bytes / ref bytes)')
    plt.ylabel('F1 (mean across seeds)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def plot_pheromone_decay(df_runs, out_png):
    df_p = df_runs.dropna(subset=['pheromone_mean_run'])
    if df_p.empty:
        print('[WARN] No pheromone column found; skip pheromone plot.')
        return
    plt.figure()
    for (b,l,m), grp in df_p.groupby(['block','label','method']):
        grp = grp.groupby('round', as_index=False)['pheromone_mean_run'].mean().sort_values('round')
        plt.plot(grp['round'].values, grp['pheromone_mean_run'].values, label=f'{b}-{l}-{m}')
    plt.xlabel('Round')
    plt.ylabel('Pheromone (mean across agents & seeds)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def plot_f1_boxplots(df_runs, out_dir):
    """Per (block,label,method), draw a boxplot of per-round F1 across seeds (after per-run agent-avg)."""
    if df_runs.empty or df_runs['f1_run'].dropna().empty:
        print('[WARN] No F1 values for boxplot.')
        return
    for (b,l,m), grp in df_runs.groupby(['block','label','method']):
        grp = grp.dropna(subset=['round','f1_run'])
        if grp.empty:
            continue
        rounds_sorted = sorted({int(r) for r in grp['round'].dropna().values})
        series = []
        for rr in rounds_sorted:
            vals = grp.loc[grp['round']==rr, 'f1_run'].dropna().values
            series.append(vals if len(vals) else np.array([]))
        if all(len(s)==0 for s in series):
            continue
        plt.figure()
        plt.boxplot(series, positions=rounds_sorted, widths=0.7, manage_ticks=False, showfliers=False)
        plt.xlabel('Round')
        plt.ylabel('F1 (per round across seeds)')
        plt.title(f'{b} / {l} / {m}')
        plt.tight_layout()
        fname = f"boxplot_f1_{b}_{l}_{m}.png".replace(' ', '_')
        plt.savefig(Path(out_dir) / fname, dpi=180)
        plt.close()

# --------------------------------- Main ---------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-root', required=True)
    ap.add_argument('--run-index', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--norm-ref', type=str, default='FedAvg')
    ap.add_argument('--bytes-cols', type=str, default='bytes,bytes_sent,tx_bytes,communication,comm_bytes')
    ap.add_argument('--pheromone-cols', type=str, default='pheromone,avg_pheromone,tau,avg_tau,pheromone_mean')
    ap.add_argument('--f1-cols', type=str, default='f1,f1_val,macro_f1')
    ap.add_argument('--round-cols', type=str, default='round,r')
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    runs = load_runs(args.run_index)

    f1_candidates = [c.strip() for c in args.f1_cols.split(',') if c.strip()]
    round_candidates = [c.strip() for c in args.round_cols.split(',') if c.strip()]
    bytes_candidates = [c.strip() for c in args.bytes_cols.split(',') if c.strip()]
    pheromone_candidates = [c.strip() for c in args.pheromone_cols.split(',') if c.strip()]

    df_runs = aggregate_per_run(repo_root, runs, f1_candidates, round_candidates, bytes_candidates, pheromone_candidates)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    df_runs.to_csv(out_dir / 'runs_aggregated.csv', index=False)

    df_stats = aggregate_across_seeds(df_runs, norm_ref_method=args.norm_ref)
    df_stats.to_csv(out_dir / 'roundwise_stats_plus.csv', index=False)

    # Also export a dispersion table with variance
    if not df_stats.empty:
        disp_cols = ['block','label','method','round','n','f1_mean','f1_std','f1_var','ci_low','ci_high','bytes_cum_mean']
        df_stats[disp_cols].to_csv(out_dir / 'round_dispersion.csv', index=False)

        # F1 vs rounds with CI
        plt.figure()
        for (b,l,m), grp in df_stats.groupby(['block','label','method']):
            grp = grp.sort_values('round')
            x = grp['round'].values
            y = grp['f1_mean'].values
            y_low = grp['ci_low'].values
            y_high = grp['ci_high'].values
            plt.fill_between(x, y_low, y_high, alpha=0.2)
            plt.plot(x, y, label=f'{b}-{l}-{m}')
        plt.xlabel('Round')
        plt.ylabel('F1 (mean with 95% CI)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / 'f1_mean_ci_timeseries.png', dpi=180)
        plt.close()

    # F1 vs normalized communication & pheromone decay & boxplots
    plot_f1_vs_norm_comm(df_stats, out_dir / 'f1_vs_normalized_communication.png')
    plot_pheromone_decay(df_runs, out_dir / 'pheromone_decay_timeseries.png')
    plot_f1_boxplots(df_runs, out_dir)

if __name__ == '__main__':
    main()
