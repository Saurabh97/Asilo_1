#!/usr/bin/env python
# Extended analysis: robust log discovery + F1 stats (mean/std/variance/CI),
# F1 across rounds (refined), F1 vs normalized communication (refined),
# pheromone decay (refined), per-round boxplots,
# and ASILO vs multiple base methods (FedAvg, DFedSAM, DeceFL) by condition.
#
# Usage:
#   python analysis/analyze_stats_plus.py --repo-root <path> --run-index <run_index.json> --out <out_dir> \
#       --norm-ref FedAvg --bytes-cols bytes,bytes_sent,tx_bytes --pheromone-cols pheromone,tau,avg_tau \
#       --base-methods FedAvg,DFedSAM,DeceFL

import os, sys, argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Robust log discovery (tolerates your naming) ----------------
SEED_DUP_RE = re.compile(r"(.*?_seed)(\d+)(?:_seed\2)?$", re.IGNORECASE)

def _variants_for_name(name: str):
    vs = set([name])
    # homogenous / homogeneous swap
    vs.add(name.replace("homogenous", "homogeneous"))
    vs.add(name.replace("homogeneous", "homogenous"))
    # if ends with _seedN_seedN -> also try _seedN and _N
    m = SEED_DUP_RE.match(name)
    if m:
        base, n = m.group(1), m.group(2)
        vs.add(f"{base}{n}")            # drop duplicate
        vs.add(f"{base}{n}_seed{n}")    # original duplicate
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
    """Average across agents within a run (per round). Bytes are summed per round."""
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
            keep = {'round': pd.to_numeric(df[rcol], errors='coerce')}
            if fcol is not None: keep['f1'] = pd.to_numeric(df[fcol], errors='coerce')
            if bcol is not None: keep['bytes'] = pd.to_numeric(df[bcol], errors='coerce')
            if pcol is not None: keep['pheromone'] = pd.to_numeric(df[pcol], errors='coerce')
            per_agent.append(pd.DataFrame(keep))

        if not per_agent:
            continue

        # Concatenate all agents, then groupby round
        df_cat = pd.concat(per_agent, ignore_index=True)
        df_cat = df_cat.dropna(subset=['round'])
        df_cat['round'] = df_cat['round'].astype(int)
        gb = df_cat.groupby('round', dropna=False)

        df_run = pd.DataFrame({'round': sorted(gb.groups.keys())})
        if 'f1' in df_cat.columns:
            df_run['f1_run'] = gb['f1'].mean().reindex(df_run['round']).values
        if 'bytes' in df_cat.columns:
            bytes_round = gb['bytes'].sum().reindex(df_run['round']).fillna(0.0).values
            df_run['bytes_round_run'] = bytes_round
            df_run['bytes_cum_run'] = pd.Series(bytes_round).cumsum().values
        if 'pheromone' in df_cat.columns:
            df_run['pheromone_mean_run'] = gb['pheromone'].mean().reindex(df_run['round']).values

        for _, row in df_run.iterrows():
            rows.append({
                'block': r.get('block'), 'label': r.get('label'), 'method': r.get('method'),
                'seed': r.get('seed'), 'round': int(row['round']),
                'f1_run': float(row['f1_run']) if 'f1_run' in df_run and pd.notna(row['f1_run']) else np.nan,
                'bytes_round_run': float(row['bytes_round_run']) if 'bytes_round_run' in df_run and pd.notna(row.get('bytes_round_run')) else np.nan,
                'bytes_cum_run': float(row['bytes_cum_run']) if 'bytes_cum_run' in df_run and pd.notna(row.get('bytes_cum_run')) else np.nan,
                'pheromone_mean_run': float(row['pheromone_mean_run']) if 'pheromone_mean_run' in df_run and pd.notna(row.get('pheromone_mean_run')) else np.nan
            })
    return pd.DataFrame(rows)

def aggregate_across_seeds(df_runs, norm_ref_method=None):
    """Aggregate across seeds: per (block,label,method,round) compute mean/std/var/CI; comm mean; normalized comm."""
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

# ----------------------------- Style helpers -----------------------------
METHOD_COLORS = {
    "ASILO":   "#1f77b4",  # blue
    "DFedSAM": "#2ca02c",  # green
    "DeceFL":  "#9467bd",  # purple
    "FedAvg":  "#ff7f0e",  # orange
}
COND_STYLES = {  # label (condition) → linestyle
    "homogeneous": "-",
    "heterogeneous-fast25": "--",
    "heterogeneous-fast50": "-.",
}

def _cond_key(lbl: str) -> str:
    """
    Canonicalize condition labels to one of:
      - 'homogeneous'
      - 'heterogeneous-fast25'
      - 'heterogeneous-fast50'
      - 'heterogeneous' (fallback when hetero but no fast bucket found)
    Handles case, spaces, underscores, doubled separators, minor variants.
    """
    s = str(lbl or '').lower().strip()
    s = s.replace('_','-').replace(' ','-')
    s = re.sub(r'-+', '-', s)
    s = re.sub(r'[^a-z0-9\-]', '', s)
    if 'homo' in s:
        return 'homogeneous'
    if 'hetero' in s or 'heterogeneous' in s:
        if ('fast' in s and '25' in s) or 'f25' in s or 'fast-25' in s:
            return 'heterogeneous-fast25'
        if ('fast' in s and '50' in s) or 'f50' in s or 'fast-50' in s:
            return 'heterogeneous-fast50'
        return 'heterogeneous'
    if ('fast' in s and '25' in s) or s.endswith('fast25'):
        return 'heterogeneous-fast25'
    if ('fast' in s and '50' in s) or s.endswith('fast50'):
        return 'heterogeneous-fast50'
    return 'homogeneous' if s == '' else s

def _smooth(y, w=3):
    if y is None or len(y) == 0: return y
    s = pd.Series(y, dtype=float).rolling(window=w, min_periods=1, center=True).mean()
    return s.values

# ----------------------------- Plot helpers -----------------------------
def plot_f1_rounds_with_ci(df_stats, out_dir: Path):
    # overall figure (all methods)
    plt.figure(figsize=(10,6))
    for (b,l,m), grp in df_stats.groupby(['block','label','method']):
        grp = grp.sort_values('round')
        x = grp['round'].values
        y = _smooth(grp['f1_mean'].values)
        y_lo = grp['ci_low'].values
        y_hi = grp['ci_high'].values
        color = METHOD_COLORS.get(m, None)
        style = COND_STYLES.get(_cond_key(l), ":")
        lab = f"{_cond_key(l)}–{m}"
        plt.fill_between(x, y_lo, y_hi, alpha=0.15, color=color)
        plt.plot(x, y, linestyle=style, linewidth=2, color=color, label=lab)
    plt.xlabel("Round")
    plt.ylabel("F1 (mean with 95% CI)")
    plt.title("F1 across rounds (all methods)")
    plt.grid(True, alpha=0.25)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(out_dir / "f1_mean_ci_timeseries.png", dpi=220)
    plt.close()

    # per-method facets (one PNG per method)
    for m, g in df_stats.groupby('method'):
        plt.figure(figsize=(9,6))
        for (b,l,_), grp in g.groupby(['block','label','method']):
            grp = grp.sort_values('round')
            x = grp['round'].values
            y = _smooth(grp['f1_mean'].values)
            y_lo = grp['ci_low'].values
            y_hi = grp['ci_high'].values
            style = COND_STYLES.get(_cond_key(l), ":")
            lab = _cond_key(l)
            plt.fill_between(x, y_lo, y_hi, alpha=0.15, color=METHOD_COLORS.get(m))
            plt.plot(x, y, linestyle=style, linewidth=2, color=METHOD_COLORS.get(m), label=lab)
        plt.xlabel("Round")
        plt.ylabel("F1 (mean with 95% CI)")
        plt.title(f"F1 across rounds — {m}")
        plt.grid(True, alpha=0.25)
        plt.legend(title="Condition", bbox_to_anchor=(1.02,1), loc="upper left", borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(out_dir / f"f1_mean_ci_timeseries_{m}.png", dpi=220)
        plt.close()

def plot_f1_vs_norm_comm_refined(df_stats, out_png):
    if df_stats.empty or 'comm_norm' not in df_stats.columns or df_stats['comm_norm'].isna().all():
        print('[WARN] No normalized communication available; skip plot.')
        return
    plt.figure(figsize=(10,6))
    for (b,l,m), grp in df_stats.groupby(['block','label','method']):
        grp = grp.dropna(subset=['comm_norm','f1_mean'])
        if grp.empty: continue

        # mask NaNs and prep arrays
        x = np.asarray(grp['comm_norm'].values, dtype=float)
        y = np.asarray(grp['f1_mean'].values, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size == 0:
            continue

        color = METHOD_COLORS.get(m, None)
        style = COND_STYLES.get(_cond_key(l), ":")

        # faint scatter
        plt.scatter(x, y, s=12, alpha=0.25, color=color)

        # robust bins
        xmin, xmax = float(np.min(x)), float(np.max(x))
        if np.isclose(xmin, xmax):
            # constant-x: create a tiny bin around the value
            eps = max(1e-9, abs(xmin) * 1e-6)
            bins = np.array([xmin - eps, xmax + eps], dtype=float)
        else:
            bins = np.linspace(xmin, xmax, 10)

        # assign bins (drop duplicates if any)
        cats = pd.cut(x, bins, include_lowest=True, duplicates='drop')

        # compute binned means
        codes = cats.codes
        uniq = np.unique(codes[codes >= 0])
        if uniq.size == 0:
            continue
        centers = np.array([(bins[i] + bins[i+1]) / 2.0 for i in uniq], dtype=float)
        mean_vals = np.array([y[codes == i].mean() for i in uniq], dtype=float)

        plt.plot(centers, _smooth(mean_vals), linestyle=style, linewidth=2, color=color, label=f"{_cond_key(l)}–{m}")

    plt.axvline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.6)  # ref (same as baseline comm)
    plt.xlabel("Normalized Communication (bytes / ref bytes)")
    plt.ylabel("F1 (mean across seeds)")
    plt.title("F1 vs Normalized Communication")
    plt.grid(True, alpha=0.25)
    plt.legend(bbox_to_anchor=(1.02,1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_pheromone_decay_refined(df_runs, out_png):
    df_p = df_runs.dropna(subset=['pheromone_mean_run'])
    if df_p.empty:
        print('[WARN] No pheromone column found; skip pheromone plot.')
        return
    # normalize pheromone per (block,label,method,seed) to start at 1, then average
    df_norm = []
    for (b,l,m,s), grp in df_p.groupby(['block','label','method','seed']):
        grp = grp.sort_values('round')
        v = grp['pheromone_mean_run'].values.astype(float)
        if len(v)==0: continue
        base = v[0] if v[0] != 0 else (np.max(v) if np.max(v)!=0 else 1.0)
        vn = v / (base if base!=0 else 1.0)
        t = grp['round'].values
        df_norm.append(pd.DataFrame({'block':b,'label':l,'method':m,'seed':s,'round':t,'pher_n':vn}))
    if not df_norm:
        print('[WARN] Pheromone normalisation failed; skip.')
        return
    D = pd.concat(df_norm, ignore_index=True)
    plt.figure(figsize=(10,6))
    for (b,l,m), grp in D.groupby(['block','label','method']):
        g2 = grp.groupby('round', as_index=False, observed=False)['pher_n'].mean().sort_values('round')
        color = METHOD_COLORS.get(m, None)
        style = COND_STYLES.get(_cond_key(l), ":")
        plt.plot(g2['round'].values, g2['pher_n'].values + 1e-6, linestyle=style, linewidth=2, color=color, label=f"{_cond_key(l)}–{m}")
    plt.yscale('log')
    plt.xlabel("Round")
    plt.ylabel("Relative pheromone (log scale, mean across seeds)")
    plt.title("Pheromone decay (normalised per run)")
    plt.grid(True, which='both', alpha=0.25)
    plt.legend(bbox_to_anchor=(1.02,1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_f1_boxplots(df_runs, out_dir: Path):
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
        plt.figure(figsize=(10,5))
        plt.boxplot(series, positions=rounds_sorted, widths=0.7, manage_ticks=False, showfliers=False)
        plt.xlabel('Round')
        plt.ylabel('F1 (per round across seeds)')
        plt.title(f'{b} / {_cond_key(l)} / {m}')
        plt.grid(True, axis='y', alpha=0.25)
        plt.tight_layout()
        fname = f"boxplot_f1_{b}_{_cond_key(l)}_{m}.png".replace(' ', '_')
        plt.savefig(out_dir / fname, dpi=220)
        plt.close()

# ---------- NEW: ASILO vs multiple base methods comparison plots ----------
def plot_asilo_vs_bases_by_condition(df_stats, out_dir: Path, base_methods):
    """
    For each condition (homogeneous, heterogeneous-fast25, heterogeneous-fast50),
    overlay ASILO + all requested base methods (e.g., FedAvg, DFedSAM, DeceFL).
    Produces:
      - f1_asilo_vs_bases_homogeneous.png
      - f1_asilo_vs_bases_fast25.png
      - f1_asilo_vs_bases_fast50.png
    """
    if df_stats.empty:
        print('[WARN] No stats available; skip ASILO vs bases comparison plots.')
        return

    if 'ASILO' not in df_stats['method'].unique():
        print('[WARN] ASILO rows missing; skip ASILO vs bases comparison plots.')
        return

    # canonical condition keys
    targets = [
        ('homogeneous',          'f1_asilo_vs_bases_homogeneous.png',   'Homogeneous'),
        ('heterogeneous-fast25', 'f1_asilo_vs_bases_fast25.png',         'Heterogeneous — Fast 25'),
        ('heterogeneous-fast50', 'f1_asilo_vs_bases_fast50.png',         'Heterogeneous — Fast 50'),
    ]

    def _series_for_method(gdf, method: str):
        """Aggregate to a single curve per method by averaging over blocks at each round."""
        gm = gdf[gdf['method'] == method]
        if gm.empty:
            return None
        agg = gm.groupby('round', as_index=False, observed=False).agg(
            f1_mean=('f1_mean','mean'),
            ci_low=('ci_low','mean'),
            ci_high=('ci_high','mean')
        ).sort_values('round')
        if agg.empty:
            return None
        return agg

    for cond_key, fname, title_suffix in targets:
        # filter rows for this condition (normalize labels)
        sel = df_stats[df_stats['label'].map(lambda v: _cond_key(v) == cond_key)]
        if sel.empty:
            print(f'[WARN] No rows for condition "{cond_key}" → skipping {fname}')
            continue

        methods_to_plot = ['ASILO'] + [m for m in base_methods if m != 'ASILO']
        present = [m for m in methods_to_plot if m in sel['method'].unique()]
        if len(present) < 2:
            print(f'[WARN] Not enough methods (have: {present}) for "{cond_key}" → skipping {fname}')
            continue

        plt.figure(figsize=(9.5,6.2))

        # Plot ASILO first with solid line, then bases (dashed)
        for m in present:
            series = _series_for_method(sel, m)
            if series is None:
                print(f'[WARN] Missing data for method {m} in "{cond_key}"')
                continue
            x = series['round'].values
            y = _smooth(series['f1_mean'].values)
            y_lo = series['ci_low'].values
            y_hi = series['ci_high'].values

            color = METHOD_COLORS.get(m, None)
            style = '-' if m == 'ASILO' else '--'

            plt.fill_between(x, y_lo, y_hi, alpha=0.15, color=color)
            plt.plot(x, y, linestyle=style, linewidth=2.2, color=color, label=m)

        plt.xlabel("Round")
        plt.ylabel("F1 (mean with 95% CI)")
        bases_title = ', '.join([b for b in base_methods if b != 'ASILO'])
        plt.title(f"ASILO vs {bases_title} — {title_suffix}")
        plt.grid(True, alpha=0.25)

        # Legend: put ASILO first if present
        handles, labels = plt.gca().get_legend_handles_labels()
        if 'ASILO' in labels:
            order = [labels.index('ASILO')] + [i for i,l in enumerate(labels) if l != 'ASILO']
            handles = [handles[i] for i in order]
            labels = [labels[i] for i in order]
        plt.legend(handles, labels, loc="best", title="Method")
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=220)
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
    ap.add_argument('--base-methods', type=str, default='FedAvg,DFedSAM,DeceFL')
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    runs = load_runs(args.run_index)

    f1_candidates = [c.strip() for c in args.f1_cols.split(',') if c.strip()]
    round_candidates = [c.strip() for c in args.round_cols.split(',') if c.strip()]
    bytes_candidates = [c.strip() for c in args.bytes_cols.split(',') if c.strip()]
    pheromone_candidates = [c.strip() for c in args.pheromone_cols.split(',') if c.strip()]
    base_methods = [m.strip() for m in args.base_methods.split(',') if m.strip()]

    df_runs = aggregate_per_run(repo_root, runs, f1_candidates, round_candidates, bytes_candidates, pheromone_candidates)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    df_runs.to_csv(out_dir / 'runs_aggregated.csv', index=False)

    df_stats = aggregate_across_seeds(df_runs, norm_ref_method=args.norm_ref)
    df_stats.to_csv(out_dir / 'roundwise_stats_plus.csv', index=False)

    # Helpful debug: show which condition buckets we actually have
    try:
        cond_counts = df_stats['label'].map(_cond_key).value_counts().to_dict()
        print(f"[INFO] Condition buckets present: {cond_counts}")
    except Exception:
        pass

    # ----- Refined plots (ALL original ones) -----
    plot_f1_rounds_with_ci(df_stats, out_dir)
    plot_f1_vs_norm_comm_refined(df_stats, out_dir / 'f1_vs_normalized_communication.png')
    plot_pheromone_decay_refined(df_runs, out_dir / 'pheromone_decay_timeseries.png')
    plot_f1_boxplots(df_runs, out_dir)

    # ----- NEW: ASILO vs multiple bases (per condition) -----
    plot_asilo_vs_bases_by_condition(df_stats, out_dir, base_methods=base_methods)

if __name__ == '__main__':
    main()
