#!/usr/bin/env python
# Extended analysis: robust log discovery + F1 stats (mean/std/variance/CI),
# F1 across rounds (refined), F1 vs normalized communication (refined),
# pheromone decay (refined), per-round boxplots,
# and ASILO vs multiple base methods (FedAvg, DFedSAM, DeceFL) by condition.
#
# PLUS: Communication graphs from edges.csv + agg_decisions.csv with
# optional per-method/per-condition splits, and auto-discovery/aggregation
# across experiments listed in run_index.json.

import os, sys, argparse, json, re, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

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

    # skip comms-related CSVs; these are handled by the comms visualizer, not metrics
    DENY = re.compile(
        r"^(edges|edge_stats|effective_stats|agg(?:_decisions)?|communication_graph|effective_contribution)"
        r".*\.csv$", re.IGNORECASE
    )

    def _list_ok(dirpath: Path):
        return [f for f in dirpath.glob("*.csv") if not DENY.match(f.name)]

    d = logs_root / exp_name
    if d.exists():
        return _list_ok(d)
    for v in _variants_for_name(exp_name):
        dv = logs_root / v
        if dv.exists():
            return _list_ok(dv)
    fuzzy = _best_fuzzy_dir(logs_root, exp_name)
    if fuzzy:
        return _list_ok(fuzzy)
    return []


# ---------- NEW: discover edges/agg files inside logs/<exp> ----------
def _discover_exp_dir(repo_root: Path, exp_name: str) -> Path | None:
    logs_root = repo_root / "logs"
    d = logs_root / exp_name
    if d.exists():
        return d
    for v in _variants_for_name(exp_name):
        dv = logs_root / v
        if dv.exists():
            return dv
    return _best_fuzzy_dir(logs_root, exp_name)

def _pick_best_csv(dirpath: Path, exact_names, substrings):
    # 1) exact names
    for nm in exact_names:
        p = dirpath / nm
        if p.exists():
            return p
    # 2) shallow fuzzy
    cands = [p for p in dirpath.glob("*.csv") if any(ss in p.name.lower() for ss in substrings)]
    if cands:
        return sorted(cands, key=lambda p: (len(p.name), p.name.lower()))[0]
    # 3) recursive fuzzy
    cands = [p for p in dirpath.rglob("*.csv") if any(ss in p.name.lower() for ss in substrings)]
    if cands:
        return sorted(cands, key=lambda p: (len(p.name), p.name.lower()))[0]
    return None

def discover_comms_files(repo_root: Path, exp_name: str):
    """
    Return (edges_path, agg_path) for a given experiment, or (None, None) if missing.
    """
    d = _discover_exp_dir(repo_root, exp_name)
    if d is None:
        return None, None
    edges_exact = ["edges.csv", "edge.csv", "comms_edges.csv", "tx_edges.csv"]
    edges_subs  = ["edges", "edge", "comms", "tx"]
    agg_exact   = ["agg_decisions.csv", "agg.csv", "aggregate.csv"]
    agg_subs    = ["agg", "decision"]
    edges_path = _pick_best_csv(d, edges_exact, edges_subs)
    agg_path   = _pick_best_csv(d, agg_exact, agg_subs)
    return edges_path, agg_path

# --------------------- CSV parsing and aggregation (metrics) ---------------------
def parse_csv_numeric(path, f1_candidates, round_candidates, bytes_candidates, pheromone_candidates):
    # robust read
    df = pd.read_csv(path, low_memory=False)
    cols = {c.lower(): c for c in df.columns}

    # skip comms-like files entirely (edges/agg/etc.)
    comm_markers = {'src', 'dst', 'from', 'to', 'action', 'kind', 'reason'}
    if any(m in cols for m in comm_markers):
        raise ValueError(f"Comms-like file {path.name}; skipping")

    def pick(cands):
        for c in cands:
            if c.lower() in cols:
                return cols[c.lower()]
        return None

    f1_col    = pick(f1_candidates)
    bytes_col = pick(bytes_candidates)
    pher_col  = pick(pheromone_candidates)

    # try explicit round-like columns first
    round_col = pick(round_candidates)

    # smart fallback: allow 't' as round ONLY if it's a metrics file
    # (has F1 or pheromone). This still avoids edges.csv contamination.
    if round_col is None and 't' in cols and (f1_col is not None or pher_col is not None):
        round_col = cols['t']

    if round_col is None:
        # keep the warning, but with context
        raise ValueError(f"No round-like column in {path.name}; columns={list(df.columns)}")

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

# includes fast40
COND_STYLES = {
    "homogeneous": "-",
    "heterogeneous-fast25": "--",
    "heterogeneous-fast40": ":",
    "heterogeneous-fast50": "-.",
}

def _cond_key(lbl: str) -> str:
    """
    Canonicalize condition labels to one of:
      - 'homogeneous'
      - 'heterogeneous-fast25'
      - 'heterogeneous-fast50'
      - 'heterogeneous-fast75'
      - 'heterogeneous-fast100'
      - 'heterogeneous' (fallback when hetero but no fast bucket found)
    Also recognizes short forms like 'fast25', 'fast40', 'fast50'.
    """
    s = str(lbl or '').lower().strip()
    s = s.replace('_','-').replace(' ','-')
    s = re.sub(r'-+', '-', s)
    s = re.sub(r'[^a-z0-9\-]', '', s)

    if 'homo' in s:
        return 'homogeneous'
    if 'hetero' in s or 'heterogeneous' in s:
        if ('fast' in s and '25' in s) or 'f25' in s or s.endswith('fast-25'):
            return 'heterogeneous-fast25'
        if ('fast' in s and '50' in s) or 'f50' in s or s.endswith('fast-50'):
            return 'heterogeneous-fast50'
        if ('fast' in s and '75' in s) or 'f75' in s or s.endswith('fast-75'):
            return 'heterogeneous-fast75'
        if ('fast' in s and '100' in s) or 'f100' in s or s.endswith('fast-100'):
            return 'heterogeneous-fast100'
        return 'heterogeneous'
    if ('fast' in s and '25' in s) or s.endswith('fast25'):
        return 'heterogeneous-fast25'
    if ('fast' in s and '50' in s) or s.endswith('fast50'):
        return 'heterogeneous-fast50'
    if ('fast' in s and '75' in s) or s.endswith('fast75'):
        return 'heterogeneous-fast75'
    if ('fast' in s and '100' in s) or s.endswith('fast100'):
        return 'heterogeneous-fast100'
    return 'homogeneous' if s == '' else s

def _smooth(y, w=3):
    if y is None or len(y) == 0: return y
    s = pd.Series(y, dtype=float).rolling(window=w, min_periods=1, center=True).mean()
    return s.values

# ----------------------------- Plot helpers (F1 etc.) -----------------------------
def plot_f1_rounds_with_ci(df_stats, out_dir: Path):
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
    plt.xlabel("Round"); plt.ylabel("F1 (mean with 95% CI)")
    plt.title("F1 across rounds (all methods)")
    plt.grid(True, alpha=0.25)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(out_dir / "f1_mean_ci_timeseries.png", dpi=220)
    plt.close()

    # per-method facets
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
        plt.xlabel("Round"); plt.ylabel("F1 (mean with 95% CI)")
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
        x = np.asarray(grp['comm_norm'].values, dtype=float)
        y = np.asarray(grp['f1_mean'].values, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size == 0: continue
        color = METHOD_COLORS.get(m, None)
        style = COND_STYLES.get(_cond_key(l), ":")
        plt.scatter(x, y, s=12, alpha=0.25, color=color)
        xmin, xmax = float(np.min(x)), float(np.max(x))
        if np.isclose(xmin, xmax):
            eps = max(1e-9, abs(xmin) * 1e-6)
            bins = np.array([xmin - eps, xmax + eps], dtype=float)
        else:
            bins = np.linspace(xmin, xmax, 10)
        cats = pd.cut(x, bins, include_lowest=True, duplicates='drop')
        codes = cats.codes
        uniq = np.unique(codes[codes >= 0])
        if uniq.size == 0: continue
        centers = np.array([(bins[i] + bins[i+1]) / 2.0 for i in uniq], dtype=float)
        mean_vals = np.array([y[codes == i].mean() for i in uniq], dtype=float)
        plt.plot(centers, _smooth(mean_vals), linestyle=style, linewidth=2, color=color, label=f"{_cond_key(l)}–{m}")
    plt.axvline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.6)
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
    plt.xlabel("Round"); plt.ylabel("Relative pheromone (log scale, mean across seeds)")
    plt.title("Pheromone decay (normalised per run)")
    plt.grid(True, which='both', alpha=0.25)
    plt.legend(bbox_to_anchor=(1.02,1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_f1_boxplots(df_runs, out_dir: Path):
    if df_runs.empty or df_runs['f1_run'].dropna().empty:
        print('[WARN] No F1 values for boxplot.')
        return
    for (b,l,m), grp in df_runs.groupby(['block','label','method']):
        grp = grp.dropna(subset=['round','f1_run'])
        if grp.empty: continue
        rounds_sorted = sorted({int(r) for r in grp['round'].dropna().values})
        series = []
        for rr in rounds_sorted:
            vals = grp.loc[grp['round']==rr, 'f1_run'].dropna().values
            series.append(vals if len(vals) else np.array([]))
        if all(len(s)==0 for s in series): continue
        plt.figure(figsize=(10,5))
        plt.boxplot(series, positions=rounds_sorted, widths=0.7, manage_ticks=False, showfliers=False)
        plt.xlabel('Round'); plt.ylabel('F1 (per round across seeds)')
        plt.title(f'{b} / {_cond_key(l)} / {m}')
        plt.grid(True, axis='y', alpha=0.25)
        plt.tight_layout()
        fname = f"boxplot_f1_{b}_{_cond_key(l)}_{m}.png".replace(' ', '_')
        plt.savefig(out_dir / fname, dpi=220)
        plt.close()

# ---------- ASILO vs bases (per condition, incl. fast40) ----------
def plot_asilo_vs_bases_by_condition(df_stats, out_dir: Path, base_methods):
    if df_stats.empty:
        print('[WARN] No stats available; skip ASILO vs bases comparison plots.')
        return
    if 'ASILO' not in df_stats['method'].unique():
        print('[WARN] ASILO rows missing; skip ASILO vs bases comparison plots.')
        return

    targets = [
        ('homogeneous',          'f1_asilo_vs_bases_homogeneous.png',   'Homogeneous'),
        ('heterogeneous-fast25', 'f1_asilo_vs_bases_fast25.png',        'Heterogeneous — Fast 25'),
        ('heterogeneous-fast50', 'f1_asilo_vs_bases_fast50.png',        'Heterogeneous — Fast 50'),
        ('heterogeneous-fast75', 'f1_asilo_vs_bases_fast75.png',        'Heterogeneous — Fast 75'),
        ('heterogeneous-fast100', 'f1_asilo_vs_bases_fast100.png',      'Heterogeneous — Fast 100'),
    ]

    def _series_for_method(gdf, method: str):
        gm = gdf[gdf['method'] == method]
        if gm.empty: return None
        agg = gm.groupby('round', as_index=False, observed=False).agg(
            f1_mean=('f1_mean','mean'),
            ci_low=('ci_low','mean'),
            ci_high=('ci_high','mean')
        ).sort_values('round')
        return agg if not agg.empty else None

    for cond_key, fname, title_suffix in targets:
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

        plt.xlabel("Round"); plt.ylabel("F1 (mean with 95% CI)")
        bases_title = ', '.join([b for b in base_methods if b != 'ASILO'])
        plt.title(f"ASILO vs {bases_title} — {title_suffix}")
        plt.grid(True, alpha=0.25)
        handles, labels = plt.gca().get_legend_handles_labels()
        if 'ASILO' in labels:
            order = [labels.index('ASILO')] + [i for i,l in enumerate(labels) if l != 'ASILO']
            handles = [handles[i] for i in order]; labels = [labels[i] for i in order]
        plt.legend(handles, labels, loc="best", title="Method")
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=220)
        plt.close()

# ===================== Communication graph helpers =====================
def _comms_load_csv(path: Path):
    if path is None:
        return None
    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return None
    try:
        df = pd.read_csv(path, low_memory=False)  # patched: robust dtype
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    except Exception as e:
        print(f"[ERROR] Failed reading {path}: {e}")
        return None

def _comms_ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def _agg_edge_counts(df: pd.DataFrame, name: str):
    if df is None or df.empty:
        return pd.DataFrame(columns=["src", "dst", name])
    g = df.groupby(["src", "dst"], as_index=False).size()
    g.rename(columns={"size": name}, inplace=True)
    return g

def _agg_edge_sum(df: pd.DataFrame, value_col: str, name: str):
    if df is None or df.empty:
        return pd.DataFrame(columns=["src", "dst", name])
    g = df.groupby(["src", "dst"], as_index=False)[value_col].sum()
    g.rename(columns={value_col: name}, inplace=True)
    return g

def build_edge_stats(E: pd.DataFrame) -> pd.DataFrame:
    # split by action
    sent_df = E[E.get("action","").eq("sent")] if "action" in E.columns else E.iloc[0:0]
    recv_df = E[E.get("action","").eq("received")] if "action" in E.columns else E.iloc[0:0]
    buff_df = E[E.get("action","").eq("buffered")] if "action" in E.columns else E.iloc[0:0]
    drop_df = E[E.get("action","").eq("drop_cos")] if "action" in E.columns else E.iloc[0:0]
    lowb_df = E[E.get("action","").eq("buffered_low_cos")] if "action" in E.columns else E.iloc[0:0]

    sent_cnt = _agg_edge_counts(sent_df, "sent_count")
    recv_cnt = _agg_edge_counts(recv_df, "received_count")
    buff_cnt = _agg_edge_counts(buff_df, "buffered_count")
    drop_cnt = _agg_edge_counts(drop_df, "drop_cos_count")
    lowb_cnt = _agg_edge_counts(lowb_df, "buffered_lowcos_count")

    sent_bytes = _agg_edge_sum(sent_df, "bytes", "sent_bytes")
    recv_bytes = _agg_edge_sum(recv_df, "bytes", "received_bytes")  # patched: include received bytes
    buff_bytes = _agg_edge_sum(buff_df, "bytes", "buffered_bytes")
    drop_bytes = _agg_edge_sum(drop_df, "bytes", "dropped_bytes")
    lowb_bytes = _agg_edge_sum(lowb_df, "bytes", "buffered_lowcos_bytes")

    edge_stats = (
        sent_cnt.merge(recv_cnt, on=["src", "dst"], how="outer")
        .merge(buff_cnt, on=["src", "dst"], how="outer")
        .merge(drop_cnt, on=["src", "dst"], how="outer")
        .merge(lowb_cnt, on=["src", "dst"], how="outer")
        .merge(sent_bytes, on=["src", "dst"], how="outer")
        .merge(recv_bytes, on=["src", "dst"], how="outer")
        .merge(buff_bytes, on=["src", "dst"], how="outer")
        .merge(drop_bytes, on=["src", "dst"], how="outer")
        .merge(lowb_bytes, on=["src", "dst"], how="outer")
    ).fillna(0)
    edge_stats = edge_stats.infer_objects(copy=False)  # patched: silence future downcast warning

    # cast types
    for c in ["sent_count","received_count","buffered_count","drop_cos_count","buffered_lowcos_count"]:
        if c in edge_stats.columns:
            edge_stats[c] = edge_stats[c].astype(int)
    for c in ["sent_bytes","received_bytes","buffered_bytes","dropped_bytes","buffered_lowcos_bytes"]:
        if c in edge_stats.columns:
            edge_stats[c] = edge_stats[c].astype(int)
    return edge_stats

def build_effective_stats(A: pd.DataFrame) -> pd.DataFrame:
    if A is None or A.empty:
        return pd.DataFrame(columns=["src", "dst", "accepted_count", "dropped_in_agg_count"])
    A = A.copy()
    A.columns = [c.strip().lower() for c in A.columns]
    A.rename(columns={"from": "src", "agent": "dst"}, inplace=True)
    acc = A[A.get("decision",'') == "accepted_in_agg"].groupby(["src", "dst"], as_index=False).size()
    acc.rename(columns={"size": "accepted_count"}, inplace=True)
    if "decision" in A.columns:
        drp = A[A["decision"] == "dropped_in_agg"].groupby(["src", "dst"], as_index=False).size()
        drp.rename(columns={"size": "dropped_in_agg_count"}, inplace=True)
    else:
        drp = pd.DataFrame(columns=["src", "dst", "dropped_in_agg_count"])
    eff = acc.merge(drp, on=["src", "dst"], how="outer").fillna(0)
    eff["accepted_count"] = eff["accepted_count"].astype(int)
    eff["dropped_in_agg_count"] = eff["dropped_in_agg_count"].astype(int)
    return eff

def _layout_positions(G, layout: str, seed: int = 42):
    layout = (layout or "spring").lower()
    if layout == "kamada":
        return nx.kamada_kawai_layout(G)
    if layout == "circular":
        return nx.circular_layout(G)
    if layout == "random":
        return nx.random_layout(G, seed=seed)
    return nx.spring_layout(G, seed=seed)

def plot_communication_graph(edge_stats: pd.DataFrame, out_png: Path, layout: str = "spring",
                             min_bytes: int = 0, min_count: int = 0, seed: int = 42):
    es = edge_stats.copy()
    # patched: ensure missing cols exist
    for col in ["sent_count","buffered_count","drop_cos_count"]:
        if col not in es.columns: es[col] = 0
    for col in ["sent_bytes","received_bytes","buffered_bytes","dropped_bytes","buffered_lowcos_bytes"]:
        if col not in es.columns: es[col] = 0

    cnt_ok = (es[["sent_count","buffered_count","drop_cos_count"]].sum(axis=1) >= min_count)
    byt_ok = (es[["sent_bytes","received_bytes","buffered_bytes","dropped_bytes","buffered_lowcos_bytes"]].sum(axis=1) >= min_bytes)
    es = es[cnt_ok & byt_ok]

    nodes = sorted(set(es["src"]).union(set(es["dst"])))
    G = nx.DiGraph(); G.add_nodes_from(nodes)

    for _, r in es.iterrows():
        total_b = int(r.get("sent_bytes",0)) + int(r.get("received_bytes",0)) + int(r.get("buffered_bytes",0)) \
                  + int(r.get("dropped_bytes",0)) + int(r.get("buffered_lowcos_bytes",0))
        width = 1.0 + math.log1p(max(0, total_b)) / 3.0  # patched: total bytes
        style = "dashed" if int(r.get("drop_cos_count",0)) > int(r.get("buffered_count",0)) else "solid"
        G.add_edge(r["src"], r["dst"], width=width, style=style)

    if G.number_of_edges() == 0:
        print("[WARN] No edges to draw in communication_graph (after filters).")
        return

    pos = _layout_positions(G, layout, seed=seed)
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=9)
    solid = [(u, v) for u, v, d in G.edges(data=True) if d.get("style") == "solid"]
    dash = [(u, v) for u, v, d in G.edges(data=True) if d.get("style") == "dashed"]
    nx.draw_networkx_edges(G, pos, edgelist=solid, width=[G[u][v]["width"] for u, v in solid])
    nx.draw_networkx_edges(G, pos, edgelist=dash, width=[G[u][v]["width"] for u, v in dash], style="dashed")
    plt.title("Communication Graph (solid: buffered ≥ drops; dashed: drops > buffered)\nEdge width ∝ total bytes")
    plt.axis("off")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png.as_posix(), dpi=160)
    plt.close()
    print(f"[OK] Wrote {out_png}")

def plot_effective_graph(eff_stats: pd.DataFrame, out_png: Path, layout: str = "spring",
                         min_accepts: int = 1, seed: int = 7):
    es = eff_stats.copy()
    if es.empty:
        print("[WARN] effective_stats is empty — skipping effective_contribution_graph.")
        return
    es = es[es.get("accepted_count", 0) >= min_accepts]
    G = nx.DiGraph()
    nodes = sorted(set(es["src"]).union(set(es["dst"])))
    G.add_nodes_from(nodes)
    for _, r in es.iterrows():
        acc = int(r.get("accepted_count", 0))
        width = 1.0 + math.log1p(max(0, acc))
        G.add_edge(r["src"], r["dst"], width=width)
    if G.number_of_edges() == 0:
        print("[WARN] No edges to draw in effective_contribution_graph (after filters).")
        return
    pos = _layout_positions(G, layout, seed=seed)
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=9)
    nx.draw_networkx_edges(G, pos, width=[G[u][v]["width"] for u, v in G.edges()])
    plt.title("Effective Contribution Graph (edge width ∝ accepted_in_agg count)")
    plt.axis("off")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png.as_posix(), dpi=160)
    plt.close()
    print(f"[OK] Wrote {out_png}")

def _safe_slug(s: str) -> str:
    s = str(s or '').strip()
    s = s.replace(' ', '_')
    s = re.sub(r'[^a-zA-Z0-9_\-\.]+', '', s)
    return s

def _maybe_split_groups(E: pd.DataFrame, A: pd.DataFrame, split_method: bool, split_cond: bool):
    have_method = ('method' in E.columns) or (A is not None and 'method' in A.columns)
    have_label  = ('label'  in E.columns) or (A is not None and 'label'  in A.columns) or ('condition' in E.columns) or (A is not None and 'condition' in A.columns)
    if 'condition' in E.columns and 'label' not in E.columns: E = E.rename(columns={'condition':'label'})
    if A is not None and 'condition' in A.columns and 'label' not in A.columns: A = A.rename(columns={'condition':'label'})
    if (split_method and have_method) or (split_cond and have_label):
        methods = sorted(pd.unique(pd.concat([
            E['method'] if 'method' in E.columns else pd.Series([], dtype=str),
            A['method'] if (A is not None and 'method' in A.columns) else pd.Series([], dtype=str)
        ], ignore_index=True).dropna().astype(str))) if split_method and have_method else [None]
        labels = sorted(pd.unique(pd.concat([
            E['label'] if 'label' in E.columns else pd.Series([], dtype=str),
            A['label'] if (A is not None and 'label' in A.columns) else pd.Series([], dtype=str)
        ], ignore_index=True).dropna().astype(str))) if split_cond and have_label else [None]
        groups = []
        for m in methods:
            for l in labels:
                Es = E.copy()
                As = A.copy() if A is not None else None
                if m is not None:
                    if 'method' in Es.columns: Es = Es[Es['method'].astype(str)==m]
                    if As is not None and 'method' in As.columns: As = As[As['method'].astype(str)==m]
                if l is not None:
                    if 'label' in Es.columns: Es = Es[Es['label'].astype(str)==l]
                    if As is not None and 'label' in As.columns: As = As[As['label'].astype(str)==l]
                if Es is not None and not Es.empty:
                    groups.append((m, l, Es, (As if As is not None else None)))
        return groups if groups else [(None, None, E, A)]
    return [(None, None, E, A)]

def run_comms_visualization(edges_path: Path, agg_path: Path | None, out_dir: Path,
                            layout: str, min_bytes: int, min_count: int, min_accepts: int,
                            seed: int, split_by_method: bool, split_by_condition: bool):
    E = _comms_load_csv(edges_path)
    if E is None or E.empty:
        print("edges.csv not found or empty — skipping comms visualization.")
        return
    # tolerate from/to
    for col in ["src","dst"]:
        if col not in E.columns:
            if col == "src" and "from" in E.columns: E.rename(columns={"from":"src"}, inplace=True)
            elif col == "dst" and "to" in E.columns: E.rename(columns={"to":"dst"}, inplace=True)
    if "src" not in E.columns or "dst" not in E.columns:
        raise SystemExit("edges.csv must contain 'src' and 'dst' (or 'from'/'to').")

    # bytes column fallback
    if "bytes" not in E.columns:
        for altb in ["tx_bytes", "comm_bytes", "size", "payload_bytes"]:
            if altb in E.columns:
                E.rename(columns={altb: "bytes"}, inplace=True); break
        if "bytes" not in E.columns:
            raise SystemExit("edges.csv is missing 'bytes' column (or known alternatives).")

    # ---- Action canonicalization (patched) ----
    if "action" in E.columns:
        E["action"] = E["action"].astype(str).str.lower().str.strip()
    else:
        E["action"] = "sent"
    ACTION_MAP = {
        # sent
        'send':'sent','sent':'sent','tx':'sent','transmit':'sent','broadcast':'sent','push':'sent',
        # received
        'receive':'received','received':'received','recv':'received','rx':'received','pull':'received',
        # buffered
        'buffer':'buffered','buffered':'buffered','queue':'buffered','queued':'buffered',
        # buffered low cos
        'buffer_low_cos':'buffered_low_cos','buffered_low_cos':'buffered_low_cos','low_cos_buffer':'buffered_low_cos',
        # dropped
        'drop':'drop_cos','dropped':'drop_cos','reject':'drop_cos','rejected':'drop_cos','pruned':'drop_cos','trimmed':'drop_cos'
    }
    allowed_actions = {'sent','received','buffered','buffered_low_cos','drop_cos'}
    E['action'] = E['action'].map(ACTION_MAP).fillna(E['action'])
    E.loc[~E['action'].isin(allowed_actions), 'action'] = 'sent'

    _comms_ensure_numeric(E, ["t", "bytes", "cos"])

    A = None
    if agg_path is not None:
        A = _comms_load_csv(agg_path)
        if A is not None and not A.empty:
            _comms_ensure_numeric(A, ["t", "bytes", "cos_thresh", "trim_p", "rolled_back"])

    groups = _maybe_split_groups(E, A, split_by_method, split_by_condition)
    out_dir.mkdir(parents=True, exist_ok=True)
    for m_key, l_key, Esub, Asub in groups:
        edge_stats = build_edge_stats(Esub)

        # patched: debug when nothing drawable
        if edge_stats.empty:
            acts = (Esub['action'].value_counts().to_dict() if 'action' in Esub.columns else {})
            print(f"[INFO] No derived edges for group method={m_key} label={l_key}. Actions present: {acts}")

        suffix = ""
        if m_key is not None: suffix += f"_{_safe_slug(m_key)}"
        if l_key is not None: suffix += f"_{_safe_slug(_cond_key(l_key))}"
        edge_stats_path = out_dir / f"edge_stats{suffix}.csv"
        edge_stats.to_csv(edge_stats_path, index=False)
        print(f"[OK] Wrote {edge_stats_path}")

        plot_communication_graph(
            edge_stats=edge_stats,
            out_png=out_dir / f"communication_graph{suffix}.png",
            layout=layout, min_bytes=min_bytes, min_count=min_count, seed=seed,
        )

        eff_stats = pd.DataFrame(columns=["src", "dst", "accepted_count", "dropped_in_agg_count"])
        if Asub is not None and not Asub.empty:
            eff_stats = build_effective_stats(Asub)
        eff_stats_path = out_dir / f"effective_stats{suffix}.csv"
        eff_stats.to_csv(eff_stats_path, index=False)
        print(f"[OK] Wrote {eff_stats_path}")
        plot_effective_graph(
            eff_stats=eff_stats,
            out_png=out_dir / f"effective_contribution_graph{suffix}.png",
            layout=layout, min_accepts=min_accepts, seed=max(1, seed - 35),
        )

# ---------- NEW: build combined comms from auto-discovered per-experiment files ----------
def _normalize_edges_df(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    # unify src/dst
    if 'src' not in df.columns and 'from' in df.columns: df.rename(columns={'from':'src'}, inplace=True)
    if 'dst' not in df.columns and 'to' in df.columns:   df.rename(columns={'to':'dst'}, inplace=True)
    # bytes fallback
    if 'bytes' not in df.columns:
        for altb in ['tx_bytes','comm_bytes','size','payload_bytes']:
            if altb in df.columns:
                df.rename(columns={altb:'bytes'}, inplace=True); break
    # default action if absent
    if 'action' not in df.columns:
        df['action'] = 'sent'

    # bytes to numeric
    if 'bytes' in df.columns:
        df['bytes'] = pd.to_numeric(df['bytes'], errors='coerce').fillna(0).astype('int64')

    # ---- Action canonicalization (patched) ----
    allowed = {'sent', 'received', 'buffered', 'buffered_low_cos', 'drop_cos'}
    df['action'] = df['action'].astype(str).str.lower().str.strip()
    ACTION_MAP = {
        'send':'sent','sent':'sent','tx':'sent','transmit':'sent','broadcast':'sent','push':'sent',
        'receive':'received','received':'received','recv':'received','rx':'received','pull':'received',
        'buffer':'buffered','buffered':'buffered','queue':'buffered','queued':'buffered',
        'buffer_low_cos':'buffered_low_cos','buffered_low_cos':'buffered_low_cos','low_cos_buffer':'buffered_low_cos',
        'drop':'drop_cos','dropped':'drop_cos','reject':'drop_cos','rejected':'drop_cos','pruned':'drop_cos','trimmed':'drop_cos'
    }
    df['action'] = df['action'].map(ACTION_MAP).fillna(df['action'])
    df.loc[~df['action'].isin(allowed), 'action'] = 'sent'

    # attach meta if missing
    for k in ['method','label','block','seed','exp_name']:
        if k not in df.columns:
            df[k] = meta.get(k)
    return df

def _normalize_agg_df(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    for k in ['method','label','block','seed','exp_name']:
        if k not in df.columns:
            df[k] = meta.get(k)
    return df

def build_combined_comms_from_runs(repo_root: Path, runs: list):
    E_list, A_list = [], []
    missing_edges, missing_agg = 0, 0
    for r in runs:
        exp = r.get('exp_name')
        edges_path, agg_path = discover_comms_files(repo_root, exp)
        meta = {
            'method': r.get('method'),
            'label':  r.get('label'),
            'block':  r.get('block'),
            'seed':   r.get('seed'),
            'exp_name': exp
        }
        # edges
        if edges_path and edges_path.exists():
            try:
                dfE = pd.read_csv(edges_path, low_memory=False)  # patched: robust read
                dfE = _normalize_edges_df(dfE, meta)
                if 'src' in dfE.columns and 'dst' in dfE.columns and 'bytes' in dfE.columns:
                    E_list.append(dfE)
                else:
                    print(f"[WARN] {edges_path} missing required columns; skipped.")
            except Exception as e:
                print(f"[WARN] Failed reading edges for {exp}: {e}")
        else:
            missing_edges += 1
        # agg
        if agg_path and agg_path.exists():
            try:
                dfA = pd.read_csv(agg_path, low_memory=False)  # patched: robust read
                if not dfA.empty:
                    dfA = _normalize_agg_df(dfA, meta)
                    A_list.append(dfA)
            except Exception as e:
                print(f"[WARN] Failed reading agg for {exp}: {e}")
        else:
            missing_agg += 1
    E_all = pd.concat(E_list, ignore_index=True) if E_list else pd.DataFrame()
    A_all = pd.concat(A_list, ignore_index=True) if A_list else pd.DataFrame()
    if missing_edges:
        print(f"[INFO] Auto-comm discovery: {missing_edges} experiment(s) had no edges file.")
    if missing_agg:
        print(f"[INFO] Auto-comm discovery: {missing_agg} experiment(s) had no agg file.")
    return E_all, A_all

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
    ap.add_argument('--round-cols', type=str,
    default='round,r,round_idx,global_round,client_round,server_round,epoch,iter,step,t')
    ap.add_argument('--base-methods', type=str, default='FedAvg,DFedSAM,DeceFL')

    # Manual comms visualization args
    ap.add_argument('--edges', type=str, default=None, help='Path to edges.csv')
    ap.add_argument('--agg', type=str, default=None, help='(Optional) Path to agg_decisions.csv')
    ap.add_argument('--pheromone', type=str, default=None, help='(Optional) Path to pheromone.csv (unused here)')
    ap.add_argument('--comms-out', type=str, default=None, help='Output dir for comms graphs/CSVs (default: <out>/comms)')
    ap.add_argument('--comms-layout', choices=['spring','kamada','circular','random'], default='spring', help='Graph layout')
    ap.add_argument('--comms-min-bytes', type=int, default=0, help='Min total bytes to show an edge')
    ap.add_argument('--comms-min-count', type=int, default=0, help='Min total event count to show an edge')
    ap.add_argument('--comms-min-accepts', type=int, default=1, help='Min accepted_in_agg to render an edge in effective graph')
    ap.add_argument('--comms-seed', type=int, default=42, help='Layout seed for reproducibility')
    ap.add_argument('--comms-split-by-method', action='store_true', help='Split graphs per method (requires method col in CSVs)')
    ap.add_argument('--comms-split-by-condition', action='store_true', help='Split graphs per condition/label (requires label/condition col in CSVs)')

    # Auto comms aggregation across experiments
    ap.add_argument('--auto-comms', action='store_true', help='Auto-discover edges/agg per experiment and aggregate across runs, then emit per-method × per-condition graphs.')

    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    runs = load_runs(args.run_index)

    f1_candidates = [c.strip() for c in args.f1_cols.split(',') if c.strip()]
    round_candidates = [c.strip() for c in args.round_cols.split(',') if c.strip()]
    bytes_candidates = [c.strip() for c in args.bytes_cols.split(',') if c.strip()]
    pheromone_candidates = [c.strip() for c in args.pheromone_cols.split(',') if c.strip()]
    base_methods = [m.strip() for m in args.base_methods.split(',') if m.strip()]

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Aggregations & plots (original pipeline) -----
    df_runs = aggregate_per_run(repo_root, runs, f1_candidates, round_candidates, bytes_candidates, pheromone_candidates)
    df_runs.to_csv(out_dir / 'runs_aggregated.csv', index=False)

    df_stats = aggregate_across_seeds(df_runs, norm_ref_method=args.norm_ref)
    df_stats.to_csv(out_dir / 'roundwise_stats_plus.csv', index=False)

    try:
        cond_counts = df_stats['label'].map(_cond_key).value_counts().to_dict()
        print(f"[INFO] Condition buckets present: {cond_counts}")
    except Exception:
        pass

    plot_f1_rounds_with_ci(df_stats, out_dir)
    plot_f1_vs_norm_comm_refined(df_stats, out_dir / 'f1_vs_normalized_communication.png')
    plot_pheromone_decay_refined(df_runs, out_dir / 'pheromone_decay_timeseries.png')
    plot_f1_boxplots(df_runs, out_dir)
    plot_asilo_vs_bases_by_condition(df_stats, out_dir, base_methods=base_methods)

    # ----- Manual comms (if provided) -----
    if args.edges:
        comms_out_dir = Path(args.comms_out) if args.comms_out else (out_dir / "comms")
        run_comms_visualization(
            edges_path=Path(args.edges),
            agg_path=(Path(args.agg) if args.agg else None),
            out_dir=comms_out_dir,
            layout=args.comms_layout,
            min_bytes=args.comms_min_bytes,
            min_count=args.comms_min_count,
            min_accepts=args.comms_min_accepts,
            seed=args.comms_seed,
            split_by_method=bool(args.comms_split_by_method),
            split_by_condition=bool(args.comms_split_by_condition),
        )

    # ----- Auto-discover & aggregate comms across experiments -----
    if args.auto_comms:
        E_all, A_all = build_combined_comms_from_runs(repo_root, runs)
        auto_dir = (Path(args.comms_out) if args.comms_out else (out_dir / "comms_auto"))
        auto_dir.mkdir(parents=True, exist_ok=True)

        if E_all.empty:
            print("[WARN] Auto-comm: No edges discovered across experiments; skipping.")
        else:
            # persist combined CSVs for reproducibility
            edges_all_path = auto_dir / "edges_all.csv"
            E_all.to_csv(edges_all_path, index=False)
            print(f"[OK] Wrote {edges_all_path}")
            agg_all_path = None
            if not A_all.empty:
                agg_all_path = auto_dir / "agg_all.csv"
                A_all.to_csv(agg_all_path, index=False)
                print(f"[OK] Wrote {agg_all_path}")

            # Now produce per-method × per-condition graphs
            run_comms_visualization(
                edges_path=edges_all_path,
                agg_path=agg_all_path,
                out_dir=auto_dir,
                layout=args.comms_layout,
                min_bytes=args.comms_min_bytes,
                min_count=args.comms_min_count,
                min_accepts=args.comms_min_accepts,
                seed=args.comms_seed,
                split_by_method=True,
                split_by_condition=True,
            )

if __name__ == '__main__':
    main()
