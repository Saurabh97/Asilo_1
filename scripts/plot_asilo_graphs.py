#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ASILO comms visualizer
Reads edges.csv / agg_decisions.csv (and optional pheromone.csv) and produces:
  - communication_graph.png
  - effective_contribution_graph.png
  - edge_stats.csv
  - effective_stats.csv

Edge semantics (expected columns, case-insensitive):
  edges.csv:        t, src, dst, bytes, kind, action, cos, reason
                    action ∈ {sent, received, buffered, drop_cos, buffered_low_cos (optional)}
  agg_decisions.csv: t, agent, from, bytes, decision, cos_thresh, mode, trim_p, rolled_back
                    decision ∈ {accepted_in_agg, dropped_in_agg}

Usage:
  python plot_asilo_graphs.py --edges logs/exp/edges.csv --agg logs/exp/agg_decisions.csv --outdir ./asilo_plots
"""

import argparse, math, os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# ---------- IO helpers ----------
def _load_csv(path: Path):
    if path is None:
        return None
    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    except Exception as e:
        print(f"[ERROR] Failed reading {path}: {e}")
        return None


def _ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


# ---------- Aggregations ----------
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
    sent_df = E[E["action"] == "sent"].copy() if "action" in E.columns else E.iloc[0:0].copy()
    recv_df = E[E["action"] == "received"].copy() if "action" in E.columns else E.iloc[0:0].copy()
    buff_df = E[E["action"] == "buffered"].copy() if "action" in E.columns else E.iloc[0:0].copy()
    drop_df = E[E["action"] == "drop_cos"].copy() if "action" in E.columns else E.iloc[0:0].copy()
    lowb_df = E[E["action"] == "buffered_low_cos"].copy() if ("action" in E.columns and "buffered_low_cos" in E["action"].unique()) else E.iloc[0:0].copy()

    sent_cnt = _agg_edge_counts(sent_df, "sent_count")
    recv_cnt = _agg_edge_counts(recv_df, "received_count")
    buff_cnt = _agg_edge_counts(buff_df, "buffered_count")
    drop_cnt = _agg_edge_counts(drop_df, "drop_cos_count")
    lowb_cnt = _agg_edge_counts(lowb_df, "buffered_lowcos_count")

    sent_bytes = _agg_edge_sum(sent_df, "bytes", "sent_bytes")
    buff_bytes = _agg_edge_sum(buff_df, "bytes", "buffered_bytes")
    drop_bytes = _agg_edge_sum(drop_df, "bytes", "dropped_bytes")
    lowb_bytes = _agg_edge_sum(lowb_df, "bytes", "buffered_lowcos_bytes")

    edge_stats = (
        sent_cnt.merge(recv_cnt, on=["src", "dst"], how="outer")
        .merge(buff_cnt, on=["src", "dst"], how="outer")
        .merge(drop_cnt, on=["src", "dst"], how="outer")
        .merge(lowb_cnt, on=["src", "dst"], how="outer")
        .merge(sent_bytes, on=["src", "dst"], how="outer")
        .merge(buff_bytes, on=["src", "dst"], how="outer")
        .merge(drop_bytes, on=["src", "dst"], how="outer")
        .merge(lowb_bytes, on=["src", "dst"], how="outer")
    ).fillna(0)

    # types
    for c in ["sent_count", "received_count", "buffered_count", "drop_cos_count", "buffered_lowcos_count"]:
        if c in edge_stats.columns:
            edge_stats[c] = edge_stats[c].astype(int)
    for c in ["sent_bytes", "buffered_bytes", "dropped_bytes", "buffered_lowcos_bytes"]:
        if c in edge_stats.columns:
            edge_stats[c] = edge_stats[c].astype(int)

    return edge_stats


def build_effective_stats(A: pd.DataFrame) -> pd.DataFrame:
    if A is None or A.empty:
        return pd.DataFrame(columns=["src", "dst", "accepted_count", "dropped_in_agg_count"])

    A = A.copy()
    A.columns = [c.strip().lower() for c in A.columns]
    A.rename(columns={"from": "src", "agent": "dst"}, inplace=True)
    acc = A[A["decision"] == "accepted_in_agg"].groupby(["src", "dst"], as_index=False).size()
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


# ---------- Plotting ----------
def _layout_positions(G, layout: str, seed: int = 42):
    layout = (layout or "spring").lower()
    if layout == "kamada":
        return nx.kamada_kawai_layout(G)
    if layout == "circular":
        return nx.circular_layout(G)
    if layout == "random":
        return nx.random_layout(G, seed=seed)
    # default
    return nx.spring_layout(G, seed=seed)


def plot_communication_graph(edge_stats: pd.DataFrame, out_png: Path, layout: str = "spring",
                             min_bytes: int = 0, min_count: int = 0, seed: int = 42):
    # filter tiny edges
    es = edge_stats.copy()
    cnt_ok = (es[["sent_count", "buffered_count", "drop_cos_count"]].sum(axis=1) >= min_count)
    byt_ok = (es.get("sent_bytes", 0) + es.get("buffered_bytes", 0) >= min_bytes)
    es = es[cnt_ok & byt_ok]

    nodes = sorted(set(es["src"]).union(set(es["dst"])))
    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    for _, r in es.iterrows():
        src, dst = r["src"], r["dst"]
        sent_b = int(r.get("sent_bytes", 0))
        buff_c = int(r.get("buffered_count", 0))
        drop_c = int(r.get("drop_cos_count", 0))
        width = 1.0 + math.log1p(sent_b) / 3.0 if sent_b > 0 else 1.0
        style = "dashed" if drop_c > buff_c else "solid"
        G.add_edge(src, dst, width=width, style=style)

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

    plt.title("Communication Graph (solid: buffered ≥ drops; dashed: drops > buffered)\nEdge width ∝ sent bytes")
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
    es = es[es["accepted_count"] >= min_accepts]

    G = nx.DiGraph()
    nodes = sorted(set(es["src"]).union(set(es["dst"])))
    G.add_nodes_from(nodes)

    for _, r in es.iterrows():
        src, dst = r["src"], r["dst"]
        acc = int(r.get("accepted_count", 0))
        width = 1.0 + math.log1p(acc)
        G.add_edge(src, dst, width=width)

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


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Plot ASILO communication graphs from CSV logs.")
    ap.add_argument("--edges", required=True, type=Path, help="Path to edges.csv")
    ap.add_argument("--agg", required=False, type=Path, default=None, help="Path to agg_decisions.csv")
    ap.add_argument("--pheromone", required=False, type=Path, default=None, help="(Optional) Path to pheromone.csv")
    ap.add_argument("--outdir", required=False, type=Path, default=Path("./asilo_plots"), help="Output directory")
    ap.add_argument("--layout", choices=["spring", "kamada", "circular", "random"], default="spring", help="Graph layout")
    ap.add_argument("--min-bytes", type=int, default=0, help="Min total bytes to show an edge")
    ap.add_argument("--min-count", type=int, default=0, help="Min total event count to show an edge")
    ap.add_argument("--min-accepts", type=int, default=1, help="Min accepted_in_agg to render an edge in effective graph")
    ap.add_argument("--seed", type=int, default=42, help="Layout seed for reproducibility")
    args = ap.parse_args()

    E = _load_csv(args.edges)
    if E is None or E.empty:
        raise SystemExit("edges.csv not found or empty.")

    # Normalize types and required columns
    for col in ["src", "dst"]:
        if col not in E.columns:
            raise SystemExit(f"edges.csv is missing '{col}' column.")
    if "bytes" not in E.columns:
        raise SystemExit("edges.csv is missing 'bytes' column.")
    if "action" not in E.columns:
        print("[WARN] edges.csv has no 'action' column; defaulting to 'sent' for all.")
        E["action"] = "sent"

    _ensure_numeric(E, ["t", "bytes", "cos"])
    edge_stats = build_edge_stats(E)

    # Save edge_stats
    args.outdir.mkdir(parents=True, exist_ok=True)
    edge_stats_path = args.outdir / "edge_stats.csv"
    edge_stats.to_csv(edge_stats_path, index=False)
    print(f"[OK] Wrote {edge_stats_path}")

    # Plot communication graph
    plot_communication_graph(
        edge_stats=edge_stats,
        out_png=args.outdir / "communication_graph.png",
        layout=args.layout,
        min_bytes=args.min_bytes,
        min_count=args.min_count,
        seed=args.seed,
    )

    # Effective contribution graph (if agg provided)
    eff_stats = pd.DataFrame(columns=["src", "dst", "accepted_count", "dropped_in_agg_count"])
    if args.agg is not None:
        A = _load_csv(args.agg)
        if A is not None and not A.empty:
            _ensure_numeric(A, ["t", "bytes", "cos_thresh", "trim_p", "rolled_back"])
            eff_stats = build_effective_stats(A)
        else:
            print("[WARN] agg_decisions.csv empty or not found — skipping effective graph.")

    eff_stats_path = args.outdir / "effective_stats.csv"
    eff_stats.to_csv(eff_stats_path, index=False)
    print(f"[OK] Wrote {eff_stats_path}")

    plot_effective_graph(
        eff_stats=eff_stats,
        out_png=args.outdir / "effective_contribution_graph.png",
        layout=args.layout,
        min_accepts=args.min_accepts,
        seed=max(1, args.seed - 35),
    )


if __name__ == "__main__":
    main()
