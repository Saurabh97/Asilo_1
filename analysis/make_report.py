#!/usr/bin/env python
import base64, io, argparse, re
from pathlib import Path
import pandas as pd
import numpy as np  # added

SECS = [
    ("f1_mean_ci_timeseries_ASILO","f1_mean_ci_timeseries_ASILO.png"),
    ("f1_mean_ci_timeseries_FedAvg","f1_mean_ci_timeseries_FedAvg.png"),
    ("f1_mean_ci_timeseries_DeceFL","f1_mean_ci_timeseries_DeceFL.png"),
    ("f1_mean_ci_timeseries_DFedSAM","f1_mean_ci_timeseries_DFedSAM.png"),
    ("f1_asilo_vs_bases_homogeneous","f1_asilo_vs_bases_homogeneous.png"),
    ("f1_asilo_vs_bases_fast50","f1_asilo_vs_bases_fast50.png"),
    ("f1_asilo_vs_bases_fast25","f1_asilo_vs_bases_fast25.png"),
    ("F1 across rounds", "f1_mean_ci_timeseries.png"),
    ("F1 vs Normalized Communication", "f1_vs_normalized_communication.png"),
    ("Pheromone decay", "pheromone_decay_timeseries.png"),
]

METHODS = ["ASILO", "FedAvg", "DFedSAM", "DeceFL"]
COND_KEYS = ["homogeneous", "heterogeneous-fast25", "heterogeneous-fast40", "heterogeneous-fast50"]
COND_DISPLAY = {
    "homogeneous": "homogeneous",
    "heterogeneous-fast25": "fast25",
    "heterogeneous-fast40": "fast40",
    "heterogeneous-fast50": "fast50",
}

def _img_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def _pick_comms_dir(adir: Path) -> Path | None:
    # prefer auto aggregation folder if present
    for sub in ("comms_auto", "comms"):
        p = adir / sub
        if p.exists() and p.is_dir():
            return p
    return None

def _label_display_from_filepiece(label_piece: str) -> str:
    # label piece will be like 'heterogeneous-fast25' or 'homogeneous'
    k = label_piece.strip().lower()
    return COND_DISPLAY.get(k, k)

def _gather_comms_images(comms_dir: Path):
    """
    Return dict:
      {
        'communication': {(method, cond_key): Path, ...},
        'effective': {(method, cond_key): Path, ...}
      }
    Only entries that exist on disk are included.
    """
    comm, eff = {}, {}
    for m in METHODS:
        for ck in COND_KEYS:
            cg = comms_dir / f"communication_graph_{m}_{ck}.png"
            eg = comms_dir / f"effective_contribution_graph_{m}_{ck}.png"
            if cg.exists():
                comm[(m, ck)] = cg
            if eg.exists():
                eff[(m, ck)] = eg
    return {"communication": comm, "effective": eff}

def _summarize_edge_stats(comms_dir: Path) -> pd.DataFrame:
    """
    Read edge_stats_*.csv files and compute totals per method × condition.
    """
    rows = []
    for p in comms_dir.glob("edge_stats_*.csv"):
        # filename pattern: edge_stats_<METHOD>_<COND>.csv
        stem = p.stem  # edge_stats_<...>
        rest = stem[len("edge_stats_"):]
        # split into method and label (method has no underscores in our setups)
        if "_" in rest:
            method, label_piece = rest.split("_", 1)
        else:
            # fallback if unexpected
            method, label_piece = rest, "homogeneous"
        label_key = label_piece
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception:
            continue
        # sum robustly
        for col in ["sent_bytes","received_bytes","buffered_bytes","dropped_bytes","buffered_lowcos_bytes"]:
            if col not in df.columns:
                df[col] = 0
        for col in ["sent_count","received_count","buffered_count","drop_cos_count","buffered_lowcos_count"]:
            if col not in df.columns:
                df[col] = 0
        total_bytes = (df["sent_bytes"] + df["received_bytes"] + df["buffered_bytes"] +
                       df["dropped_bytes"] + df["buffered_lowcos_bytes"]).sum()
        total_edges = len(df)
        sent_edges = int(df["sent_count"].sum())
        recv_edges = int(df["received_count"].sum())
        rows.append({
            "method": method,
            "condition": _label_display_from_filepiece(label_key),
            "edges": int(total_edges),
            "sent_events": sent_edges,
            "recv_events": recv_edges,
            "total_bytes": int(total_bytes),
        })
    if not rows:
        return pd.DataFrame(columns=["method","condition","edges","sent_events","recv_events","total_bytes"])
    out = pd.DataFrame(rows)
    # Order nicely
    cat_cond = ["homogeneous","fast25","fast40","fast50"]
    out["condition"] = pd.Categorical(out["condition"], categories=cat_cond, ordered=True)
    out = out.sort_values(["method","condition"])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", required=True, help="outputs/analysis folder")
    ap.add_argument("--out", default="report.html")
    args = ap.parse_args()

    adir = Path(args.analysis_dir)
    out = Path(args.out)

    # ---------- Load & normalize labels ----------
    df = pd.read_csv(adir / "roundwise_stats_plus.csv")

    # Homogeneous came through as NaN in 'label'. Normalize to human-friendly keys.
    def _label_display(v):
        if pd.isna(v) or str(v).strip() == "" or str(v).lower() in ("none", "nan"):
            return "homogeneous"
        s = str(v).strip().lower()
        s = s.replace("_", "").replace("-", "").replace(" ", "")
        if "fast25" in s or (("fast" in s) and ("25" in s)):
            return "fast25"
        if "fast50" in s or (("fast" in s) and ("50" in s)):
            return "fast50"
        if "fast40" in s or (("fast" in s) and ("40" in s)):
            return "fast40"
        return str(v)

    df["label_norm"] = df["label"].map(_label_display)

    # ---------- Build final-round snapshot (include homogeneous now) ----------
    last = (
        df.groupby(["block", "label_norm", "method"], as_index=False)
          .apply(lambda g: g.loc[g["round"].idxmax()])
          .reset_index(drop=True)
    )

    cat_order = ["homogeneous", "fast25", "fast40", "fast50"]
    last["label_norm"] = pd.Categorical(last["label_norm"], categories=cat_order, ordered=True)

    tbl = last[[
        "block","label_norm","method","round","f1_mean","f1_std","ci_low","ci_high","bytes_cum_mean"
    ]].sort_values(["block","label_norm","method"]).rename(columns={"label_norm":"label"})

    table_html = tbl.to_html(index=False, float_format=lambda x: f"{x:.4f}")

    # ---------- Embed known figures if present + any boxplots ----------
    items = []
    for title, fname in SECS:
        f = adir / fname
        if f.exists():
            items.append(
                f"<h2>{title}</h2><img style='max-width:100%;' src='data:image/png;base64,{_img_b64(f)}'/>"
            )

    for f in sorted(adir.glob("boxplot_f1_*.png")):
        items.append(
            f"<h3>{f.name.replace('_',' ')}</h3><img style='max-width:100%;' src='data:image/png;base64,{_img_b64(f)}'/>"
        )

    # ---------- Communication section (auto-discovers comms outputs) ----------
    comms_dir = _pick_comms_dir(adir)
    comm_html = ""
    if comms_dir:
        imgs = _gather_comms_images(comms_dir)
        # Summary table from edge_stats_*.csv
        summary_df = _summarize_edge_stats(comms_dir)
        if not summary_df.empty:
            comm_html += "<h2>Communication summary (per method × condition)</h2>"
            comm_html += summary_df.to_html(index=False)

        # Communication graphs
        if imgs["communication"]:
            comm_html += "<h2>Communication Graphs</h2>"
            for m in METHODS:
                # Collect existing conditions for this method
                pairs = [(m, ck) for ck in COND_KEYS if (m, ck) in imgs["communication"]]
                if not pairs:
                    continue
                comm_html += f"<h3>{m}</h3><div style='display:flex;flex-wrap:wrap;gap:12px;'>"
                for _, ck in pairs:
                    p = imgs["communication"][(m, ck)]
                    cap = f"{m} — {COND_DISPLAY.get(ck, ck)}"
                    comm_html += f"<figure style='margin:0;'><img style='max-width:460px;' src='data:image/png;base64,{_img_b64(p)}'/><figcaption style='text-align:center;font-size:12px;color:#666'>{cap}</figcaption></figure>"
                comm_html += "</div>"

        # Effective contribution graphs
        if imgs["effective"]:
            comm_html += "<h2>Effective Contribution Graphs</h2>"
            for m in METHODS:
                pairs = [(m, ck) for ck in COND_KEYS if (m, ck) in imgs["effective"]]
                if not pairs:
                    continue
                comm_html += f"<h3>{m}</h3><div style='display:flex;flex-wrap:wrap;gap:12px;'>"
                for _, ck in pairs:
                    p = imgs["effective"][(m, ck)]
                    cap = f"{m} — {COND_DISPLAY.get(ck, ck)}"
                    comm_html += f"<figure style='margin:0;'><img style='max-width:460px;' src='data:image/png;base64,{_img_b64(p)}'/><figcaption style='text-align:center;font-size:12px;color:#666'>{cap}</figcaption></figure>"
                comm_html += "</div>"

    # ---------- Build and write HTML ----------
    html = f"""
    <html>
    <head>
    <meta charset="utf-8">
    <title>ASILO Robustness — Report</title>
    <style>
      body {{ font-family: Segoe UI, Arial, sans-serif; margin: 20px; }}
      table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
      th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
      th {{ background: #f3f3f3; }}
      td:nth-child(1), td:nth-child(2), td:nth-child(3) {{ text-align: left; }}
      h1,h2,h3 {{ margin: 18px 0 8px; }}
      figure {{ display:inline-block; }}
      figcaption {{ margin-top: 4px; }}
    </style>
    </head>
    <body>
    <h1>ASILO Robustness — Summary</h1>
    <h2>Final-round snapshot (mean ± 95% CI)</h2>
    {table_html}
    {''.join(items)}
    {comm_html}
    </body></html>
    """

    out.write_text(html, encoding="utf-8")
    print(f"[OK] Wrote report: {out.resolve()}")

if __name__ == "__main__":
    main()
