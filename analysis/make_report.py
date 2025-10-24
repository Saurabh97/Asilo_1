#!/usr/bin/env python
import base64, io, argparse
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

def _img_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

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
    # Keep fast25/fast50 as-is; rename if you prefer.
    def _label_display(v):
        if pd.isna(v) or str(v).strip() == "" or str(v).lower() in ("none", "nan"):
            return "homogeneous"
        s = str(v).strip().lower()
        # unify variants if needed
        s = s.replace("_", "").replace("-", "").replace(" ", "")
        if "fast25" in s or (("fast" in s) and ("25" in s)):
            return "fast25"
        if "fast50" in s or (("fast" in s) and ("50" in s)):
            return "fast50"
        # fallback to original text (unmodified) if it's something else
        return str(v)

    df["label_norm"] = df["label"].map(_label_display)

    # ---------- Build final-round snapshot (include homogeneous now) ----------
    # Use label_norm for grouping so NaN won't be dropped
    last = (
        df.groupby(["block", "label_norm", "method"], as_index=False)
          .apply(lambda g: g.loc[g["round"].idxmax()])
          .reset_index(drop=True)
    )

    # Nicely ordered categories for label column
    cat_order = ["homogeneous", "fast25", "fast50"]
    last["label_norm"] = pd.Categorical(last["label_norm"], categories=cat_order, ordered=True)

    tbl = last[[
        "block","label_norm","method","round","f1_mean","f1_std","ci_low","ci_high","bytes_cum_mean"
    ]].sort_values(["block","label_norm","method"]).rename(columns={"label_norm":"label"})

    # Convert table to HTML
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
    </style>
    </head>
    <body>
    <h1>ASILO Robustness — Summary</h1>
    <h2>Final-round snapshot (mean ± 95% CI)</h2>
    {table_html}
    {''.join(items)}
    </body></html>
    """

    out.write_text(html, encoding="utf-8")
    print(f"[OK] Wrote report: {out.resolve()}")

if __name__ == "__main__":
    main()
