"""
ASILO — Wearables Demo (v0.1)
=============================

Goal
----
End-to-end demo of the **original proposal**: fully decentralized FL with
**pheromone** (utility) driven neighbor selection, **capability-aware** policies,
and **sparse delta** sharing — applied to a *wearables stress detection* case.

What works in v0.1
------------------
- Pure P2P (no central router in hot path). Each agent runs its own TCP server.
- Pheromone logic tied to *learning utility* (ΔF1/AUPRC proxy on validation).
- Two delta strategies implemented:
  1) `head_delta` — for scikit-learn LogisticRegression (coefficients + bias).
  2) `proto_delta` — class prototypes for embedding spaces or tabular features.
- Capability-aware policies: local steps, top-k neighbors, byte budgets.
- Config-driven launcher spawning N agents on one machine.

What this demo assumes
----------------------
- You preprocess WESAD wrist data into a CSV with feature columns + label.
  Path convention: `data/processed/WESAD_wrist/<SUBJECT>.csv`.
  Each CSV should have columns: `ts,label,feat_0,feat_1,...,feat_D`.
- If you don't have WESAD ready yet, you can still run agents on synthetic data
  via `--synthetic` flag inside `experiments/run.py`.

Quick start
-----------
1) Create a venv; install deps: `pip install pyyaml scikit-learn uvloop` (uvloop optional on Linux/Mac).
2) Preprocess WESAD into `data/processed/WESAD_wrist/` (or toggle synthetic).
3) Run: `python experiments/run.py experiments/configs/wesad_case.yaml`
4) Watch logs — each agent prints pheromone, utility, bytes/round.

Next steps
----------
- Swap LogisticRegression → 1D-TCN for raw windows (add `vision_torch.py`).
- Add Streamlit dashboard to visualize pheromone heatmap + neighbor graph.
- Plug real PPG/EDA feature extractor and proper AUPRC computation.

"""
