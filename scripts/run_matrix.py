#!/usr/bin/env python
# Parallel experiment runner with unique ports + resume/skip-completed + aggregation
# Usage:
#   python scripts/run_matrix.py --repo-root C:\path\to\Asilo_1 --matrix experiments\configs\robustness\matrix.yaml \
#       --python python --max-parallel 6 --max-parallel-gpu 1 --aggregate
#
#   (optional) control port pool:
#       --port-base-start 30000 --port-stride 100 --port-blocks 300
#
#   (resume default ON) to force rerun all:
#       --no-skip-completed

import os, sys, argparse, subprocess, time, json, csv, io, signal, hashlib
from pathlib import Path

# Robust streams on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    import yaml
except Exception:
    print("PyYAML is required. pip install pyyaml", file=sys.stderr)
    raise

# (script, default_exp_name, uses_gpu?)
METHOD_BIN = {
    "ASILO":   ("experiments/run.py",        "asilo",  True),
    "FedAvg":  ("experiments/run_fedavg.py", "fedavg", False),
    "DFedSAM": ("experiments/run_dfedsam.py","dfedsam",False),
    "DeceFL":  ("experiments/run_decefl.py", "decefl", False),
}

def normpath(p): return str(Path(p))
def ensure_exists(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing path: {path}")

def build_exp_name(block, label, method, seed):
    parts = [block]
    if label: parts.append(label)
    parts.append(method.lower())
    parts.append(f"seed{seed}")
    return "_".join(parts)

def discover_log_files(repo_root, exp_name):
    logs_dir = Path(repo_root) / "logs" / exp_name
    return list(logs_dir.glob("*.csv"))

def parse_csv_numeric(path):
    import pandas as pd
    try:
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        round_col = cols.get("round") or cols.get("r") or list(df.columns)[0]
        f1_col = cols.get("f1") or cols.get("f1_val") or cols.get("macro_f1") or None
        return df, round_col, f1_col
    except Exception:
        rounds, f1s = [], []
        with open(path, newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            hdr = [h.strip() for h in reader.fieldnames] if reader.fieldnames else []
            lc_hdr = [h.lower() for h in hdr]
            round_idx = lc_hdr.index("round") if "round" in lc_hdr else 0
            f1_idx = None
            for name in ["f1","f1_val","macro_f1"]:
                if name in lc_hdr: f1_idx = lc_hdr.index(name); break
            for row in reader:
                row_vals = [row.get(h, "") for h in hdr]
                try:
                    r = int(row_vals[round_idx])
                    f1 = float(row_vals[f1_idx]) if f1_idx is not None else None
                    rounds.append(r); f1s.append(f1)
                except Exception:
                    continue
        return {"rounds": rounds, "f1": f1s}, "rounds", "f1"

def load_matrix(matrix_yaml):
    with open(matrix_yaml, "r", encoding="utf-8") as f:
        M = yaml.safe_load(f)
    blocks = M.get("matrix", {}).get("blocks", [])
    if not blocks:
        raise ValueError("No blocks defined in matrix.yaml")
    return blocks

def build_tasks(repo_root, blocks, rounds_default=150, pybin="python"):
    tasks = []  # dicts: cmd, exp_name, method, gpu, block, label, seed, rounds
    for block in blocks:
        block_name = block.get("name", "block")
        rounds = int(block.get("rounds", rounds_default))
        methods = block.get("methods", [])
        seeds = block.get("seeds", [])
        hetero_configs = block.get("hetero_configs", None)
        config_map = block.get("config_map", None)

        configs_enum = []
        if hetero_configs:
            for h in hetero_configs:
                configs_enum.append((h.get("label","hetero"), h.get("config_map", {})))
        elif config_map:
            configs_enum.append(("", config_map))
        else:
            raise ValueError(f"Block {block_name} has neither config_map nor hetero_configs.")

        for label, cmap in configs_enum:
            for method in methods:
                bin_rel, _, uses_gpu = METHOD_BIN[method]
                runner = Path(repo_root) / bin_rel
                ensure_exists(runner)
                cfg_path = cmap.get(method)
                if not cfg_path:
                    raise ValueError(f"No config path for method {method} in block {block_name}/{label}")
                cfg_abs = Path(repo_root) / cfg_path
                ensure_exists(cfg_abs)

                for seed in seeds:
                    exp_name = build_exp_name(block_name, label, method, seed)
                    cmd = [
                        pybin, normpath(runner),
                        normpath(cfg_abs),
                        "--seed", str(seed),
                        "--rounds", str(rounds),
                        "--exp", exp_name
                    ]
                    tasks.append({
                        "cmd": cmd,
                        "exp_name": exp_name,
                        "method": method,
                        "gpu": uses_gpu,
                        "block": block_name,
                        "label": label,
                        "seed": seed,
                        "rounds": rounds
                    })
    return tasks

# -------------------- COMPLETION / RESUME --------------------
def done_marker_path(repo_root: Path, exp_name: str) -> Path:
    return repo_root / "outputs" / "completed" / f"{exp_name}.done.json"

def write_done_marker(repo_root: Path, task: dict, status: str = "ok"):
    p = done_marker_path(repo_root, task["exp_name"])
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "exp_name": task["exp_name"],
        "block": task["block"],
        "label": task["label"],
        "method": task["method"],
        "seed": task["seed"],
        "rounds": task.get("rounds"),
        "status": status,
        "ts": int(time.time())
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def has_done_marker(repo_root: Path, exp_name: str) -> bool:
    return done_marker_path(repo_root, exp_name).exists()

def logs_reach_rounds(repo_root: Path, exp_name: str, expected: int) -> bool:
    """Consider complete if any agent log shows max round >= expected-1 or
       if unique rounds across logs >= expected. Robust to 0/1-based."""
    files = discover_log_files(repo_root, exp_name)
    if not files:
        return False
    max_r = -1
    rounds_seen = set()
    for fp in files:
        df_or_dict, rcol, _ = parse_csv_numeric(fp)
        try:
            if isinstance(df_or_dict, dict):
                rounds = df_or_dict.get("rounds", [])
            else:
                rounds = list(df_or_dict[rcol].dropna().astype(int).values)
        except Exception:
            continue
        if not rounds:
            continue
        max_r = max(max_r, max(rounds))
        rounds_seen.update(rounds)
    if max_r >= expected - 1:
        return True
    if len(rounds_seen) >= expected:
        return True
    return False

def is_task_completed(repo_root: Path, task: dict) -> bool:
    # Marker wins
    if has_done_marker(repo_root, task["exp_name"]):
        return True
    # Otherwise inspect logs
    return logs_reach_rounds(repo_root, task["exp_name"], task.get("rounds", 0))

# -------------------- PARALLEL EXECUTION WITH UNIQUE PORTS --------------------
def _stable_block_from_name(name: str, max_blocks: int) -> int:
    """Stable 0..max_blocks-1 index from a string (md5-based)."""
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % max_blocks

def run_parallel(repo_root, tasks, max_parallel=4, max_parallel_gpu=1,
                 port_base_start=30000, port_stride=100, max_blocks=500,
                 skip_completed=True, env_base=None):
    """
    Launch tasks in parallel with caps on total and GPU-using jobs.
    Unique port block per run via env:
      PORT_BASE=<port_base_start + block_idx*port_stride>, PORT_STRIDE=<port_stride>
    Skips tasks that are already complete (marker or logs reach expected rounds).
    """
    repo_root = Path(repo_root)
    outputs_dir = repo_root / "outputs"
    logs_dir = outputs_dir / "run_stdout"
    logs_dir.mkdir(parents=True, exist_ok=True)

    running, completed, skipped = [], [], []
    gpu_in_use = 0
    env_base = env_base or os.environ.copy()

    # Clamp blocks so PORT_BASE never exceeds 65535
    max_fit = max(1, (65535 - port_base_start - port_stride) // port_stride + 1)
    eff_blocks = min(max_blocks, max_fit)

    def can_start(t):
        if len(running) >= max_parallel:
            return False
        if t["gpu"] and gpu_in_use >= max_parallel_gpu:
            return False
        return True

    def pop_finished():
        nonlocal gpu_in_use
        still = []
        for r in running:
            proc = r["proc"]
            if proc.poll() is None:
                still.append(r)
            else:
                if r["gpu"]:
                    gpu_in_use = max(0, gpu_in_use - 1)
                code = proc.returncode
                # close files
                try:
                    r["stdout"].close(); r["stderr"].close()
                except Exception:
                    pass
                if code == 0:
                    write_done_marker(repo_root, r, status="ok")
                    completed.append(r)
                else:
                    # no marker; keep as failed
                    print(f"[WARN] {r['exp_name']} exited with {code}", file=sys.stderr)
                # continue loop
        running[:] = still

    # Graceful Ctrl+C
    def _sigint(sig, frame):
        print("\n[INFO] Ctrl+C received. Terminating running jobs...", file=sys.stderr)
        for r in running:
            try:
                r["proc"].terminate()
            except Exception:
                pass
        sys.exit(1)
    signal.signal(signal.SIGINT, _sigint)

    q = list(tasks)
    while q or running:
        # Start as many as possible
        started_any = False
        i = 0
        while i < len(q):
            t = q[i]

            # Skip if already complete
            if skip_completed and is_task_completed(repo_root, t):
                print(f"[SKIP completed] {t['exp_name']}")
                skipped.append(t)
                q.pop(i)
                continue

            if not can_start(t):
                i += 1
                continue

            # per-run stdout/stderr
            out_path = logs_dir / f"{t['exp_name']}.out"
            err_path = logs_dir / f"{t['exp_name']}.err"
            fout = open(out_path, "w", encoding="utf-8", errors="replace")
            ferr = open(err_path, "w", encoding="utf-8", errors="replace")

            # Build per-run env and unique port block
            env = env_base.copy()
            block_idx = _stable_block_from_name(t["exp_name"], max_blocks=eff_blocks)
            port_base = port_base_start + block_idx * port_stride
            if not (0 <= port_base <= 65535 - port_stride):
                # final guard
                port_base = port_base_start
            env["PORT_BASE"] = str(port_base)
            env["PORT_STRIDE"] = str(port_stride)
            # Optional: pin GPU
            # env["CUDA_VISIBLE_DEVICES"] = "0"

            print(f">>> {' '.join(t['cmd'])}   [PORT_BASE={port_base} STRIDE={port_stride}]")
            proc = subprocess.Popen(t["cmd"], cwd=repo_root, env=env, stdout=fout, stderr=ferr)
            running.append({**t, "proc": proc, "stdout": fout, "stderr": ferr})
            if t["gpu"]:
                gpu_in_use += 1
            started_any = True
            q.pop(i)  # remove task
        if not started_any:
            time.sleep(0.5)
        pop_finished()

    print(f"[OK] Completed {len(completed)} runs. Skipped {len(skipped)} already-done runs.")

    # Build index from completed + skipped (so aggregator can pick up existing logs)
    results_index = []
    for r in completed + skipped:
        results_index.append({
            "block": r["block"], "label": r["label"], "method": r["method"],
            "seed": r["seed"], "rounds": r.get("rounds"), "exp_name": r["exp_name"]
        })

    out_idx = Path(repo_root) / "outputs" / f"run_index_{int(time.time())}.json"
    out_idx.parent.mkdir(parents=True, exist_ok=True)
    with open(out_idx, "w", encoding="utf-8") as f:
        json.dump(results_index, f, indent=2)
    print(f"[OK] Wrote run index: {out_idx}")
    return out_idx

# -------------------- Aggregation --------------------
def aggregate_roundwise(repo_root, run_index_json, out_csv):
    import pandas as pd
    import numpy as np
    with open(run_index_json, "r", encoding="utf-8") as f:
        runs = json.load(f)

    rows = []
    for r in runs:
        exp = r["exp_name"]
        files = discover_log_files(repo_root, exp)
        if not files:
            print(f"[WARN] No logs found for {exp}", file=sys.stderr)
            continue
        per_agent = []
        for fp in files:
            df_or_dict, rcol, fcol = parse_csv_numeric(fp)
            if isinstance(df_or_dict, dict):
                import pandas as pd
                df_tmp = pd.DataFrame({"round": df_or_dict["rounds"], "f1": df_or_dict["f1"]})
                per_agent.append(df_tmp)
            else:
                df = df_or_dict
                if fcol is None:
                    continue
                per_agent.append(df[[rcol, fcol]].rename(columns={rcol: "round", fcol: "f1"}))

        if not per_agent:
            continue

        df_all = None
        for item in per_agent:
            df_all = item if df_all is None else df_all.merge(item, on="round", how="outer")

        if df_all is None or df_all.empty:
            continue

        f_cols = [c for c in df_all.columns if c != "round"]
        df_all["f1_mean_run"] = df_all[f_cols].mean(axis=1, skipna=True)

        for _, row in df_all.iterrows():
            if row["f1_mean_run"] == row["f1_mean_run"]:
                rows.append({
                    "block": r.get("block"), "label": r.get("label"), "method": r.get("method"),
                    "seed": r.get("seed"), "round": int(row["round"]), "f1_run": float(row["f1_mean_run"])
                })

    if not rows:
        print("[WARN] No data to aggregate.", file=sys.stderr)
        return

    import pandas as pd
    import numpy as np
    df = pd.DataFrame(rows).dropna(subset=["f1_run"])
    stats = []
    for (b, l, m, rr), grp in df.groupby(["block","label","method","round"]):
        vals = grp["f1_run"].values
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        sem = sd / (len(vals) ** 0.5) if len(vals) > 1 else 0.0
        ci_low = mu - 1.96 * sem
        ci_high = mu + 1.96 * sem
        stats.append({"block":b,"label":l,"method":m,"round":rr,
                      "f1_mean":mu,"f1_std":sd,"f1_sem":sem,"ci_low":ci_low,"ci_high":ci_high,"n":len(vals)})
    df_stats = pd.DataFrame(stats).sort_values(["block","label","method","round"])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_stats.to_csv(out_csv, index=False)
    print(f"[OK] Wrote aggregated stats: {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True, help="Path to your Asilo_1 repository root")
    ap.add_argument("--matrix", required=True, help="Path to matrix YAML (see template)")
    ap.add_argument("--python", default="python", help="Python executable to use for sub-runs")
    ap.add_argument("--max-parallel", type=int, default=4, help="Max total concurrent runs")
    ap.add_argument("--max-parallel-gpu", type=int, default=1, help="Max concurrent GPU-using runs")
    # Unique port block controls
    ap.add_argument("--port-base-start", type=int, default=30000, help="Start of port pool for runs")
    ap.add_argument("--port-stride", type=int, default=100, help="Ports reserved per run")
    ap.add_argument("--port-blocks", type=int, default=500, help="Max distinct blocks (will be clamped to fit < 65536)")
    # Resume control
    ap.add_argument("--skip-completed", dest="skip_completed", action="store_true", default=True,
                    help="Skip runs that already appear completed (marker or logs)")
    ap.add_argument("--no-skip-completed", dest="skip_completed", action="store_false",
                    help="Disable skipping; rerun everything")
    ap.add_argument("--dry", action="store_true", help="Print commands without running")
    ap.add_argument("--aggregate", action="store_true", help="Aggregate round-wise F1 across seeds after running")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    ensure_exists(repo_root)
    matrix_yaml = Path(args.matrix).resolve()
    ensure_exists(matrix_yaml)

    blocks = load_matrix(matrix_yaml)
    tasks = build_tasks(repo_root, blocks, pybin=args.python)

    if args.dry:
        for t in tasks:
            print("[DRY RUN]", " ".join(t["cmd"]))
        print(f"[DRY] {len(tasks)} tasks.")
        return

    out_idx = run_parallel(
        repo_root,
        tasks,
        max_parallel=args.max_parallel,
        max_parallel_gpu=args.max_parallel_gpu,
        port_base_start=args.port_base_start,
        port_stride=args.port_stride,
        max_blocks=args.port_blocks,
        skip_completed=args.skip_completed
    )

    if args.aggregate:
        outputs_dir = repo_root / "outputs"
        outputs_dir.mkdir(exist_ok=True, parents=True)
        out_csv = outputs_dir / "aggregate_roundwise.csv"
        aggregate_roundwise(repo_root, out_idx, out_csv)

if __name__ == "__main__":
    main()
