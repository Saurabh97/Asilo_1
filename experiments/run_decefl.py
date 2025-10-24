# experiments/run_decefl.py
import os, sys, yaml, asyncio, io, random,re
from typing import List, Dict, Tuple

# repo roots
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# robust I/O
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
# Use Selector loop on Windows to avoid Proactor shutdown chatter
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def _win_suppress_reset(loop, context):
    exc = context.get("exception")
    if isinstance(exc, ConnectionResetError) and sys.platform.startswith("win"):
        # Suppress noisy "existing connection was forcibly closed" during shutdown
        return
    loop.default_exception_handler(context)
from Asilo_1.agents.decefl_agent import DeceFLAgent, DeceFLConfig
from Asilo_1.fl.trainers.tabular_sklearn import make_trainer_for_subject

def set_seed(seed: int | None):
    if seed is None: return
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np; np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

def build_symmetric_ring_peers(agents_cfg: List[dict], k_peers: int) -> Dict[str, Dict[str, Tuple[str,int]]]:
    """
    Build a symmetric K-regular (undirected) ring over all agents.
    For each agent i (in a stable order), choose floor(K/2) predecessors and ceil(K/2) successors.
    Each edge is reciprocal to equalize in- and out-degree.
    Returns: {agent_id: {peer_id: (host, port), ...}, ...}
    """
    k = max(1, int(k_peers))
    n = len(agents_cfg)
    ids = [a["id"] for a in agents_cfg]
    addr = {a["id"]: (a["host"], a["port"]) for a in agents_cfg}

    left = k // 2
    right = k - left

    # adjacency as sets to ensure reciprocity
    adj: Dict[str, set] = {a["id"]: set() for a in agents_cfg}
    for idx, me in enumerate(ids):
        # neighbors in ring (predecessors and successors)
        for j in range(1, left + 1):
            peer = ids[(idx - j) % n]
            adj[me].add(peer); adj[peer].add(me)
        for j in range(1, right + 1):
            peer = ids[(idx + j) % n]
            adj[me].add(peer); adj[peer].add(me)

    # finalize dict with (host, port)
    peers_map: Dict[str, Dict[str, Tuple[str,int]]] = {}
    for me, peers in adj.items():
        peers_map[me] = {pid: addr[pid] for pid in sorted(peers)}
    return peers_map

async def build_agents(cfg: dict, exp_name: str, rounds_override: int | None):
    agents_cfg = cfg["agents"]
    data_dir = cfg["data_dir"]

    # global knobs from YAML
    k_peers = int(cfg.get("decefl", {}).get("k_peers", 3))
    agg_mode = str(cfg.get("decefl", {}).get("agg_mode", "weighted"))
    round_time_s = float(cfg.get("round_time_s", 0.1))
    default_rounds = int(cfg.get("run_limits", {}).get("max_rounds", 200))

    # SYMMETRIC ring peers (no server bias)
    peers_map = build_symmetric_ring_peers(agents_cfg, k_peers=k_peers)

    agents: List[DeceFLAgent] = []
    for a in agents_cfg:
        role = "server" if a.get("is_server", False) else "client"
        dcfg = DeceFLConfig(
            agent_id=a["id"],
            host=a["host"],
            port=a["port"],
            model_id=a["model_id"],
            max_rounds=rounds_override if rounds_override is not None else a.get("max_rounds", default_rounds),
            is_server=a.get("is_server", False),
            round_time_s=a.get("round_time_s", round_time_s),
            exp_name=exp_name,
            agg_mode=agg_mode,
        )
        trainer = make_trainer_for_subject(data_dir, a["id"])
        peers = peers_map.get(a["id"], {})
        agents.append(DeceFLAgent(dcfg, trainer, role, peers))

    # Log the peer sets once for transparency
    print("=== Topology (symmetric ring) ===", flush=True)
    for ag in agents:
        ids = ", ".join(sorted(ag.peers.keys()))
        print(f"{ag.id}: K={len(ag.peers)} -> {{{ids}}}", flush=True)
    print("===============================", flush=True)

    return agents

async def main(cfg_path: str, seed: int | None, rounds: int | None, exp_name: str,
               local_epochs: int | None, wait_s: float | None):
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(_win_suppress_reset)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    try:
        _PORT_BASE = int(os.environ.get("PORT_BASE", "0"))
        _PORT_STRIDE = int(os.environ.get("PORT_STRIDE", "0"))
    except ValueError:
        _PORT_BASE, _PORT_STRIDE = 0, 0

    _HOSTPORT_RE = re.compile(r"^(?P<host>[^:]+):(?P<port>\d+)$")

    def _shift_port_num(p):
        if _PORT_BASE and _PORT_STRIDE:
            try:
                return _PORT_BASE + (int(p) % _PORT_STRIDE)
            except Exception:
                return p
        return p

    def _shift_endpoint(v):
        """Shift ports in various formats: int, 'host:port', {'host','port'}."""
        if v is None:
            return v
        if isinstance(v, int):
            return _shift_port_num(v)
        if isinstance(v, str):
            m = _HOSTPORT_RE.match(v.strip())
            if m:
                host = m.group("host")
                port = _shift_port_num(int(m.group("port")))
                return f"{host}:{port}"
            return v
        if isinstance(v, dict):
            if "port" in v:
                v["port"] = _shift_endpoint(v["port"])
            if "endpoint" in v:  # e.g., "127.0.0.1:9002"
                v["endpoint"] = _shift_endpoint(v["endpoint"])
            return v
        return v

    def _shift_ports_in_cfg(obj):
        """Recursively shift anything that looks like a port or endpoint."""
        if isinstance(obj, dict):
            for k, val in list(obj.items()):
                kl = str(k).lower()
                if kl in {"port","server_port","rpc_port","listen_port","coord_port","master_port"}:
                    obj[k] = _shift_endpoint(val)
                elif kl in {"endpoint","addr","address"}:
                    obj[k] = _shift_endpoint(val)
                else:
                    obj[k] = _shift_ports_in_cfg(val)
            return obj
        if isinstance(obj, list):
            return [_shift_ports_in_cfg(x) for x in obj]
        return obj

    if _PORT_BASE and _PORT_STRIDE:
        cfg = _shift_ports_in_cfg(cfg)
    # ---- end hook ----

    set_seed(seed if seed is not None else cfg.get("seed"))

    agents = await build_agents(cfg, exp_name=exp_name, rounds_override=rounds)

    # Start networking
    await asyncio.gather(*(ag.start() for ag in agents))

    # Rounds + per-round knobs
    total_rounds = min(getattr(ag.cfg, "max_rounds", 200) for ag in agents)
    le = int(local_epochs) if local_epochs is not None else int(cfg.get("local_epochs", 5))
    wt = float(wait_s) if wait_s is not None else float(cfg.get("server_wait_timeout_s", 1.0))

    for r in range(total_rounds):
        print(f"===== GLOBAL ROUND {r} =====", flush=True)
        await asyncio.gather(*(ag.run_one_round(local_epochs=le, wait_timeout_s=wt) for ag in agents))

    await asyncio.gather(*(ag.shutdown() for ag in agents if hasattr(ag, "shutdown")))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg", help="Path to YAML config")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (overrides YAML seed)")
    ap.add_argument("--round", dest="rounds", type=int, default=None, help="Number of rounds override")
    ap.add_argument("--rounds", dest="rounds_alt", type=int, default=None, help="Alias for --round")
    ap.add_argument("--exp", type=str, default="decefl", help="Experiment name")
    ap.add_argument("--local-epochs", type=int, default=None, help="Local epochs per round")
    ap.add_argument("--wait", type=float, default=None, help="Wait time (s) for peer updates per round")
    args = ap.parse_args()

    rounds = args.rounds if args.rounds is not None else args.rounds_alt

    asyncio.run(main(
        cfg_path=args.cfg,
        seed=args.seed,
        rounds=rounds,
        exp_name=args.exp,
        local_epochs=args.local_epochs,
        wait_s=args.wait,
    ))
