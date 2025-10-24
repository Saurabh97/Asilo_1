# experiments/run_dfedsam.py
import os, sys, yaml, asyncio, io,re, random, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
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
from Asilo_1.agents.dfedsam_agent import DFedSAMOrchestrator, DFedSAMClientAgent, DFedSAMClientConfig
from Asilo_1.fl.trainers.tabular_sklearn import make_trainer_for_subject


def _synthesize_orchestrator_from_first_agent(agents_cfg: list[dict]) -> dict:
    """Create an orchestrator block if none provided (FedAvg-like convenience)."""
    first = agents_cfg[0]
    return {
        "id": "ORCH",
        "host": first["host"],
        "port": int(first["port"]) + 1000,
        "model_id": first["model_id"],
    }


async def main(cfg_path: str):

    loop = asyncio.get_running_loop()
    loop.set_exception_handler(_win_suppress_reset)
    
    with open(cfg_path, 'r', encoding='utf-8') as f:
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

    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--rounds', type=int, default=None)
    ap.add_argument('--exp', type=str, default=None)
    ap.add_argument('--client_barrier_timeout', type=float, default=None)
    ap.add_argument('--sam_rho', type=float, default=None)
    ap.add_argument('--no_sam', action='store_true', default=False)
    known, _ = ap.parse_known_args()

    # seed
    seed = known.seed if known.seed is not None else cfg.get('seed')
    if seed is not None:
        random.seed(seed)
        try: np.random.seed(seed)
        except Exception: pass
        os.environ.setdefault("PYTHONHASHSEED", str(seed))

    # rounds
    if known.rounds is not None:
        cfg.setdefault('run_limits', {})['max_rounds'] = known.rounds
        for a in cfg.get('agents', []):
            if isinstance(a, dict):
                a['max_rounds'] = known.rounds

    # experiment name
    exp = known.exp if known.exp is not None else "dfedsam_orch"
    if seed is not None: exp = f"{exp}_seed{seed}"
    os.environ["EXP_NAME"] = exp

    # orchestrator config
    orch_cfg = cfg.get('orchestrator') or _synthesize_orchestrator_from_first_agent(cfg['agents'])
    round_time_s = cfg.get('round_time_s', 0.1)
    max_rounds = cfg.get('run_limits', {}).get('max_rounds', 200)
    barrier_timeout = known.client_barrier_timeout if known.client_barrier_timeout is not None else cfg.get('client_barrier_timeout_s', 5.0)
    sam_rho = (known.sam_rho if known.sam_rho is not None else cfg.get('sam_rho', 0.05))
    sam_enabled = not known.no_sam if known.no_sam is not None else cfg.get('sam_enabled', True)

    # build orchestrator
    orch = DFedSAMOrchestrator(
        agent_id=orch_cfg['id'],
        host=orch_cfg['host'],
        port=int(orch_cfg['port']),
        model_id=orch_cfg['model_id'],
        round_time_s=round_time_s,
        eps=1e-8
    )

    # build clients from agents[]
    data_dir = cfg['data_dir']
    clients: list[DFedSAMClientAgent] = []
    orch_tuple = (orch_cfg['id'], (orch_cfg['host'], int(orch_cfg['port'])))

    for a in cfg['agents']:
        agent = DFedSAMClientAgent(
            DFedSAMClientConfig(
                agent_id=a['id'],
                host=a['host'],
                port=int(a['port']),
                model_id=a['model_id'],
                round_time_s=round_time_s,
                max_rounds=a.get("max_rounds", max_rounds),
                client_barrier_timeout_s=barrier_timeout,
                sam_rho=sam_rho,
                sam_enabled=sam_enabled
            ),
            trainer=make_trainer_for_subject(data_dir, a['id']),
            orchestrator=orch_tuple
        )
        clients.append(agent)

    # start network
    await orch.start()
    for c in clients:
        await c.start()

    # rounds (FedAvg-like): orchestrator and clients step per round
    try:
        for r in range(max_rounds):
            print(f"\n=== GLOBAL ROUND {r} ===", flush=True)
            await asyncio.gather(
                orch.run_round(r),                 # waits for quorum then broadcasts
                *(c.run_round(r) for c in clients) # each client trains → sends → waits for broadcast
            )
    finally:
        await asyncio.gather(orch.shutdown(), *(c.shutdown() for c in clients), return_exceptions=True)


if __name__ == "__main__":
    import argparse, asyncio
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--rounds', type=int, default=None)
    ap.add_argument('--exp', type=str, default=None)
    ap.add_argument('--client_barrier_timeout', type=float, default=None)
    ap.add_argument('--sam_rho', type=float, default=None)
    ap.add_argument('--no_sam', action='store_true', default=False)
    ap.add_argument("cfg", help="YAML config")
    args = ap.parse_args()
    asyncio.run(main(args.cfg))
