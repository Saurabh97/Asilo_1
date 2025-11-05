import os, sys, yaml, asyncio, io, random,re
import numpy as np
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
    
from Asilo_1.agents.fedavg_agent import (
    FedAvgClientAgent, FedAvgClientConfig,
    FedAvgOrchestrator, FedAvgOrchestratorConfig
)
from Asilo_1.fl.trainers.tabular_sklearn import make_trainer_for_subject


def _synthesize_orchestrator_from_first_agent(agents_cfg: list[dict]) -> dict:
    """
    If no explicit 'orchestrator' block in YAML, create one:
      - host = first agent host
      - port = first agent port + 1000
      - id = 'ORCH'
      - model_id = agents_cfg[0].model_id
    """
    first = agents_cfg[0]
    return {
        "id": "ORCH",
        "host": first["host"],
        "port": int(first["port"]) + 1000,
        "model_id": first["model_id"],
    }


async def main(cfg_path: str):

    # install quiet handler on the running loop that asyncio.run() created
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(_win_suppress_reset)

    with open(cfg_path, 'r') as f:
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
    known, _ = ap.parse_known_args()

    # seed
    _seed = known.seed if known.seed is not None else cfg.get('seed')
    if _seed is not None:
        random.seed(_seed)
        try: np.random.seed(_seed)
        except Exception: pass
        os.environ.setdefault("PYTHONHASHSEED", str(_seed))

    # rounds override
    if known.rounds is not None:
        cfg.setdefault('run_limits', {})['max_rounds'] = known.rounds
        for _a in cfg.get('agents', []):
            if isinstance(_a, dict):
                _a['max_rounds'] = known.rounds

    # exp name
    _exp = known.exp if known.exp is not None else "fedavg_orch"
    if _seed is not None:
        _exp = f"{_exp}"
    os.environ["EXP_NAME"] = _exp

    # discover orchestrator
    orch_cfg = cfg.get('orchestrator')
    if orch_cfg is None:
        orch_cfg = _synthesize_orchestrator_from_first_agent(cfg['agents'])
        print(f"[runner] No 'orchestrator' in YAML → synthesized {orch_cfg}")

    round_time_s = cfg.get('round_time_s', 0.1)
    max_rounds = cfg.get('run_limits', {}).get('max_rounds', 200)
    barrier_timeout = known.client_barrier_timeout if known.client_barrier_timeout is not None else cfg.get('client_barrier_timeout_s', 5.0)

    # build orchestrator
    orch = FedAvgOrchestrator(FedAvgOrchestratorConfig(
        agent_id=orch_cfg['id'],
        host=orch_cfg['host'],
        port=int(orch_cfg['port']),
        model_id=orch_cfg['model_id'],
        round_time_s=round_time_s,
        max_rounds=max_rounds
    ))

    # build clients (ALL entries in cfg['agents'] are clients now, including S2)
    clients: list[FedAvgClientAgent] = []
    data_dir = cfg['data_dir']
    orch_tuple = (orch_cfg['id'], (orch_cfg['host'], int(orch_cfg['port'])))

    for a in cfg['agents']:
        c = FedAvgClientAgent(
            FedAvgClientConfig(
                agent_id=a['id'],
                host=a['host'],
                port=int(a['port']),
                model_id=a['model_id'],
                round_time_s=round_time_s,
                max_rounds=a.get("max_rounds", max_rounds),
                client_barrier_timeout_s=barrier_timeout
            ),
            trainer=make_trainer_for_subject(data_dir, a['id']),
            orchestrator=orch_tuple
        )
        clients.append(c)

    # start network
    await orch.start()
    for c in clients:
        await c.start()

    # drive rounds (orchestrator + clients per round)
    try:
        for r in range(max_rounds):
            await asyncio.gather(
                orch.run_round(r),
                *(c.run_round(r) for c in clients),
            )
    finally:
        # optional graceful shutdowns if your P2PNode exposes them
        for obj in [orch, *clients]:
            try:
                shutdown = getattr(obj, "shutdown", None)
                if callable(shutdown):
                    await shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--rounds', type=int, default=None)
    ap.add_argument('--exp', type=str, default=None)
    ap.add_argument('--client_barrier_timeout', type=float, default=None)
    ap.add_argument("cfg", help="YAML config")
    args = ap.parse_args()
    asyncio.run(main(args.cfg))
