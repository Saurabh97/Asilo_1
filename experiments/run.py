# Asilo_1/experiments/run.py

import os, sys, yaml, asyncio, io, random,re
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# robust stdout/err on Windows for odd bytes
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def _win_suppress_reset(loop, context):
    exc = context.get("exception")
    if isinstance(exc, ConnectionResetError) and sys.platform.startswith("win"):
        # Suppress noisy "existing connection was forcibly closed" during shutdown
        return
    loop.default_exception_handler(context)

from Asilo_1.core.pheromone import PheromoneConfig
from Asilo_1.core.capability import CapabilityProfile
from Asilo_1.core.trigger import TriggerConfig
from Asilo_1.agents.base_agent import Agent, AgentConfig
from Asilo_1.fl.trainers.tabular_sklearn import make_trainer_for_subject


def _make_agent_cfg(a, caps, pherocfg, trigcfg, global_robust, global_limits, global_misc):
    capd = caps[a['capability']]
    cap = CapabilityProfile(
        a['capability'],
        capd['width'],
        capd['local_batches'],
        capd['k_peers'],
        capd['max_bytes_round'],
    )

    robust = {**(global_robust or {}), **(a.get('robust') or {})}
    limits = {**(global_limits or {}), **(a.get('run_limits') or {})}
    misc   = {**(global_misc   or {}), **(a.get('misc') or {})}

    return AgentConfig(
        agent_id=a['id'],
        host=a['host'],
        port=a['port'],
        model_id=a['model_id'],
        pheromone=pherocfg,
        capability=cap,
        round_time_s=float(os.getenv('ROUND_TIME', global_misc.get('round_time_s', 0.1))),
        trigger=trigcfg,

        ttl_seconds=float(misc.get('ttl_seconds', 20.0)),
        max_rounds=limits.get('max_rounds', None),
        max_seconds=limits.get('max_seconds', None),

        u_eps=float(misc.get('u_eps', 1e-3)),
        speak_gate_factor=float(misc.get('speak_gate_factor', 0.9)),
        artifact_order=tuple(misc.get('artifact_order', ["proto","head"])),
        head_every=int(misc.get('head_every', 0)),

        cos_gate_thresh=float(robust.get('cos_gate_thresh', 0.2)),
        agg_mode=str(robust.get('agg_mode', 'median')),
        trim_p=float(robust.get('trim_p', 0.1)),
        aggregate_every=int(robust.get('aggregate_every', 5)),
        rollback_tol=float(robust.get('rollback_tol', 0.005)),
    )


async def main(cfg_path: str):

    # install quiet handler on the running loop that asyncio.run() created
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(_win_suppress_reset)


    # ===== load yaml =====
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
    # ===== CLI overrides & seed =====
    try:
        import argparse as _argparse
        _ap = _argparse.ArgumentParser(add_help=False)
        _ap.add_argument('--seed', type=int, default=None)
        _ap.add_argument('--rounds', type=int, default=None)
        _ap.add_argument('--exp', type=str, default=None)
        known, _ = _ap.parse_known_args()
    except Exception:
        class Tmp: pass
        known = Tmp(); known.seed=None; known.rounds=None; known.exp=None

    _seed = known.seed if known.seed is not None else cfg.get('seed')
    if _seed is not None:
        random.seed(_seed)
        try: np.random.seed(_seed)
        except Exception: pass
        os.environ.setdefault("PYTHONHASHSEED", str(_seed))

    if known.rounds is not None:
        cfg.setdefault('run_limits', {})
        cfg['run_limits']['max_rounds'] = known.rounds
        for _a in cfg.get('agents', []):
            if isinstance(_a, dict):
                _a.setdefault('run_limits', {})
                _a['run_limits']['max_rounds'] = known.rounds

    _exp = known.exp if known.exp else "asilo"
    if _seed is not None:
        _exp = f"{_exp}"
    os.environ["EXP_NAME"] = _exp

    # ===== objects from cfg =====
    pherocfg = PheromoneConfig(**cfg['pheromone'])
    trigcfg  = TriggerConfig(**cfg['trigger'])
    caps     = cfg['capabilities']
    data_dir = cfg['data_dir']

    global_robust = cfg.get('robust', {}) or {}
    global_limits = cfg.get('run_limits', {}) or {}
    global_misc = {
        "ttl_seconds": cfg.get('ttl_seconds'),
        "u_eps": cfg.get('u_eps'),
        "speak_gate_factor": cfg.get('speak_gate_factor'),
        "artifact_order": cfg.get('artifact_order'),
        "head_every": cfg.get('head_every'),
        "round_time_s": cfg.get('round_time_s'),
    }

    # ===== build agents (full-mesh peers) =====
    agent_specs = cfg['agents']
    id2addr = {a['id']: (a['host'], a['port']) for a in agent_specs}
    N = len(agent_specs)

    # global start gate & round slot (lock-step)
    start_gate = asyncio.Event()
    round_slot_s = float(os.getenv("ROUND_SLOT_S", global_misc.get('round_time_s', 0.1)))
    join_window = float(os.getenv("JOIN_WINDOW_S", "15"))  # discovery wait before opening the gate

    agents = []
    for a in agent_specs:
        peers = {aid: addr for aid, addr in id2addr.items() if aid != a['id']}  # full mesh (exclude self)
        cfg_a = _make_agent_cfg(a, caps, pherocfg, trigcfg, global_robust, global_limits, global_misc)
        trainer = make_trainer_for_subject(data_dir, a['id'])
        agent = Agent(cfg_a, trainer, peers,
                      start_gate=start_gate,
                      min_peers=len(peers),           # require full quorum
                      round_slot_s=round_slot_s)
        agents.append(agent)

    # ===== launch: setup all agents (no while-loops in agents) =====
    async def wait_all_discovered(timeout: float) -> None:
        end = asyncio.get_event_loop().time() + timeout
        while True:
            have = sum(int(a._have_quorum()) for a in agents)
            if have == N:
                print(f"[runner] discovery complete {have}/{N}; opening start gate.", flush=True)
                break
            if asyncio.get_event_loop().time() >= end:
                print(f"[runner] discovery window expired ({have}/{N}); opening start gate anyway.", flush=True)
                break
            await asyncio.sleep(0.25)

    try:
        # one-time per-agent networking & discovery prep
        await asyncio.gather(*[a.setup() for a in agents])

        # wait for discovery window, then let everyone start round 0 together
        await wait_all_discovered(join_window)
        start_gate.set()

        # determine total rounds
        R = (cfg.get('run_limits', {}) or {}).get('max_rounds')
        if R is None:
            vals = [a.cfg.max_rounds for a in agents if a.cfg.max_rounds is not None]
            R = min(vals) if vals else 200
        R = int(R)
        print(f"[runner] starting lock-step rounds: R={R}", flush=True)

        # lock-step: every round calls each agent's run_round()
        for r in range(R):
            await asyncio.gather(*[a.run_round() for a in agents])

        print(f"[runner] rounds complete.", flush=True)

    except KeyboardInterrupt:
        print("[runner] KeyboardInterrupt — shutting down...", flush=True)
    finally:
        # clean shutdown for all agents
        try:
            await asyncio.gather(*[a.teardown() for a in agents], return_exceptions=True)
        except Exception:
            pass
        print("[runner] all agents stopped.", flush=True)

      


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--rounds', type=int, default=None)
    ap.add_argument('--exp', type=str, default=None)
    ap.add_argument('cfg', nargs='?', default='Asilo_1/experiments/configs/wesad_case.yaml')
    args = ap.parse_args()
    asyncio.run(main(args.cfg))
