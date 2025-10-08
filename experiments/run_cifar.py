# Asilo_1/experiments/run.py
import os, sys, yaml, asyncio, signal
from functools import partial
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Asilo_1.core.pheromone import PheromoneConfig
from Asilo_1.core.capability import CapabilityProfile
from Asilo_1.core.trigger import TriggerConfig
from Asilo_1.agents.base_agent import Agent, AgentConfig
from Asilo_1.fl.trainers.cifar_federated import make_trainer_for_subject

async def launch_agent(a, caps, pherocfg, trigcfg, data_dir, peers, global_robust, global_limits, global_misc):
    capd = caps[a['capability']]
    cap = CapabilityProfile(a['capability'], capd['width'], capd['local_batches'],
                            capd['k_peers'], capd['max_bytes_round'])

    a_robust = a.get('robust') or {}
    robust = {**global_robust, **a_robust}

    a_limits = a.get('run_limits') or {}
    limits = {**global_limits, **a_limits}

    # misc (ttl_seconds, u_eps, speak_gate_factor, artifact_order, head_every)
    a_misc = a.get('misc') or {}
    misc = {**global_misc, **a_misc}

    cfg = AgentConfig(
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

    trainer = make_trainer_for_subject(data_dir, a['id'])
    ag = Agent(cfg, trainer, peers)
    try:
        await ag.run_forever()
    finally:
        await ag.send_bye()


async def main(cfg_path: str):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    pherocfg = PheromoneConfig(**cfg['pheromone'])
    trigcfg = TriggerConfig(**cfg['trigger'])
    caps = cfg['capabilities']
    data_dir = cfg['data_dir']

    global_robust = cfg.get('robust', {}) or {}
    global_limits = cfg.get('run_limits', {}) or {}

    # NEW: misc at top level
    global_misc = {
        "ttl_seconds": cfg.get('ttl_seconds'),
        "u_eps": cfg.get('u_eps'),
        "speak_gate_factor": cfg.get('speak_gate_factor'),
        "artifact_order": cfg.get('artifact_order'),
        "head_every": cfg.get('head_every'),
        "round_time_s": cfg.get('round_time_s'),
    }

    agents = cfg['agents']
    seed = agents[0]
    seed_id, seed_host, seed_port = seed['id'], seed['host'], seed['port']

    tasks = []
    for a in agents:
        peers = {} if a['id'] == seed_id else {seed_id: (seed_host, seed_port)}
        tasks.append(asyncio.create_task(
            launch_agent(a, caps, pherocfg, trigcfg, data_dir, peers,
            global_robust, global_limits, global_misc)
        ))

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cfg', nargs='?', default='Asilo_1/experiments/configs/cifar_asilo.yaml')
    args = ap.parse_args()
    asyncio.run(main(args.cfg))
