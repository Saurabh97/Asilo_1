# Asilo_1/experiments/run.py
import os, sys, yaml, asyncio, signal
from functools import partial
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Asilo_1.core.pheromone import PheromoneConfig
from Asilo_1.core.capability import CapabilityProfile
from Asilo_1.core.trigger import TriggerConfig
from Asilo_1.agents.base_agent import Agent, AgentConfig
from Asilo_1.fl.trainers.tabular_sklearn import make_trainer_for_subject

async def launch_agent(a, caps, pherocfg, trigcfg, data_dir, peers):
    capd = caps[a['capability']]
    cap = CapabilityProfile(a['capability'], capd['width'], capd['local_batches'], capd['k_peers'], capd['max_bytes_round'])
    cfg = AgentConfig(agent_id=a['id'], host=a['host'], port=a['port'], model_id=a['model_id'],
    pheromone=pherocfg, capability=cap,
    round_time_s=float(os.getenv('ROUND_TIME', 0.1)),
    trigger=trigcfg, max_rounds=None, max_seconds=None)
    trainer = make_trainer_for_subject(data_dir, a['id'])
    ag = Agent(cfg, trainer, peers)  # ← pass peers here
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

    agents = cfg['agents']
    # choose the first agent as the seed
    seed = agents[0]
    seed_id, seed_host, seed_port = seed['id'], seed['host'], seed['port']

    tasks = []
    for a in agents:
        # seed has no bootstrap; others know only the seed
        if a['id'] == seed_id:
            peers = {}
        else:
            peers = {seed_id: (seed_host, seed_port)}
        tasks.append(asyncio.create_task(launch_agent(a, caps, pherocfg, trigcfg, data_dir, peers)))

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cfg', nargs='?', default='Asilo_1/experiments/configs/wesad_case.yaml')
    args = ap.parse_args()
    asyncio.run(main(args.cfg))
