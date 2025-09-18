import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import yaml
import asyncio
import random
import numpy as np
from functools import partial


from Asilo_1.core.pheromone import PheromoneConfig
from Asilo_1.core.policies import CapabilityProfile
from Asilo_1.agents.base_agent import Agent, AgentConfig
from Asilo_1.agents.wesad_agent import WESADData

async def launch_agent(entry, cfg, peers_map, pher_cfg, cap_profiles, model_id, wesad_dir):
    name = entry['name']
    host = entry['host']
    port = int(entry['port'])
    subject = entry['subject']
    cap_name = entry['capability']
    cap = cap_profiles[cap_name]

    csv_path = os.path.join(wesad_dir, f"{subject}.csv")
    data = WESADData(csv_path)
    trainer = data.make_trainer()

    acfg = AgentConfig(
        agent_id=name,
        host=host,
        port=port,
        model_id=model_id,
        pheromone=pher_cfg,
        capability=cap,
        round_time_s=cfg['round_time_s'],
    )

    agent = Agent(acfg, trainer, peers_map)
    await agent.run_forever()

async def main(config_path: str):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    random.seed(cfg.get('seed', 42))
    np.random.seed(cfg.get('seed', 42))

    pher_cfg = PheromoneConfig(**cfg['pheromone'])

    # Capability profiles
    caps = cfg['capability_profiles']
    cap_profiles = {
        k: CapabilityProfile(**v) for k, v in caps.items()
    }

    model_id = cfg.get('model_id', 'wesad_v01')
    wesad_dir = cfg['paths']['wesad_dir']

    # Build peers map (each agent sees others as peers)
    agents = cfg['agents']
    peers_map_global = {}
    for a in agents:
        peers_map_global[a['name']] = (a['host'], int(a['port']))

    tasks = []
    for a in agents:
        peers_excluding_self = {k: v for k, v in peers_map_global.items() if k != a['name']}
        coro = launch_agent(a, cfg, peers_excluding_self, pher_cfg, cap_profiles, model_id, wesad_dir)
        tasks.append(asyncio.create_task(coro))

    await asyncio.gather(*tasks)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python experiments/run.py experiments/configs/wesad_case.yaml")
        sys.exit(1)
    path = sys.argv[1]
    try:
        import uvloop  # optional
        uvloop.install()
    except Exception:
        pass
    asyncio.run(main(path))