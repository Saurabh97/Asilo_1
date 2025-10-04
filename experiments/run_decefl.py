import os, sys, yaml, asyncio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Asilo_1.agents.decefl_agent import DeceFLAgent, DeceFLConfig
from Asilo_1.fl.trainers.tabular_sklearn import make_trainer_for_subject

async def launch_agent(a, data_dir, peers, seed_id):
    role = "server" if a['id'] == seed_id else "client"
    cfg = DeceFLConfig(
        agent_id=a['id'],
        host=a['host'],
        port=a['port'],
        model_id=a['model_id'],
        max_rounds=a.get("max_rounds", 200),
        is_server=a.get("is_server", False)
    )
    trainer = make_trainer_for_subject(data_dir, a['id'])
    ag = DeceFLAgent(cfg, trainer, role, peers)
    await ag.run_forever()

async def main(cfg_path: str):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    agents = cfg['agents']
    seed_id = agents[0]["id"]
    data_dir = cfg['data_dir']

    tasks = []
    for a in agents:
        if a.get("is_server", False):
            peers = {}
        else:
            server = [x for x in agents if x.get("is_server", False)][0]
            peers = {server['id']: (server['host'], server['port'])}
        tasks.append(asyncio.create_task(launch_agent(a, data_dir, peers, seed_id)))

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg", help="YAML config for DeceFL")
    args = ap.parse_args()
    asyncio.run(main(args.cfg))
