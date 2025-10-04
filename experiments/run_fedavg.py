import os, sys, yaml, asyncio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Asilo_1.agents.fedavg_agent import FedAvgAgent, FedAvgConfig
from Asilo_1.fl.trainers.tabular_sklearn import make_trainer_for_subject

async def launch_agent(a, data_dir, peers, seed_id):
    # role derived before building config
    role = "server" if a['id'] == seed_id else "client"
    cfg = FedAvgConfig(
        agent_id=a['id'],
        host=a['host'],
        port=a['port'],
        model_id=a['model_id'],
        max_rounds=a.get("max_rounds", 200),
        is_server=(a['id'] == seed_id)  # auto-mark server
    )
    trainer = make_trainer_for_subject(data_dir, a['id'])
    ag = FedAvgAgent(cfg, trainer, role, peers)
    await ag.run_forever()

async def main(cfg_path: str):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    agents = cfg['agents']
    seed_id = agents[0]["id"]  # first agent is the server
    data_dir = cfg['data_dir']

    tasks = []
    for a in agents:
        if a['id'] == seed_id:
            # server = no bootstrap peers
            peers = {}
        else:
            # all clients connect only to the server
            peers = {seed_id: (agents[0]['host'], agents[0]['port'])}
        tasks.append(asyncio.create_task(launch_agent(a, data_dir, peers, seed_id)))

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg", help="YAML config for FedAvg")
    args = ap.parse_args()
    asyncio.run(main(args.cfg))
