import os, sys, yaml, asyncio, io
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# ensure stdout/err don’t crash on odd bytes
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from Asilo_1.agents.dfedsam_agent import DFedSAMAgent, DFedSAMConfig
from Asilo_1.fl.trainers.tabular_sklearn import make_trainer_for_subject

async def launch_agent(a, data_dir, peers, seed_id):
    """
    Signature unchanged.
    Starts a DFedSAM agent, runs its loop, and always shuts it down cleanly.
    """
    role = "server" if a['id'] == seed_id else "client"
    cfg = DFedSAMConfig(
        agent_id=a['id'],
        host=a['host'],
        port=a['port'],
        model_id=a['model_id'],
        max_rounds=a.get("max_rounds", 200),
        is_server=a.get("is_server", False)
    )
    trainer = make_trainer_for_subject(data_dir, a['id'])
    ag = DFedSAMAgent(cfg, trainer, role, peers)

    try:
        # If your agent has a start() step, you can call it here before run_forever().
        # await ag.start()
        await ag.run_forever()
    except asyncio.CancelledError:
        pass
    finally:
        try:
            shutdown = getattr(ag, "shutdown", None)
            if callable(shutdown):
                await shutdown()
        except Exception:
            pass

async def main(cfg_path: str):
    """
    Signature unchanged.
    Launch all DFedSAM agents and ensure graceful shutdown on any exit path.
    """
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    agents = cfg['agents']
    seed_id = agents[0]["id"]
    data_dir = cfg['data_dir']

    tasks = []
    try:
        for a in agents:
            if a.get("is_server", False):
                peers = {}
            else:
                server = [x for x in agents if x.get("is_server", False)][0]
                peers = {server['id']: (server['host'], server['port'])}
            tasks.append(asyncio.create_task(launch_agent(a, data_dir, peers, seed_id), name=f"agent:{a['id']}"))

        await asyncio.gather(*tasks)
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg", help="YAML config for DFedSAM")
    args = ap.parse_args()
    asyncio.run(main(args.cfg))
