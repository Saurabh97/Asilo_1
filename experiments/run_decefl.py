import os, sys, yaml, asyncio, io
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# ensure stdout/err don’t crash on odd bytes
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from Asilo_1.agents.decefl_agent import DeceFLAgent, DeceFLConfig
from Asilo_1.fl.trainers.tabular_sklearn import make_trainer_for_subject

async def launch_agent(a, data_dir, peers, seed_id):
    """
    Start a single DeceFL agent and run its main loop.
    On cancellation or exit, always attempt a graceful shutdown.
    (Signature unchanged.)
    """
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

    try:
        # If your agent has a separate start(), you can call it here:
        # await ag.start()
        await ag.run_forever()
    except asyncio.CancelledError:
        # allow cooperative cancellation
        pass
    finally:
        # ensure transport/resources are closed to avoid "Task was destroyed..." warnings
        try:
            shutdown = getattr(ag, "shutdown", None)
            if callable(shutdown):
                await shutdown()
        except Exception:
            # swallow shutdown exceptions so we don't mask the real cause
            pass


async def main(cfg_path: str):
    """
    Launch all agents defined in the YAML and run them concurrently.
    Always cancels and awaits the tasks on exit for a clean shutdown.
    (Signature unchanged.)
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

        # run until they finish or we get cancelled/exception
        await asyncio.gather(*tasks)
    finally:
        # on any exit path, cancel remaining tasks and await them
        for t in tasks:
            if not t.done():
                t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg", help="YAML config for DeceFL")
    args = ap.parse_args()
    asyncio.run(main(args.cfg))
