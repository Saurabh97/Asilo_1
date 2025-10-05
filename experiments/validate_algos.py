#!/usr/bin/env python3
import asyncio, os, sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Asilo_1.agents.fedavg_agent import FedAvgAgent, FedAvgConfig
from Asilo_1.agents.dfedsam_agent import DFedSAMAgent, DFedSAMConfig
from Asilo_1.agents.decefl_agent import DeceFLAgent, DeceFLConfig
from Asilo_1.agents.base_agent import Agent, AgentConfig
from Asilo_1.core.pheromone import PheromoneConfig
from Asilo_1.core.capability import CapabilityProfile
from Asilo_1.core.trigger import TriggerConfig
from Asilo_1.fl.trainers.tabular_sklearn import make_trainer_for_subject

# -------------------- Results dict --------------------
results = {}

# -------------------- Common setup --------------------
DATA_DIR = "Asilo_1/data/processed/WESAD_wrist"
SUBJECTS = ["S02", "S05"]  # keep it small for validation speed

# -------------------- FedAvg --------------------
async def run_fedavg():
    print("\n=== Validating FedAvg ===")
    trainer1 = make_trainer_for_subject(DATA_DIR, SUBJECTS[0])
    trainer2 = make_trainer_for_subject(DATA_DIR, SUBJECTS[1])

    cfg_server = FedAvgConfig(agent_id=SUBJECTS[0], host="127.0.0.1", port=9500,
                            model_id="wear", max_rounds=5, is_server=True)
    cfg_client = FedAvgConfig(agent_id=SUBJECTS[1], host="127.0.0.1", port=9501,
                            model_id="wear", max_rounds=5, is_server=False)

    peers_client = {SUBJECTS[0]: ("127.0.0.1", 9500)}

    ag_server = FedAvgAgent(cfg_server, trainer1, role="server", peers={})
    ag_client = FedAvgAgent(cfg_client, trainer2, role="client", peers=peers_client)

    await asyncio.gather(ag_server.run_forever(), ag_client.run_forever())

    results["FedAvg"] = {
        "final": trainer1.eval().get("f1_val", 0.0),
        "best": trainer1.eval().get("f1_val", 0.0),  # tabular_sklearn doesn't track best
        "bytes": 0
    }

# -------------------- DFedSAM --------------------
async def run_dfedsam():
    print("\n=== Validating DFedSAM ===")
    trainer1 = make_trainer_for_subject(DATA_DIR, SUBJECTS[0])
    trainer2 = make_trainer_for_subject(DATA_DIR, SUBJECTS[1])

    cfg_server = DFedSAMConfig(agent_id=SUBJECTS[0], host="127.0.0.1", port=9600,
                            model_id="wear", max_rounds=5, is_server=True)
    cfg_client = DFedSAMConfig(agent_id=SUBJECTS[1], host="127.0.0.1", port=9601,
                            model_id="wear", max_rounds=5, is_server=False)

    peers_client = {SUBJECTS[0]: ("127.0.0.1", 9600)}

    ag_server = DFedSAMAgent(cfg_server, trainer1, role="server", peers={})
    ag_client = DFedSAMAgent(cfg_client, trainer2, role="client", peers=peers_client)

    await asyncio.gather(ag_server.run_forever(), ag_client.run_forever())

    results["DFedSAM"] = {
        "final": trainer1.eval().get("f1_val", 0.0),
        "best": trainer1.eval().get("f1_val", 0.0),
        "bytes": 0
    }

# -------------------- DeceFL --------------------
async def run_decefl():
    print("\n=== Validating DeceFL ===")
    trainer1 = make_trainer_for_subject(DATA_DIR, SUBJECTS[0])
    trainer2 = make_trainer_for_subject(DATA_DIR, SUBJECTS[1])

    cfg_server = DeceFLConfig(agent_id=SUBJECTS[0], host="127.0.0.1", port=9700,
                            model_id="wear", max_rounds=5, is_server=True)
    cfg_client = DeceFLConfig(agent_id=SUBJECTS[1], host="127.0.0.1", port=9701,
                            model_id="wear", max_rounds=5, is_server=False)

    peers_client = {SUBJECTS[0]: ("127.0.0.1", 9700)}

    ag_server = DeceFLAgent(cfg_server, trainer1, role="server", peers={})
    ag_client = DeceFLAgent(cfg_client, trainer2, role="client", peers=peers_client)

    await asyncio.gather(ag_server.run_forever(), ag_client.run_forever())

    results["DeceFL"] = {
        "final": trainer1.eval().get("f1_val", 0.0),
        "best": trainer1.eval().get("f1_val", 0.0),
        "bytes": 0
    }

# -------------------- ASILO --------------------
async def run_asilo():
    print("\n=== Validating ASILO ===")
    trainer1 = make_trainer_for_subject(DATA_DIR, SUBJECTS[0])
    trainer2 = make_trainer_for_subject(DATA_DIR, SUBJECTS[1])

    pheromone_cfg = PheromoneConfig()
    trigger_cfg = TriggerConfig()
    cap = CapabilityProfile("VAL", 1.0, 3, 1, 1000)

    cfg0 = AgentConfig(
        agent_id=SUBJECTS[0], host="127.0.0.1", port=9800, model_id="wear",
        pheromone=pheromone_cfg, capability=cap,
        round_time_s=0.1, trigger=trigger_cfg,
        ttl_seconds=20.0, max_rounds=5, max_seconds=30,
        u_eps=1e-3, speak_gate_factor=0.9,
        artifact_order=("proto","head"), head_every=0,
        cos_gate_thresh=0.2, agg_mode="median", trim_p=0.1,
        aggregate_every=2, rollback_tol=0.005
    )

    cfg1 = AgentConfig(
        agent_id=SUBJECTS[1], host="127.0.0.1", port=9801, model_id="wear",
        pheromone=pheromone_cfg, capability=cap,
        round_time_s=0.1, trigger=trigger_cfg,
        ttl_seconds=20.0, max_rounds=5, max_seconds=30,
        u_eps=1e-3, speak_gate_factor=0.9,
        artifact_order=("proto","head"), head_every=0,
        cos_gate_thresh=0.2, agg_mode="median", trim_p=0.1,
        aggregate_every=2, rollback_tol=0.005
    )

    peers0, peers1 = {SUBJECTS[1]: ("127.0.0.1", 9801)}, {SUBJECTS[0]: ("127.0.0.1", 9800)}
    ag0, ag1 = Agent(cfg0, trainer1, peers=peers0), Agent(cfg1, trainer2, peers=peers1)

    await asyncio.gather(ag0.run_forever(), ag1.run_forever())

    results["ASILO"] = {
        "final": trainer1.eval().get("f1_val", 0.0),
        "best": trainer1.eval().get("f1_val", 0.0),
        "bytes": 0
    }

# -------------------- Main --------------------
async def main():
    await run_fedavg()
    await run_dfedsam()
    await run_decefl()
    await run_asilo()

    print("\n=== Comparison Summary ===")
    print(f"{'Algo':<10} {'Final F1':<10} {'Best F1':<10} {'Bytes Sent':<12}")
    print("-"*45)
    for algo, vals in results.items():
        print(f"{algo:<10} {vals['final']:<10.3f} {vals['best']:<10.3f} {vals['bytes']:<12}")

if __name__ == "__main__":
    asyncio.run(main())
