# ASILO — Adaptive Swarm-Inspired Local Optimisation (Fully Decentralized FL)

Prototype + experiments for **fully decentralized federated learning (no central server)** on **non-IID** and **heterogeneous** clients, with a focus on **communication efficiency**, **robustness**, and **fairness**.

Repository (thesis code): https://github.com/Saurabh97/Asilo_1

---

## What this project is about

Fully decentralized FL is hard because:

- **Non-IID data**: each client has different label distributions and behavior.
- **Heterogeneous clients**: devices differ in compute budget (local epochs/steps), bandwidth, and reliability.
- **Communication constraints**: limited fan-out, byte budgets, message drops/buffering.
- **Fairness**: some clients can dominate the learning while others get ignored.

This repo implements **ASILO** (a swarm-inspired decentralized method) and compares it against multiple baselines.

---

## Key idea of ASILO

ASILO combines:

- **Pheromone-guided peer selection** (learns which neighbors are high-utility)
- **Capability-aware local optimisation** (clients train according to their own budgets)
- **Event-triggered / lightweight communication** (communicate when useful)
- **Robust gating + aggregation** (accept/buffer/drop updates based on similarity/budget/robust rules)

The goal is to improve the **accuracy–communication tradeoff** while maintaining **stability** and **fairness** across clients.

---

## Implemented methods

- **ASILO** — swarm-inspired, decentralized, capability-aware, event-triggered communication
- **FedAvg** — classic FL baseline adapted to this framework
- **DeceFL** — peer-based decentralized update exchange
- **DFedSAM** — decentralized training with **SAM** optimizer support

> Note: configs are designed so that **local compute budget comes from YAML** (e.g., `local_round`, `local_epochs`, or steps-per-round depending on trainer).

---

## Dataset

Experiments target **WESAD (wrist wearable signals)** with:
- strong **subject-level non-IID**
- **label imbalance** and varying class coverage across subjects
- common setups include **binary stress vs non-stress** (depending on preprocessing)

---

## Training pipeline highlights

To make FL rounds realistic and comparable across heterogeneous clients:

- **Round-wise training budget** (fixed steps/epochs per round)
- **Streaming DataLoader iterator** (no full-epoch loop unless desired)
- iterator reinitializes on exhaustion (optional reshuffle)
- optional replay buffer growth for online/streamed data
- gradient clipping + safe evaluation guards

---

## Configuration (YAML)

Most experiments are driven by YAML configs (example structure):

```yaml
round_time_s: 0.1

run_limits:
  max_rounds: 100
  max_seconds: 1200

agents:
  - { id: S2, host: 127.0.0.1, port: 9202, local_round: 5,  model_id: wear }
  - { id: S3, host: 127.0.0.1, port: 9203, local_round: 10, model_id: wear }
  - { id: S4, host: 127.0.0.1, port: 9204, local_round: 5,  model_id: wear }

# method-specific blocks:
asilo:
  k_peers: 3
  trigger: ...
  pheromone: ...
  agg_mode: "robust"
