# Asilo_1/ui/observer_dashboard.py
import asyncio, argparse, time, os
import streamlit as st
import pandas as pd
from collections import defaultdict
from p2p.transport import P2PNode
from p2p.messages import Join, Welcome, Introduce, PheromoneMsg, StatsMsg

st.set_page_config(page_title="ASILO — Observer", layout="wide")
st.title("ASILO — Decentralized Wearables (Observer)")

class Observer:
    def __init__(self, host: str, port: int, bootstrap: tuple[str,int] | None):
        self.id = f"OBS_{port}"
        self.node = P2PNode(host, port)
        self.bootstrap = bootstrap
        self.members = {}  # id -> (host,port,cap)
        self.p = {}
        self.bytes = defaultdict(int)
        self.node.route('Welcome', self._on_welcome)
        self.node.route('Introduce', self._on_introduce)
        self.node.route('PheromoneMsg', self._on_phero)
        self.node.route('StatsMsg', self._on_stats)

    async def start(self):
        await self.node.start()
        if self.bootstrap:
            h, p = self.bootstrap
            j = Join(agent_id=self.id, host=self.node.host, port=self.node.port, capability="OBSERVER")
            await self.node.send(h, p, j)

    async def _on_welcome(self, msg: Welcome):
        for aid, h, p, cap in msg.peers:
            self.members[aid] = (h, p, cap)
    async def _on_introduce(self, msg: Introduce):
        self.members[msg.agent_id] = (msg.host, msg.port, msg.capability)
    async def _on_phero(self, msg: PheromoneMsg):
        self.p[msg.agent_id] = msg.p
    async def _on_stats(self, msg: StatsMsg):
        self.bytes[msg.agent_id] = msg.bytes_sent

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9900)
    ap.add_argument("--bootstrap", type=str, default="127.0.0.1:9002")
    args = ap.parse_args()
    b = tuple(args.bootstrap.split(":")) if args.bootstrap else None
    b = (b[0], int(b[1])) if b else None

    obs = Observer(args.host, args.port, b)
    await obs.start()

    placeholder = st.empty()
    while True:
        ids = sorted(set(list(obs.members.keys()) + list(obs.p.keys())))
        df = pd.DataFrame({
            "agent": ids,
            "pheromone": [obs.p.get(i, None) for i in ids],
            "bytes_sent": [obs.bytes.get(i, 0) for i in ids],
        })
        with placeholder.container():
            st.subheader("Membership & Metrics")
            st.dataframe(df, use_container_width=True)
            st.caption("Observer is a peer; training continues without it.")
        await asyncio.sleep(1.5)

if __name__ == "__main__":
    asyncio.run(main())
