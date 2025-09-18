import asyncio
from typing import Callable, Awaitable, Dict, Tuple
from .messages import MsgCodec

class P2PNode:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.server: asyncio.base_events.Server | None = None
        self.handlers: Dict[str, Callable[[object], Awaitable[None]]] = {}

    async def start(self):
        self.server = await asyncio.start_server(self._on_client, self.host, self.port)

    def route(self, msg_type: str, handler: Callable[[object], Awaitable[None]]):
        self.handlers[msg_type] = handler

    async def _on_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        while True:
            line = await reader.readline()
            if not line:
                break
            msg = MsgCodec.load(line)
            tname = msg.__class__.__name__
            h = self.handlers.get(tname)
            if h:
                await h(msg)
        writer.close()
        await writer.wait_closed()

    async def send(self, host: str, port: int, msg: object):
        reader, writer = await asyncio.open_connection(host, port)
        writer.write(MsgCodec.dump(msg))
        await writer.drain()
        writer.close()
        await writer.wait_closed()