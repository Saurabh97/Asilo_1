import asyncio
from typing import Callable, Awaitable, Dict
from .messages import MsgCodec

class P2PNode:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.server: asyncio.base_events.Server | None = None
        self.handlers: Dict[str, Callable[[object], Awaitable[None]]] = {}
        self._client_tasks: set[asyncio.Task] = set()
        self._ready = asyncio.Event()

    async def start(self):
        # IMPORTANT: accept callback must be a normal def, not async def
        self.server = await asyncio.start_server(self._accept_client, self.host, self.port)
        self._ready.set()  # let senders know the socket is listening

    def route(self, msg_type: str, handler: Callable[[object], Awaitable[None]]):
        self.handlers[msg_type] = handler

    # ---- accept wrapper: SYNC def that schedules the real coroutine ----
    def _accept_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        task = asyncio.create_task(self._on_client(reader, writer))
        self._client_tasks.add(task)
        task.add_done_callback(lambda t: self._client_tasks.discard(t))

    async def wait_ready(self):
        await self._ready.wait()

    # ADD THIS METHOD ↓↓↓
    async def wait_started(self):
        # alias expected by base_agent.py
        await self._ready.wait()

    # ---- per-connection loop ----
    async def _on_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                msg = MsgCodec.load(line)
                tname = msg.__class__.__name__
                h = self.handlers.get(tname)
                if h:
                    await h(msg)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def wait_ready(self):
        await self._ready.wait()

    # ---- robust send with retry for startup races ----
    async def send(self, host: str, port: int, msg: object, retries: int = 6):
        # ensure our own server is up (helps when both sides start together)
        await self.wait_ready()
        delay = 0.1
        for attempt in range(retries):
            try:
                reader, writer = await asyncio.open_connection(host, port)
                try:
                    writer.write(MsgCodec.dump(msg))
                    await writer.drain()
                finally:
                    writer.close()
                    await writer.wait_closed()
                return
            except (ConnectionRefusedError, OSError):
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, 1.0)  # backoff up to 1s

    async def shutdown(self):
        # stop accepting new connections
        if self.server is not None:
            self.server.close()
            try:
                await self.server.wait_closed()
            except Exception:
                pass
            self.server = None

        # cancel and await client tasks
        for t in list(self._client_tasks):
            t.cancel()
        if self._client_tasks:
            await asyncio.gather(*self._client_tasks, return_exceptions=True)
        self._client_tasks.clear()
