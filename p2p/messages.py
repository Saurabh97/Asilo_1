# Asilo_1/p2p/messages.py
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import base64
import json 
from dataclasses import asdict, is_dataclass

@dataclass
class Hello:
    agent_id: str
    peers: List[str]

@dataclass
class PheromoneMsg:
    agent_id: str
    t: int
    p: float
    # sketch_bits: Optional[int] = None  # enable if you decide to piggyback

@dataclass
class ModelDeltaMsg:
    agent_id: str
    t: int
    model_id: str
    strategy: str  # "proto" | "head"
    payload: Dict[str, Any]
    bytes_size: int

@dataclass
class StatsMsg:
    agent_id: str
    t: int
    bytes_sent: int

# dynamic membership
@dataclass
class Join:
    agent_id: str
    host: str
    port: int
    capability: str

@dataclass
class Welcome:
    peers: List[Tuple[str, str, int, Optional[str]]]  # (id, host, port, cap)

@dataclass
class Introduce:
    agent_id: str
    host: str
    port: int
    capability: str

@dataclass
class Bye:
    agent_id: str

# Registry of message types by name
_MSG_REGISTRY = {}
def _register_msg(cls):
    _MSG_REGISTRY[cls.__name__] = cls
    return cls

# Register all message types
for _cls in [Hello, PheromoneMsg, ModelDeltaMsg, StatsMsg, Join, Welcome, Introduce, Bye]:
    _register_msg(_cls)

def _b64ify(obj):
    if isinstance(obj, (bytes, bytearray)):
        return {"__b64__": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, dict):
        return {k: _b64ify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_b64ify(v) for v in obj]
    return obj

def _deb64ify(obj):
    if isinstance(obj, dict):
        if "__b64__" in obj and len(obj) == 1:
            return base64.b64decode(obj["__b64__"].encode("ascii"))
        return {k: _deb64ify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deb64ify(v) for v in obj]
    return obj

class MsgCodec:
    @staticmethod
    def encode(msg_obj):
        if not is_dataclass(msg_obj):
            raise TypeError(f"MsgCodec.encode expects a dataclass, got {type(msg_obj)}")
        data = asdict(msg_obj)
        if data.get("payload") is not None:
            data["payload"] = _b64ify(data["payload"])
        return {"type": msg_obj.__class__.__name__, "data": data}

    @staticmethod
    def decode(obj):
        if not isinstance(obj, dict) or "type" not in obj or "data" not in obj:
            raise TypeError("MsgCodec.decode expects {'type':..., 'data':...}")
        typ = obj["type"]; data = obj["data"]
        cls = _MSG_REGISTRY.get(typ)
        if cls is None:
            raise ValueError(f"Unknown message type: {typ}")
        if data.get("payload") is not None:
            data["payload"] = _deb64ify(data["payload"])
        return cls(**data)

    # === Methods your transport expects ===
    @staticmethod
    def dump(msg_obj) -> bytes:
        """Return newline-terminated JSON bytes."""
        wire = MsgCodec.encode(msg_obj)
        return (json.dumps(wire) + "\n").encode("utf-8")

    @staticmethod
    def load(line: bytes):
        """Parse one newline-delimited JSON line -> message object."""
        obj = json.loads(line.decode("utf-8"))
        return MsgCodec.decode(obj)