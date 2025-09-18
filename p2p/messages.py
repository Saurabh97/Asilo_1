from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict

# Simple newline-delimited JSON messages

@dataclass
class Hello:
    agent_id: str
    port: int

@dataclass
class PheromoneMsg:
    agent_id: str
    t: int
    p: float

@dataclass
class ModelDeltaMsg:
    agent_id: str
    t: int
    model_id: str
    strategy: str  # 'head' or 'proto'
    payload: Dict[str, Any]
    bytes_size: int

@dataclass
class StatsMsg:
    agent_id: str
    t: int
    bytes_sent: int
    utility: float
    f1_val: float

class MsgCodec:
    @staticmethod
    def dump(obj: Any) -> bytes:
        d = asdict(obj)
        d['__type__'] = obj.__class__.__name__
        return (json.dumps(d) + "\n").encode('utf-8')

    @staticmethod
    def load(line: bytes) -> Any:
        d = json.loads(line.decode('utf-8').strip())
        t = d.pop('__type__', None)
        if t == 'Hello':
            return Hello(**d)
        if t == 'PheromoneMsg':
            return PheromoneMsg(**d)
        if t == 'ModelDeltaMsg':
            return ModelDeltaMsg(**d)
        if t == 'StatsMsg':
            return StatsMsg(**d)
        return d