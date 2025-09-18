from typing import Dict, Any, Tuple


def delta_size(payload: Dict[str, Any]) -> int:
# Fallback size if not provided
    return len(str(payload).encode('utf-8'))