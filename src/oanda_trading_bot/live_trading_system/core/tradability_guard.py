from __future__ import annotations

import time
from typing import Dict, Any

_SUSPENDED: Dict[str, Dict[str, Any]] = {}

def suspend(symbol: str, reason: str = ""):
    _SUSPENDED[symbol] = {"reason": reason, "since": time.time()}

def resume(symbol: str):
    _SUSPENDED.pop(symbol, None)

def is_suspended(symbol: str) -> bool:
    return symbol in _SUSPENDED

def get_all() -> Dict[str, Any]:
    return dict(_SUSPENDED)

