"""
Utility functions for JSON serialization of nested belief states.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def to_json(payload: Dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def from_json(payload: str) -> Dict[str, Any]:
    return json.loads(payload)


__all__ = ["to_json", "from_json"]
