"""Configuration loading (config/default.json) and project paths."""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG = os.path.join(ROOT, "config", "default.json")

_cache: Dict[str, Dict[str, Any]] = {}


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the JSON config; `TOURISM_CONFIG` overrides the default path."""
    path = path or os.environ.get("TOURISM_CONFIG") or DEFAULT_CONFIG
    if path not in _cache:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        cfg["strategies"] = {name: resolve_strategy(cfg["strategies"], name) for name in cfg["strategies"]}
        _cache[path] = cfg
    return copy.deepcopy(_cache[path])


def resolve_strategy(strategies: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Apply `inherits` so ablations only list what they change."""
    spec = strategies[name]
    if "inherits" not in spec:
        out = copy.deepcopy(spec)
    else:
        out = resolve_strategy(strategies, spec["inherits"])
        for key, value in spec.items():
            if key != "inherits":
                out[key] = copy.deepcopy(value)
    out["name"] = name
    return out


def path(cfg: Dict[str, Any], key: str, *parts: str) -> str:
    """Absolute path for a configured directory (data_dir, outputs_dir, engine)."""
    base = cfg["paths"][key]
    if not os.path.isabs(base):
        base = os.path.join(ROOT, base)
    return os.path.join(base, *parts) if parts else base


def engine_path(cfg: Dict[str, Any]) -> str:
    env = os.environ.get("TOURISM_ENGINE")
    if env:
        return env
    exe = path(cfg, "engine")
    if os.name == "nt" and not exe.endswith(".exe"):
        exe += ".exe"
    return exe


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p
