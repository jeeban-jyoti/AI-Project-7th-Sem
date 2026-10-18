"""Engine wrappers for routing (UCS / Greedy / A*) and the itinerary CSP."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from tourism.core.types import TravellerProfile
from tourism.engine.client import EngineClient

FASTEST = {"time": 1.0, "money": 0.0, "co2": 0.0, "transfer": 0.0}
GREENEST = {"time": 0.05, "money": 0.0, "co2": 20.0, "transfer": 0.0}
CHEAPEST = {"time": 0.05, "money": 1.0, "co2": 0.0, "transfer": 0.0}


def route(engine: EngineClient, src: int, dst: int, depart: int, modes: List[str],
          weights: Optional[Dict[str, float]] = None, algorithm: str = "astar",
          accessible_only: bool = False, context: Optional[str] = None) -> Dict[str, Any]:
    return engine.route(src, dst, depart, modes, weights or FASTEST, algorithm, accessible_only, context)


def compare_modes(engine: EngineClient, src: int, dst: int, depart: int, accessible_only: bool = False,
                  context: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """The same trip under different objectives / mode sets (what a transport
    agent would compare for the traveller)."""
    options = {
        "fastest_any_mode": (["walk", "transit", "car"], FASTEST),
        "public_transport": (["walk", "transit"], FASTEST),
        "walk_only": (["walk"], FASTEST),
        "lowest_co2": (["walk", "transit", "car"], GREENEST),
        "cheapest": (["walk", "transit", "car"], CHEAPEST),
    }
    out = {}
    for name, (modes, w) in options.items():
        r = engine.route(src, dst, depart, modes, w, "astar", accessible_only, context)
        out[name] = r
    return out


def benchmark(engine: EngineClient, pairs: int = 200, seed: int = 7,
              weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    return engine.call("bench", pairs=pairs, seed=seed, weights=weights or {"time": 1.0, "money": 0.01, "co2": 1.0,
                                                                            "transfer": 2.0})


def solve_itinerary(engine: EngineClient, profile: TravellerProfile, candidates: List[Dict[str, Any]],
                    weights: Dict[str, float], context: str, hard_capacity: float,
                    csp_options: Dict[str, Any], lunch_window=(720, 870)) -> Dict[str, Any]:
    return engine.call("solve", context=context, profile=profile.to_engine(hard_capacity, lunch_window),
                       candidates=candidates, weights=weights, options=csp_options)
