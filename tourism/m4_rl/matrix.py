"""Precomputed travel matrices (from the C++ engine) for fast RL training and
for quick what-if estimates when the RL policy proposes changes."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from tourism.core.city import City

FAST = {"time": 1.0, "money": 0.005, "co2": 0.0, "transfer": 1.0}
BANDS = {"offpeak": 720, "peak": 540}


class TravelMatrix:
    def __init__(self, city: City, nodes: List[int], data: Dict[str, Dict[str, List[List[float]]]]):
        self.city = city
        self.nodes = nodes
        self.idx = {n: i for i, n in enumerate(nodes)}
        self.data = data

    @classmethod
    def build(cls, engine, city: City) -> "TravelMatrix":
        nodes = [p["node"] for p in city.pois] + list(city.hotels)
        data = {}
        for band, dep in BANDS.items():
            for green in (False, True):
                modes = ["walk", "transit"] if green else ["walk", "transit", "car"]
                m = engine.matrix(nodes, dep, modes, FAST)
                data["%s|%s" % (band, "green" if green else "any")] = {
                    "time": m["time"], "co2": m["co2_kg"], "car": [[1.0 if c else 0.0 for c in row] for row in m["car"]],
                    "money": m["money"], "dist": m["dist_m"]}
        return cls(city, nodes, data)

    def _band(self, t: float) -> str:
        return "peak" if any(w[0] <= t < w[1] for w in self.city.params["peak_windows"]) else "offpeak"

    def get(self, key: str, a: int, b: int, t: float, green: bool) -> float:
        if a == b:
            return 0.0
        m = self.data["%s|%s" % (self._band(t), "green" if green else "any")][key]
        return m[self.idx[a]][self.idx[b]]

    def time(self, a: int, b: int, t: float, green: bool = False) -> float:
        return self.get("time", a, b, t, green)

    def co2(self, a: int, b: int, t: float, green: bool = False) -> float:
        return self.get("co2", a, b, t, green)

    def uses_car(self, a: int, b: int, t: float, green: bool = False) -> bool:
        return self.get("car", a, b, t, green) > 0.5

    def money(self, a: int, b: int, t: float, green: bool = False) -> float:
        return self.get("money", a, b, t, green)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"nodes": self.nodes, "data": self.data}, fh)

    @classmethod
    def load(cls, city: City, path: str) -> "TravelMatrix":
        with open(path, "r", encoding="utf-8") as fh:
            d: Dict[str, Any] = json.load(fh)
        return cls(city, d["nodes"], d["data"])
