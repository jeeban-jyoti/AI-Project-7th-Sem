"""City model on the Python side (the same city.json the C++ engine loads)."""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional


class City:
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.name: str = data.get("name", "city")
        self.themes: List[str] = data["themes"]
        self.slot_min: int = data["slot_min"]
        self.day_start: int = data["day_start_min"]
        self.day_end: int = data["day_end_min"]
        self.nodes: List[Dict[str, Any]] = data["nodes"]
        self.zones: List[Dict[str, Any]] = data["zones"]
        self.lines: List[Dict[str, Any]] = data["lines"]
        self.pois: List[Dict[str, Any]] = data["pois"]
        self.hotels: List[int] = data["hotels"]
        self.params: Dict[str, Any] = data["params"]
        self.poi_by_node: Dict[int, Dict[str, Any]] = {p["node"]: p for p in self.pois}
        self.poi_index: Dict[int, int] = {p["node"]: i for i, p in enumerate(self.pois)}
        self.line_by_id: Dict[str, Dict[str, Any]] = {l["id"]: l for l in self.lines}

    @classmethod
    def load(cls, path: str) -> "City":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    @property
    def n_slots(self) -> int:
        return (self.day_end - self.day_start) // self.slot_min

    def slot_of(self, t: float) -> int:
        s = int((t - self.day_start) // self.slot_min)
        return max(0, min(self.n_slots - 1, s))

    def slot_start(self, s: int) -> int:
        return self.day_start + s * self.slot_min

    def dist(self, a: int, b: int) -> float:
        na, nb = self.nodes[a], self.nodes[b]
        return math.hypot(na["x"] - nb["x"], na["y"] - nb["y"])

    def name_of(self, node: int) -> str:
        return self.nodes[node]["name"]

    def zone_name(self, zone: int) -> str:
        return self.zones[zone]["name"]

    def attractions(self) -> List[Dict[str, Any]]:
        return [p for p in self.pois if p["kind"] == "attraction"]

    def restaurants(self) -> List[Dict[str, Any]]:
        return [p for p in self.pois if p["kind"] == "restaurant"]

    def local_businesses(self) -> List[Dict[str, Any]]:
        return [p for p in self.pois if p["independent"] and p["kind"] in ("shop", "experience", "restaurant")]

    def theme_similarity(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        key = (a["node"], b["node"])
        cache = self.__dict__.setdefault("_sim_cache", {})
        if key not in cache:
            ta, tb = a["themes"], b["themes"]
            dot = sum(x * y for x, y in zip(ta, tb))
            na = math.sqrt(sum(x * x for x in ta)) or 1.0
            nb = math.sqrt(sum(x * x for x in tb)) or 1.0
            cache[key] = dot / (na * nb)
        return cache[key]

    def nearby(self, node: int, radius_m: float, kinds: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        key = (node, radius_m, tuple(kinds or ()))
        cache = self.__dict__.setdefault("_near_cache", {})
        if key not in cache:
            out = []
            for p in self.pois:
                if kinds and p["kind"] not in kinds:
                    continue
                if p["node"] != node and self.dist(node, p["node"]) <= radius_m:
                    out.append(p)
            cache[key] = out
        return cache[key]
