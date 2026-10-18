"""Evaluation scenarios from the project brief.

Each scenario fixes the day context and a timeline of events. Announced
information (calendar, season, weather forecast, planned public events,
timed-entry capacity limits) is known to the planners in the morning;
everything else (sudden crowds, delays, closures, preference changes) only
becomes known when it happens.
"""
from __future__ import annotations

from typing import Any, Dict, List

from tourism.core.city import City
from tourism.sim.truth import DayContext

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "S0_normal": {
        "label": "Normal shoulder-season weekday",
        "context": {"daykind": "weekday", "season": "shoulder", "weather": "clear"},
    },
    "S1_peak_weekend": {
        "label": "Peak-season weekend",
        "context": {"daykind": "weekend", "season": "peak", "weather": "clear"},
    },
    "S2_sudden_crowd": {
        "label": "Sudden crowd formation (unannounced)",
        "context": {"daykind": "weekend", "season": "shoulder", "weather": "clear"},
        "surges": [{"poi_name": "Royal Fort", "from": 660, "to": 900, "factor": 2.3},
                   {"poi_name": "National Art Gallery", "from": 720, "to": 960, "factor": 2.2},
                   {"poi_name": "Botanical Garden", "from": 780, "to": 1000, "factor": 2.0}],
    },
    "S3_public_event": {
        "label": "Major public event in the Old Town",
        "context": {"daykind": "weekend", "season": "shoulder", "weather": "clear"},
        "public_events": [{"zone_name": "Old Town", "from": 840, "to": 1260, "factor": 1.9}],
        "timeline": [
            {"at": 840, "type": "closure", "poi_name": "Clock Tower", "until": 1260},
            {"at": 840, "type": "closure", "poi_name": "Old Palace", "until": 1260},
            {"at": 840, "type": "car_closure", "zone_name": "Old Town", "until": 1260},
        ],
    },
    "S4_transport_delay": {
        "label": "Transport delays and a line suspension",
        "context": {"daykind": "weekday", "season": "peak", "weather": "clear"},
        "timeline": [
            {"at": 660, "type": "line_delay", "line": "M1", "minutes": 25, "until": 960},
            {"at": 780, "type": "line_suspension", "line": "M2", "until": 930},
            {"at": 660, "type": "congestion", "factor": 1.5, "until": 960},
        ],
    },
    "S5_capacity_limit": {
        "label": "Attraction capacity limits (timed entry)",
        "context": {"daykind": "weekend", "season": "peak", "weather": "clear"},
        "capacity": [{"poi_name": "Royal Fort", "factor": 0.5}, {"poi_name": "Grand Temple", "factor": 0.5},
                     {"poi_name": "City Zoo", "factor": 0.6}, {"poi_name": "National Art Gallery", "factor": 0.6}],
    },
    "S6_rain": {
        "label": "Rain (60% forecast) and travellers changing plans",
        "context": {"daykind": "weekend", "season": "shoulder", "weather": "rain", "rain_prob": 0.6},
        "timeline": [{"at": 780, "type": "preference_change", "share": 0.15, "end_earlier": 75}],
    },
}


def _node(city: City, name: str) -> int:
    for p in city.pois:
        if p["name"] == name:
            return p["node"]
    raise KeyError(name)


def _zone(city: City, name: str) -> int:
    for z in city.zones:
        if z["name"] == name:
            return z["id"]
    raise KeyError(name)


def resolve(city: City, name: str) -> Dict[str, Any]:
    """Turn names into node / zone ids and build the known and true contexts."""
    spec = SCENARIOS[name]
    c = spec["context"]
    events = [{"type": "public_event", "zone": _zone(city, e["zone_name"]), "from": e["from"], "to": e["to"],
               "factor": e["factor"]} for e in spec.get("public_events", [])]
    surges = [{"poi": _node(city, s["poi_name"]), "from": s["from"], "to": s["to"], "factor": s["factor"]}
              for s in spec.get("surges", [])]
    known = DayContext(c["daykind"], c["season"], c["weather"], c.get("rain_prob"), events, [])
    true = DayContext(c["daykind"], c["season"], c["weather"], None, events, surges)
    capacity = {str(_node(city, x["poi_name"])): x["factor"] for x in spec.get("capacity", [])}
    timeline: List[Dict[str, Any]] = []
    for ev in spec.get("timeline", []):
        e = dict(ev)
        if "poi_name" in e:
            e["poi"] = _node(city, e.pop("poi_name"))
        if "zone_name" in e:
            e["zone"] = _zone(city, e.pop("zone_name"))
        timeline.append(e)
    return {"name": name, "label": spec["label"], "known": known, "true": true,
            "initial_conditions": {"capacity_factor": capacity}, "timeline": sorted(timeline, key=lambda e: e["at"])}
