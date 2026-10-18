"""Builds the synthetic city (data/city.json).

The city is synthetic so the project runs anywhere without downloads, but its
schema mirrors what real open data gives you (OSM POIs and streets, GTFS
transit lines), so a real city can be dropped in by writing the same JSON.

Layout: famous sights sit in the central zones (Old Town, Museum Quarter,
Market District); each theme also has similar but less-visited options in
outer zones (Riverside, Hill Park, Craft Village, University Town), which is
what makes crowd-aware substitution possible.
"""
from __future__ import annotations

import json
import math
import os
import random
from typing import Any, Dict, List, Tuple

THEMES = ["history", "art", "nature", "food", "shopping", "spiritual", "architecture", "family", "crafts"]

ZONES = [
    ("Old Town", 5000, 4000),
    ("Riverside", 6600, 3100),
    ("Museum Quarter", 3700, 4900),
    ("Market District", 5700, 5300),
    ("Lakeside", 2100, 2400),
    ("Hill Park", 7900, 6300),
    ("Craft Village", 8700, 1700),
    ("University Town", 1700, 6500),
]

# name, zone, themes, popularity, rating, capacity/15 min, duration, price, indoor, open, close, ticketed, accessible
ATTRACTIONS: List[Tuple] = [
    ("Royal Fort", 0, {"history": 1, "architecture": .8}, .98, 4.7, 140, 105, 400, False, 540, 1080, True, True),
    ("Old Palace", 0, {"history": .9, "architecture": .9, "art": .3}, .90, 4.6, 110, 90, 350, True, 570, 1050, True, True),
    ("City History Museum", 2, {"history": 1, "art": .4, "family": .3}, .75, 4.5, 90, 90, 200, True, 600, 1080, True, True),
    ("Riverside Ruins", 1, {"history": .9, "nature": .4, "architecture": .4}, .40, 4.4, 70, 75, 100, False, 540, 1110, True, True),
    ("Hill Fortress", 5, {"history": .85, "nature": .5, "architecture": .5}, .35, 4.5, 60, 90, 120, False, 540, 1050, True, False),
    ("Heritage Walk House", 6, {"history": .8, "crafts": .5}, .20, 4.4, 35, 60, 80, True, 600, 1080, False, True),
    ("National Art Gallery", 2, {"art": 1, "architecture": .3}, .85, 4.6, 100, 90, 300, True, 600, 1080, True, True),
    ("Modern Art Hall", 2, {"art": .95}, .55, 4.4, 70, 75, 200, True, 630, 1110, True, True),
    ("Street Art Lane", 3, {"art": .8, "shopping": .3}, .45, 4.3, 80, 45, 0, False, 480, 1260, False, True),
    ("University Art Collection", 7, {"art": .9, "history": .3}, .20, 4.5, 40, 60, 50, True, 600, 1050, False, True),
    ("Riverside Sculpture Garden", 1, {"art": .7, "nature": .6}, .35, 4.4, 60, 60, 0, False, 540, 1140, False, True),
    ("Botanical Garden", 4, {"nature": 1, "family": .6}, .80, 4.6, 150, 90, 100, False, 480, 1080, True, True),
    ("Lakeside Promenade", 4, {"nature": .8, "family": .5}, .60, 4.5, 200, 60, 0, False, 480, 1260, False, True),
    ("Hill Park Viewpoint", 5, {"nature": .9, "architecture": .2}, .70, 4.7, 120, 60, 0, False, 480, 1200, False, False),
    ("Butterfly Reserve", 6, {"nature": .9, "family": .7}, .30, 4.5, 50, 75, 150, False, 540, 1050, True, True),
    ("University Arboretum", 7, {"nature": .85, "family": .3}, .20, 4.4, 80, 60, 0, False, 480, 1140, False, True),
    ("River Boat Ride", 1, {"nature": .6, "family": .7}, .75, 4.5, 60, 45, 250, False, 600, 1140, True, True),
    ("Grand Temple", 0, {"spiritual": 1, "architecture": .8, "history": .6}, .95, 4.8, 160, 60, 0, False, 480, 1260, False, True),
    ("Riverside Shrine", 1, {"spiritual": .9, "architecture": .5, "history": .4}, .40, 4.6, 70, 45, 0, False, 480, 1200, False, True),
    ("Hill Monastery", 5, {"spiritual": .95, "nature": .4, "history": .5}, .30, 4.6, 50, 60, 0, True, 540, 1080, False, False),
    ("Lotus Meditation Centre", 4, {"spiritual": .8, "nature": .3}, .25, 4.4, 40, 60, 50, True, 540, 1140, False, True),
    ("Clock Tower", 0, {"architecture": .9, "history": .5}, .85, 4.4, 90, 30, 50, False, 540, 1140, True, True),
    ("Old Bridge", 1, {"architecture": .8, "history": .6}, .60, 4.4, 150, 30, 0, False, 480, 1260, False, True),
    ("Colonial Library", 2, {"architecture": .85, "history": .6, "art": .3}, .35, 4.5, 50, 45, 0, True, 600, 1080, False, True),
    ("Merchant Mansions", 3, {"architecture": .8, "history": .5, "shopping": .2}, .30, 4.3, 60, 45, 60, True, 600, 1080, False, True),
    ("Science Centre", 2, {"family": 1, "art": .2}, .70, 4.5, 120, 120, 350, True, 600, 1080, True, True),
    ("City Zoo", 4, {"family": 1, "nature": .7}, .85, 4.4, 180, 120, 300, False, 540, 1050, True, True),
    ("Adventure Park", 5, {"family": .9, "nature": .5}, .45, 4.3, 80, 120, 400, False, 600, 1080, True, False),
    ("Toy and Puppet Museum", 3, {"family": .85, "crafts": .4, "art": .3}, .25, 4.4, 50, 60, 100, True, 600, 1080, False, True),
    ("Grand Bazaar", 3, {"shopping": 1, "food": .5, "crafts": .4}, .90, 4.4, 200, 75, 0, False, 600, 1260, False, True),
    ("Spice Market", 3, {"food": .9, "shopping": .6}, .65, 4.5, 120, 45, 0, False, 540, 1140, False, True),
    ("Night Food Street", 0, {"food": 1, "shopping": .3}, .70, 4.5, 150, 60, 0, False, 1020, 1260, False, True),
    ("Riverside Food Market", 1, {"food": .9, "shopping": .3}, .35, 4.5, 90, 45, 0, False, 660, 1200, False, True),
    ("Weavers' Hall", 6, {"crafts": 1, "history": .4, "shopping": .4}, .25, 4.6, 40, 60, 50, True, 600, 1080, False, True),
    ("Pottery Courtyard", 6, {"crafts": .95, "art": .5}, .20, 4.5, 30, 60, 0, False, 600, 1080, False, True),
]

# Independent local businesses and community experiences: name, zone, kind, themes, rating, capacity, duration, price, open, close
LOCAL: List[Tuple] = [
    ("Handloom Co-op Store", 6, "shop", {"shopping": .8, "crafts": .9}, 4.6, 25, 30, 0, 600, 1140),
    ("Village Cooking Class", 6, "experience", {"food": 1, "crafts": .4}, 4.8, 15, 90, 600, 600, 1020),
    ("Riverside Bookshop Cafe", 1, "shop", {"art": .4, "food": .5, "shopping": .5}, 4.5, 20, 30, 150, 540, 1200),
    ("Hill Tea Garden", 5, "experience", {"nature": .6, "food": .6}, 4.6, 25, 45, 120, 540, 1080),
    ("Guided Lane Walk", 3, "experience", {"history": .7, "architecture": .6, "food": .3}, 4.7, 20, 75, 250, 540, 1080),
    ("Artisan Jewellery Studio", 3, "shop", {"crafts": .9, "shopping": .7, "art": .4}, 4.5, 15, 30, 0, 600, 1200),
    ("Lakeside Kayak Rentals", 4, "experience", {"nature": .8, "family": .6}, 4.5, 20, 60, 300, 540, 1080),
    ("Student Theatre Matinee", 7, "experience", {"art": .8, "family": .3}, 4.4, 40, 90, 150, 840, 1110),
    ("Old Town Pottery Workshop", 0, "experience", {"crafts": 1, "art": .5}, 4.6, 15, 60, 300, 600, 1140),
    ("Print Studio", 2, "shop", {"art": .8, "crafts": .6}, 4.4, 15, 30, 0, 600, 1140),
]

# name, zone, independent, popularity, rating, capacity, price
RESTAURANTS: List[Tuple] = [
    ("Fort View Diner", 0, False, .80, 4.0, 60, 600),
    ("Burger Plaza", 0, False, .70, 3.8, 80, 400),
    ("Temple Street Thali", 0, True, .75, 4.6, 50, 350),
    ("Gallery Cafe", 2, False, .60, 4.2, 40, 450),
    ("Pizza Corner", 2, False, .50, 3.9, 50, 450),
    ("Riverside Fish House", 1, True, .50, 4.6, 40, 500),
    ("Spice Route Bistro", 1, True, .40, 4.5, 35, 450),
    ("Market Biryani House", 3, True, .70, 4.7, 50, 300),
    ("Food Court Central", 3, False, .60, 3.9, 100, 350),
    ("Lakeside Grill", 4, False, .50, 4.1, 60, 550),
    ("Garden Kitchen", 4, True, .30, 4.5, 30, 400),
    ("Hilltop Dhaba", 5, True, .30, 4.6, 30, 250),
    ("Craft Village Kitchen", 6, True, .20, 4.7, 25, 250),
    ("Campus Canteen Co-op", 7, True, .20, 4.3, 40, 150),
]

HOTELS = [("Central Hotel", 0), ("Riverside Inn", 1), ("Museum Suites", 2), ("Lakeside Resort", 4),
          ("Hill Lodge", 5), ("Campus Guesthouse", 7)]

# id, name, kind, zone sequence, speed km/h, headway min, fare base, fare per km, gCO2 per passenger-km
LINES = [
    ("M1", "Blue Metro", "metro", [4, 2, 0, 3, 5], 34, 6, 10, 1.5, 25),
    ("M2", "Red Metro", "metro", [7, 2, 0, 1, 6], 34, 6, 10, 1.5, 25),
    ("T1", "Green Tram", "tram", [4, 0, 1, 5], 20, 8, 8, 1.0, 35),
    ("B7", "Bus 7", "bus", [6, 1, 3, 7], 18, 12, 10, 1.0, 70),
]

PARAMS = {
    "walk_m_per_min": 75,
    "wheelchair_m_per_min": 55,
    "car_kmh_offpeak": 28,
    "car_kmh_peak": 14,
    "car_access_min": 6,
    "taxi_base": 50,
    "taxi_per_km": 16,
    "car_co2_g_pkm": 160,
    "peak_windows": [[510, 630], [1020, 1170]],
}


def _theme_vec(d: Dict[str, float]) -> List[float]:
    return [float(d.get(t, 0.0)) for t in THEMES]


def generate_city(seed: int = 7) -> Dict[str, Any]:
    rng = random.Random(seed)
    nodes: List[Dict[str, Any]] = []
    pois: List[Dict[str, Any]] = []

    def add_node(name: str, ntype: str, x: float, y: float, zone: int, accessible: bool = True) -> int:
        nodes.append({"id": len(nodes), "name": name, "type": ntype, "x": round(x, 1), "y": round(y, 1),
                      "zone": zone, "accessible": accessible})
        return len(nodes) - 1

    def place(zone: int, spread: float = 480.0) -> Tuple[float, float]:
        _, cx, cy = ZONES[zone]
        return cx + rng.gauss(0, spread), cy + rng.gauss(0, spread)

    for (name, zone, themes, pop, rating, cap, dur, price, indoor, op, cl, ticketed, acc) in ATTRACTIONS:
        x, y = place(zone)
        n = add_node(name, "poi", x, y, zone, acc)
        pois.append({"node": n, "name": name, "kind": "attraction", "zone": zone, "themes": _theme_vec(themes),
                     "popularity": pop, "rating": rating, "capacity": cap, "duration_min": dur, "price": price,
                     "indoor": indoor, "open_min": op, "close_min": cl, "ticketed": ticketed, "accessible": acc,
                     "independent": False})
    for (name, zone, kind, themes, rating, cap, dur, price, op, cl) in LOCAL:
        x, y = place(zone, 420)
        n = add_node(name, "poi", x, y, zone, True)
        pois.append({"node": n, "name": name, "kind": kind, "zone": zone, "themes": _theme_vec(themes),
                     "popularity": 0.15 + 0.1 * rng.random(), "rating": rating, "capacity": max(45, 3 * cap), "duration_min": dur,
                     "price": price, "indoor": kind == "shop", "open_min": op, "close_min": cl,
                     "ticketed": False, "accessible": zone != 5, "independent": True})
    for (name, zone, indep, pop, rating, cap, price) in RESTAURANTS:
        x, y = place(zone, 420)
        n = add_node(name, "poi", x, y, zone, True)
        themes = {"food": 1.0}
        pois.append({"node": n, "name": name, "kind": "restaurant", "zone": zone, "themes": _theme_vec(themes),
                     "popularity": pop, "rating": rating, "capacity": 2 * cap, "duration_min": 50, "price": price,
                     "indoor": True, "open_min": 690, "close_min": 1260, "ticketed": False, "accessible": True,
                     "independent": indep})
    hotels = []
    for (name, zone) in HOTELS:
        x, y = place(zone, 350)
        hotels.append(add_node(name, "hotel", x, y, zone, True))

    # transit: one interchange station per zone centre, one intermediate stop per line segment
    station = {}
    for z, (zname, cx, cy) in enumerate(ZONES):
        station[z] = add_node(zname + " Station", "stop", cx + rng.gauss(0, 60), cy + rng.gauss(0, 60), z, True)
    lines = []
    edges: List[Dict[str, Any]] = []
    for (lid, lname, kind, seq, speed, headway, fare_base, fare_km, co2) in LINES:
        stops = []
        for i, z in enumerate(seq):
            stops.append(station[z])
            if i + 1 < len(seq):
                a, b = nodes[station[z]], nodes[station[seq[i + 1]]]
                mx, my = (a["x"] + b["x"]) / 2 + rng.gauss(0, 150), (a["y"] + b["y"]) / 2 + rng.gauss(0, 150)
                mz = z if math.hypot(mx - a["x"], my - a["y"]) < math.hypot(mx - b["x"], my - b["y"]) else seq[i + 1]
                stops.append(add_node("%s stop %d" % (lname, i + 1), "stop", mx, my, mz, True))
        for u, v in zip(stops, stops[1:]):
            d = _dist(nodes, u, v) * 1.1
            edges.append({"u": u, "v": v, "mode": "transit", "line": lid, "dist_m": round(d, 1), "accessible": True})
        lines.append({"id": lid, "name": lname, "kind": kind, "stops": stops, "speed_kmh": speed,
                      "headway_min": headway, "fare_base": fare_base, "fare_per_km": fare_km, "co2_g_pkm": co2})

    n = len(nodes)
    # walking network: neighbours within 1.3 km, at least the 2 nearest
    walk_pairs = set()
    for u in range(n):
        order = sorted((v for v in range(n) if v != u), key=lambda v: _dist(nodes, u, v))
        for k, v in enumerate(order):
            if _dist(nodes, u, v) <= 1300 or k < 2:
                walk_pairs.add((min(u, v), max(u, v)))
    # make the walk + transit network connected
    _connect(nodes, walk_pairs, edges)
    for (u, v) in sorted(walk_pairs):
        steep = (nodes[u]["zone"] == 5 or nodes[v]["zone"] == 5) and rng.random() < 0.5
        edges.append({"u": u, "v": v, "mode": "walk", "dist_m": round(_dist(nodes, u, v) * 1.25, 1),
                      "accessible": not steep and nodes[u]["accessible"] and nodes[v]["accessible"]})
    # road network for taxis / cars
    car_pairs = set()
    for u in range(n):
        order = sorted((v for v in range(n) if v != u), key=lambda v: _dist(nodes, u, v))
        for k, v in enumerate(order):
            if _dist(nodes, u, v) <= 2500 or k < 4:
                car_pairs.add((min(u, v), max(u, v)))
    for (u, v) in sorted(car_pairs):
        edges.append({"u": u, "v": v, "mode": "car", "dist_m": round(_dist(nodes, u, v) * 1.3, 1), "accessible": True})

    return {
        "name": "Synthetic City (crowd-aware tourism planner)",
        "version": 1,
        "seed": seed,
        "slot_min": 15,
        "day_start_min": 480,
        "day_end_min": 1260,
        "themes": THEMES,
        "params": PARAMS,
        "zones": [{"id": i, "name": z[0], "x": z[1], "y": z[2]} for i, z in enumerate(ZONES)],
        "nodes": nodes,
        "lines": lines,
        "edges": edges,
        "pois": pois,
        "hotels": hotels,
    }


def _dist(nodes: List[Dict[str, Any]], a: int, b: int) -> float:
    return math.hypot(nodes[a]["x"] - nodes[b]["x"], nodes[a]["y"] - nodes[b]["y"])


def _connect(nodes: List[Dict[str, Any]], walk_pairs: set, edges: List[Dict[str, Any]]) -> None:
    n = len(nodes)
    while True:
        adj = {i: set() for i in range(n)}
        for (u, v) in walk_pairs:
            adj[u].add(v); adj[v].add(u)
        for e in edges:
            adj[e["u"]].add(e["v"]); adj[e["v"]].add(e["u"])
        seen = {0}
        stack = [0]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); stack.append(v)
        if len(seen) == n:
            return
        rest = [v for v in range(n) if v not in seen]
        u, v = min(((a, b) for a in seen for b in rest), key=lambda p: _dist(nodes, p[0], p[1]))
        walk_pairs.add((min(u, v), max(u, v)))


def write_city(path: str, seed: int = 7) -> Dict[str, Any]:
    city = generate_city(seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(city, fh, indent=1)
    return city
