"""Traveller utility for POIs and candidate selection for the CSP."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from tourism.core.city import City
from tourism.core.types import TravellerProfile


def interest_match(profile: TravellerProfile, poi: Dict[str, Any], city: City) -> float:
    """Weighted average of the traveller's interests over the POI's themes (0..1)."""
    themes = poi["themes"]
    total = sum(themes) or 1.0
    return sum(profile.interests.get(t, 0.0) * w for t, w in zip(city.themes, themes)) / total


def quality(poi: Dict[str, Any]) -> float:
    """Rating and fame: famous landmarks are what most people want to see,
    which is exactly why recommending them to everyone creates crowds."""
    r = min(1.0, max(0.0, (poi["rating"] - 3.5) / 1.5))
    return 0.45 + 0.2 * r + 0.35 * poi["popularity"]


def utility(profile: TravellerProfile, poi: Dict[str, Any], city: City) -> float:
    """Expected enjoyment of a visit on a 0..10 scale (fame counts a little:
    people do enjoy seeing the famous sights). Cached per profile object."""
    cache = profile.__dict__.setdefault("_utility_cache", {})
    u = cache.get(poi["node"])
    if u is None:
        if poi["kind"] == "restaurant":
            u = round(2.5 + 4.0 * max(0.2, profile.interests.get("food", 0.3)) * quality(poi), 3)
        else:
            u = round(10.0 * interest_match(profile, poi, city) * quality(poi), 3)
        cache[poi["node"]] = u
    return u


def _allowed(profile: TravellerProfile, poi: Dict[str, Any]) -> bool:
    return poi["accessible"] or not profile.accessible_only


def to_candidate(profile: TravellerProfile, poi: Dict[str, Any], city: City, must: bool = False,
                 u: Optional[float] = None) -> Dict[str, Any]:
    return {
        "poi": poi["node"],
        "utility": utility(profile, poi, city) if u is None else u,
        "duration_min": poi["duration_min"],
        "price": poi["price"],
        "must": must,
        "restaurant": poi["kind"] == "restaurant",
        "local": bool(poi["independent"]),
    }


def select_candidates(profile: TravellerProfile, city: City, planning: Dict[str, Any],
                      exclude: Iterable[int] = (), extra: Iterable[int] = ()) -> List[Dict[str, Any]]:
    """Top attractions by utility, the best local businesses, a few restaurants
    and every must-visit. `extra` forces POIs in (e.g. an accepted substitute)."""
    exclude = set(exclude)
    extra = [e for e in extra if e not in exclude]
    must = set(profile.must_visit) | set(extra)
    chosen: Dict[int, Dict[str, Any]] = {}

    activities = [p for p in city.pois if p["kind"] != "restaurant" and _allowed(profile, p) and p["node"] not in exclude]
    activities.sort(key=lambda p: -utility(profile, p, city))
    for p in activities[: planning.get("candidates", 12)]:
        chosen[p["node"]] = to_candidate(profile, p, city)

    locals_ = [p for p in activities if p["independent"] and p["node"] not in chosen]
    for p in locals_[: planning.get("local_businesses", 3)]:
        chosen[p["node"]] = to_candidate(profile, p, city)

    if profile.lunch:
        home = profile.start_node
        rests = [p for p in city.restaurants() if p["node"] not in exclude]
        rests.sort(key=lambda p: -(utility(profile, p, city) - 0.4 * city.dist(home, p["node"]) / 1000.0))
        for p in rests[: planning.get("restaurants", 3)]:
            chosen[p["node"]] = to_candidate(profile, p, city)

    for node in must:
        poi = city.poi_by_node.get(node)
        if poi is not None and node not in exclude:
            chosen[node] = to_candidate(profile, poi, city, must=True)
    return list(chosen.values())


def similar_alternatives(profile: TravellerProfile, poi: Dict[str, Any], city: City, k: int = 3,
                         exclude: Iterable[int] = (), min_similarity: float = 0.6) -> List[Dict[str, Any]]:
    """Less-visited POIs of the same kind that match the same interests."""
    exclude = set(exclude) | {poi["node"]}
    pool = []
    for q in city.pois:
        if q["node"] in exclude or not _allowed(profile, q):
            continue
        if (q["kind"] == "restaurant") != (poi["kind"] == "restaurant"):
            continue
        sim = city.theme_similarity(poi, q)
        if sim < min_similarity:
            continue
        pool.append((sim * utility(profile, q, city) - 0.15 * city.dist(poi["node"], q["node"]) / 1000.0, q))
    pool.sort(key=lambda x: -x[0])
    return [q for _, q in pool[:k]]
