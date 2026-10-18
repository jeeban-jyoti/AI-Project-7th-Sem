"""Synthetic travellers: a visible profile (what the planner is told) and hidden
traits (how the simulated person actually reacts to suggestions)."""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

from tourism.core.city import City
from tourism.core.types import TravellerProfile
from tourism.m1_search.candidates import utility

ARCHETYPES: Dict[str, Dict[str, float]] = {
    "culture": {"history": .9, "art": .7, "architecture": .8, "spiritual": .4, "crafts": .3, "food": .3},
    "foodie": {"food": 1, "shopping": .5, "crafts": .5, "history": .3, "art": .2},
    "nature": {"nature": 1, "family": .4, "spiritual": .4, "food": .3},
    "family": {"family": 1, "nature": .6, "history": .3, "food": .4, "art": .2},
    "shopper": {"shopping": 1, "crafts": .7, "food": .5, "art": .3, "architecture": .2},
    "spiritual": {"spiritual": 1, "history": .6, "architecture": .6, "nature": .3},
    "explorer": {},
}

# Beta(a, b) parameters of the hidden willingness to accept sustainable alternatives
WILLINGNESS = {"low": (2.0, 6.0), "medium": (4.0, 4.0), "high": (6.0, 2.0)}


@dataclass
class HiddenTraits:
    willingness: float
    time_sensitivity: float
    quality_sensitivity: float
    fame_attachment: float
    car_preference: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def willingness_level(w: float) -> str:
    return "low" if w < 0.4 else ("medium" if w < 0.65 else "high")


def generate_travellers(city: City, n: int, seed: int, willingness: str = "mixed") -> List[Tuple[TravellerProfile, HiddenTraits]]:
    rng = random.Random(seed)
    out = []
    names = list(ARCHETYPES)
    for k in range(n):
        arch = rng.choice(names)
        base = ARCHETYPES[arch]
        interests = {}
        for t in city.themes:
            v = base.get(t, 0.5 if arch == "explorer" else 0.1) + rng.gauss(0, 0.12)
            interests[t] = round(min(1.0, max(0.0, v)), 3)
        accessible = rng.random() < 0.08
        car_ok = rng.random() < 0.6 and not accessible
        hotel = rng.choice(city.hotels)
        pace = rng.choice([3, 4, 4, 5])
        level = willingness if willingness in WILLINGNESS else rng.choice(list(WILLINGNESS))
        a, b = WILLINGNESS[level]
        traits = HiddenTraits(
            willingness=rng.betavariate(a, b),
            time_sensitivity=rng.uniform(0.5, 1.5),
            quality_sensitivity=rng.uniform(0.5, 1.5),
            fame_attachment=rng.uniform(0.0, 1.0),
            car_preference=rng.uniform(0.0, 1.0) if car_ok else 0.0,
        )
        profile = TravellerProfile(
            id="T%04d" % (k + 1),
            name="%s traveller %d" % (arch.capitalize(), k + 1),
            interests=interests,
            budget=float(rng.choice([1500, 2500, 3500, 5000, 7000])),
            start_node=hotel,
            end_node=hotel,
            start_min=rng.choice([540, 570, 600, 630]),
            end_min=rng.choice([1080, 1110, 1140, 1170, 1200]),
            max_walk_m=float(rng.randint(2500, 5000) if accessible else rng.randint(4000, 10000)),
            accessible_only=accessible,
            modes=["walk", "transit", "car"] if car_ok else ["walk", "transit"],
            max_activities=pace,
            lunch=rng.random() < 0.9,
            archetype=arch,
        )
        if rng.random() < 0.3:
            famous = [p for p in city.attractions() if p["popularity"] >= 0.8
                      and (p["accessible"] or not accessible)]
            if famous:
                best = max(famous, key=lambda p: utility(profile, p, city))
                profile.must_visit = [best["node"]]
        out.append((profile, traits))
    return out
