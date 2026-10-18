"""The hidden 'true' crowd and delay process.

Used to generate historical data (which the Bayesian model learns from) and to
drive the simulated test days. The planner never reads this module directly:
it only sees history, forecasts and noisy observations.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

TOD_EDGES = [480, 600, 720, 840, 960, 1080, 1261]
TOD_NAMES = ["08-10", "10-12", "12-14", "14-16", "16-18", "18-21"]
DAYKINDS = ["weekday", "weekend", "holiday"]
SEASONS = ["off", "shoulder", "peak"]
WEATHERS = ["clear", "rain"]

DAY_FACTOR = {"weekday": 0.62, "weekend": 1.0, "holiday": 1.15}
SEASON_FACTOR = {"off": 0.6, "shoulder": 0.82, "peak": 1.08}


def tod_bin(t: float) -> int:
    for i in range(len(TOD_EDGES) - 1):
        if TOD_EDGES[i] <= t < TOD_EDGES[i + 1]:
            return i
    return len(TOD_NAMES) - 1 if t >= TOD_EDGES[-1] else 0


@dataclass
class DayContext:
    """What is known about a day in advance (calendar, weather forecast, planned
    events). `surges` are unannounced crowd formations: the forecaster does not
    know about them and has to learn them from observations."""
    daykind: str = "weekday"
    season: str = "shoulder"
    weather: str = "clear"
    rain_prob: Optional[float] = None
    events: List[Dict[str, Any]] = field(default_factory=list)   # public events: zone, from, to, factor
    surges: List[Dict[str, Any]] = field(default_factory=list)   # hidden: poi, from, to, factor

    def to_dict(self) -> Dict[str, Any]:
        return {"daykind": self.daykind, "season": self.season, "weather": self.weather,
                "rain_prob": self.rain_prob, "events": self.events, "surges": self.surges}


def poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    if lam > 30:
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def queue_waits(arrivals: List[float], capacity: float, slot_min: int) -> List[float]:
    """Fluid queue: backlog carries over between slots, wait = backlog / service rate."""
    q, waits = 0.0, []
    for a in arrivals:
        waits.append(slot_min * max(0.0, q + 0.5 * a - 0.5 * capacity) / capacity)
        q = max(0.0, q + a - capacity)
    return waits


def hour_profile(kind: str, t: float, evening_venue: bool = False) -> float:
    h = t / 60.0
    if evening_venue:
        return 0.2 + 0.8 * math.exp(-((h - 19.5) / 1.3) ** 2)
    if kind == "attraction":
        v = 0.22 + 0.78 * math.exp(-((h - 11.75) / 1.6) ** 2) + 0.55 * math.exp(-((h - 15.5) / 1.4) ** 2)
        if h > 18:
            v *= max(0.3, 1 - (h - 18) * 0.25)
        return min(1.0, v)
    if kind == "restaurant":
        return min(1.0, 0.12 + 0.88 * math.exp(-((h - 13.0) / 0.8) ** 2) + 0.75 * math.exp(-((h - 19.75) / 0.9) ** 2))
    return min(1.0, 0.3 + 0.5 * math.exp(-((h - 17) / 2.0) ** 2) + 0.25 * math.exp(-((h - 12) / 1.5) ** 2))


class TruthModel:
    def __init__(self, city):
        self.city = city
        self.slots = city.n_slots
        self.base = []
        self.profile = []
        for p in city.pois:
            self.base.append(0.12 + 0.78 * p["popularity"] ** 1.6)
            evening = p["open_min"] >= 960
            prof = []
            for s in range(self.slots):
                t = city.slot_start(s) + city.slot_min / 2
                open_ = p["open_min"] <= t < p["close_min"]
                prof.append(hour_profile(p["kind"], t, evening) if open_ else 0.0)
            self.profile.append(prof)

    def event_factor(self, poi: Dict[str, Any], t: float, ctx: DayContext, include_surges: bool = True) -> float:
        f = 1.0
        for ev in ctx.events:
            if not (ev["from"] <= t < ev["to"]):
                continue
            zc = self.city.zones[ev["zone"]]
            node = self.city.nodes[poi["node"]]
            d = math.hypot(node["x"] - zc["x"], node["y"] - zc["y"])
            if poi["zone"] == ev["zone"]:
                f *= ev.get("factor", 1.6)
            elif d < 2000:
                f *= 1 + (ev.get("factor", 1.6) - 1) * 0.4
        if include_surges:
            for sg in ctx.surges:
                if sg["poi"] == poi["node"] and sg["from"] <= t < sg["to"]:
                    f *= sg.get("factor", 1.9)
        return f

    def expected_rates(self, ctx: DayContext, include_surges: bool = True) -> List[List[float]]:
        out = []
        for i, p in enumerate(self.city.pois):
            w = 1.0
            if ctx.weather == "rain":
                w = 1.35 if p["indoor"] else 0.5
            k = p["capacity"] * self.base[i] * DAY_FACTOR[ctx.daykind] * SEASON_FACTOR[ctx.season] * w
            row = []
            for s in range(self.slots):
                t = self.city.slot_start(s) + self.city.slot_min / 2
                row.append(k * self.profile[i][s] * self.event_factor(p, t, ctx, include_surges))
            out.append(row)
        return out

    def sample_day(self, ctx: DayContext, rng: random.Random) -> List[List[int]]:
        """Background (non-routed) arrivals per POI per slot for one day."""
        rates = self.expected_rates(ctx)
        out = []
        for row in rates:
            noise = math.exp(rng.gauss(0, 0.12))
            out.append([poisson(rng, r * noise) for r in row])
        return out

    @staticmethod
    def delay_probs(weather: str, event_on_line: bool, peak: bool) -> List[float]:
        minor, major = 0.08, 0.02
        mult = (2.0 if weather == "rain" else 1.0) * (1.5 if event_on_line else 1.0) * (1.5 if peak else 1.0)
        minor, major = min(0.5, minor * mult), min(0.25, major * mult)
        return [1 - minor - major, minor, major]

    def sample_delay(self, rng: random.Random, weather: str, event_on_line: bool, peak: bool) -> float:
        p = self.delay_probs(weather, event_on_line, peak)
        u = rng.random()
        if u < p[0]:
            return 0.0
        if u < p[0] + p[1]:
            return rng.uniform(4, 10)
        return rng.uniform(15, 30)
