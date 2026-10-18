r"""Crowd, waiting-time and transport-delay forecasts from Bayesian networks.

Crowd network. Each POI has its own historical baseline load for every slot;
the network learns how the day's context scales demand relative to that
baseline, and how demand and baseline together give the crowd level:

    TimeOfDay DayKind Season Weather Event Indoor Kind
         \______\______\______|______/______/______/
                           Demand  (relative to the POI's usual level)
                              \        Baseline (POI's usual load in this slot)
                               \      /
                                Crowd  (low / medium / high / very high)
                                  |      Popularity
                                  v     /
                                Wait  (0-5 / 5-15 / 15-30 / 30+ min)

Delay network: Weather, EventOnLine, Peak -> Delay (none / minor / major).

`CrowdForecaster.forecast_day(ctx)` turns a day context (calendar, season,
weather or a rain probability, announced events) into per-POI, per-slot
expected load ratios, P(very high crowd) and expected queue waits, which the
C++ engine uses as path / CSP costs. `HistoricalAverageForecaster` (the
baseline alone) is the ablation that ignores the context.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tourism.core.city import City
from tourism.m3_uncertainty.bayes_net import BayesianNetwork
from tourism.sim.truth import DAYKINDS, SEASONS, TOD_NAMES, WEATHERS, DayContext, queue_waits, tod_bin

CROWD_LEVELS = ["low", "medium", "high", "very_high"]
CROWD_EDGES = [0.4, 0.7, 1.0]
DEMAND_LEVELS = ["very_low", "low", "normal", "high", "very_high", "extreme"]
DEMAND_EDGES = [0.6, 0.85, 1.15, 1.5, 2.0]
BASELINE_LEVELS = ["<0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", ">=1.0"]
BASELINE_EDGES = [0.2, 0.4, 0.6, 0.8, 1.0]
WAIT_BUCKETS = ["0-5", "5-15", "15-30", "30+"]
WAIT_EDGES = [5, 15, 30]
DELAY_LEVELS = ["none", "minor", "major"]
POP_LEVELS = ["low", "mid", "high"]
KINDS = ["attraction", "restaurant", "local"]

VARIABLES = ["TimeOfDay", "DayKind", "Season", "Weather", "Event", "Indoor", "Kind", "Demand", "Baseline",
             "Crowd", "Popularity", "Wait"]


def _bin(x: float, edges: Sequence[float]) -> int:
    for i, e in enumerate(edges):
        if x < e:
            return i
    return len(edges)


def crowd_level(rho: float) -> int:
    return _bin(rho, CROWD_EDGES)


def wait_bucket(w: float) -> int:
    return _bin(w, WAIT_EDGES)


def pop_level(p: float) -> int:
    return 0 if p < 0.4 else (1 if p < 0.7 else 2)


def kind_index(kind: str) -> int:
    return 0 if kind == "attraction" else (1 if kind == "restaurant" else 2)


def event_near(city: City, poi: Dict[str, Any], t: float, events: Sequence[Dict[str, Any]]) -> int:
    node = city.nodes[poi["node"]]
    for ev in events:
        if not (ev["from"] <= t < ev["to"]):
            continue
        z = city.zones[ev["zone"]]
        if poi["zone"] == ev["zone"] or math.hypot(node["x"] - z["x"], node["y"] - z["y"]) < 2000:
            return 1
    return 0


def build_crowd_network() -> BayesianNetwork:
    bn = BayesianNetwork()
    bn.add_node("TimeOfDay", TOD_NAMES)
    bn.add_node("DayKind", DAYKINDS)
    bn.add_node("Season", SEASONS)
    bn.add_node("Weather", WEATHERS)
    bn.add_node("Event", ["none", "nearby"])
    bn.add_node("Indoor", ["outdoor", "indoor"])
    bn.add_node("Kind", KINDS)
    bn.add_node("Demand", DEMAND_LEVELS,
                ["TimeOfDay", "DayKind", "Season", "Weather", "Event", "Indoor", "Kind"],
                backoff=["DayKind", "Season", "Weather", "Indoor"], backoff_strength=20.0)
    bn.add_node("Baseline", BASELINE_LEVELS)
    bn.add_node("Crowd", CROWD_LEVELS, ["Demand", "Baseline"])
    bn.add_node("Popularity", POP_LEVELS)
    bn.add_node("Wait", WAIT_BUCKETS, ["Crowd", "Popularity"])
    return bn


def build_delay_network() -> BayesianNetwork:
    bn = BayesianNetwork()
    bn.add_node("Weather", WEATHERS)
    bn.add_node("EventOnLine", ["no", "yes"])
    bn.add_node("Peak", ["no", "yes"])
    bn.add_node("Delay", DELAY_LEVELS, ["Weather", "EventOnLine", "Peak"])
    return bn


@dataclass
class DayForecast:
    load: List[List[float]]          # expected load ratio (arrivals / capacity) per POI per slot
    wait: List[List[float]]          # expected queue wait (min)
    pvh: List[List[float]]           # P(crowd level = very high), i.e. P(at or over capacity)
    probs: Optional[List[List[List[float]]]] = None

    def copy(self) -> "DayForecast":
        return DayForecast([r[:] for r in self.load], [r[:] for r in self.wait], [r[:] for r in self.pvh],
                           None if self.probs is None else [[p[:] for p in r] for r in self.probs])


def expected_waits(city: City, load: List[List[float]]) -> List[List[float]]:
    return [queue_waits([l * p["capacity"] for l in load[i]], p["capacity"], city.slot_min)
            for i, p in enumerate(city.pois)]


class HistoricalAverageForecaster:
    """Each POI's average load in each slot over the training days. It is the
    Bayesian model's baseline and, on its own, the ablation that ignores the
    day's calendar, season, weather and events."""

    def __init__(self, city: City, mean_load: List[List[float]], level_freq: List[List[List[float]]]):
        self.city = city
        self.mean_load = mean_load
        self.level_freq = level_freq

    @classmethod
    def fit(cls, city: City, history_path: str, train_days: int) -> "HistoricalAverageForecaster":
        P, S = len(city.pois), city.n_slots
        sums = [[0.0] * S for _ in range(P)]
        cnt = [[0] * S for _ in range(P)]
        freq = [[[0.5] * 4 for _ in range(S)] for _ in range(P)]
        idx = city.poi_index
        with gzip.open(history_path, "rt", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if int(r["day"]) >= train_days:
                    continue
                i, s, rho = idx[int(r["poi"])], int(r["slot"]), float(r["rho"])
                sums[i][s] += rho
                cnt[i][s] += 1
                freq[i][s][crowd_level(rho)] += 1
        mean = [[sums[i][s] / cnt[i][s] if cnt[i][s] else 0.0 for s in range(S)] for i in range(P)]
        lf = [[[c / sum(f) for c in f] for f in row] for row in freq]
        return cls(city, mean, lf)

    def forecast_day(self, ctx: DayContext) -> DayForecast:
        load = [row[:] for row in self.mean_load]
        pvh = [[f[3] if load[i][s] > 0 else 0.0 for s, f in enumerate(self.level_freq[i])] for i in range(len(load))]
        return DayForecast(load, expected_waits(self.city, load), pvh, self.level_freq)

    def to_dict(self) -> Dict[str, Any]:
        return {"mean_load": [[round(v, 4) for v in r] for r in self.mean_load],
                "level_freq": [[[round(v, 4) for v in f] for f in r] for r in self.level_freq]}

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def from_dict(cls, city: City, d: Dict[str, Any]) -> "HistoricalAverageForecaster":
        return cls(city, d["mean_load"], d["level_freq"])

    @classmethod
    def load(cls, city: City, path: str) -> "HistoricalAverageForecaster":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(city, json.load(fh))


class CrowdForecaster:
    def __init__(self, city: City, crowd_bn: BayesianNetwork, delay_bn: BayesianNetwork,
                 baseline: HistoricalAverageForecaster, demand_mean: List[float], delay_mean: List[float]):
        self.city = city
        self.bn = crowd_bn
        self.delay_bn = delay_bn
        self.baseline = baseline
        self.demand_mean = demand_mean    # mean demand ratio within each Demand level
        self.delay_mean = delay_mean      # mean minutes per delay level
        self._cache: Dict[Tuple, List[float]] = {}

    # ------------------------------------------------------------------ training
    @classmethod
    def fit(cls, city: City, history_path: str, transit_path: str, train_days: int) -> "CrowdForecaster":
        baseline = HistoricalAverageForecaster.fit(city, history_path, train_days)
        bn = build_crowd_network()
        poi_by_node, idx = city.poi_by_node, city.poi_index
        dsum = [0.0] * len(DEMAND_LEVELS)
        dcnt = [0] * len(DEMAND_LEVELS)

        def rows():
            with gzip.open(history_path, "rt", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if int(r["day"]) >= train_days:
                        continue
                    p = poi_by_node[int(r["poi"])]
                    s = int(r["slot"])
                    base = baseline.mean_load[idx[p["node"]]][s]
                    if base < 0.03:
                        continue
                    t = city.slot_start(s) + city.slot_min / 2
                    ez = int(r["event_zone"])
                    events = [{"zone": ez, "from": int(r["event_from"]), "to": int(r["event_to"])}] if ez >= 0 else []
                    rho = float(r["rho"])
                    ratio = rho / base
                    d = _bin(ratio, DEMAND_EDGES)
                    dsum[d] += ratio
                    dcnt[d] += 1
                    yield (tod_bin(t), DAYKINDS.index(r["daykind"]), SEASONS.index(r["season"]),
                           WEATHERS.index(r["weather"]), event_near(city, p, t, events), int(p["indoor"]),
                           kind_index(p["kind"]), d, _bin(base, BASELINE_EDGES), crowd_level(rho),
                           pop_level(p["popularity"]), wait_bucket(float(r["wait"])))

        bn.fit_counts(VARIABLES, rows())
        demand_mean = [dsum[i] / dcnt[i] if dcnt[i] else [0.45, 0.72, 1.0, 1.3, 1.72, 2.3][i]
                       for i in range(len(DEMAND_LEVELS))]
        dbn = build_delay_network()
        lsum, lcnt = [0.0] * 3, [0] * 3
        drows = []
        with open(transit_path, "r", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if int(r["day"]) >= train_days:
                    continue
                d = float(r["delay_min"])
                lvl = 0 if d <= 0 else (1 if d < 12 else 2)
                lsum[lvl] += d
                lcnt[lvl] += 1
                drows.append((WEATHERS.index(r["weather"]), int(r["event_on_line"]), int(r["peak"]), lvl))
        dbn.fit_counts(["Weather", "EventOnLine", "Peak", "Delay"], drows)
        delay_mean = [lsum[i] / lcnt[i] if lcnt[i] else [0, 7, 22][i] for i in range(3)]
        return cls(city, bn, dbn, baseline, demand_mean, delay_mean)

    # ----------------------------------------------------------------- inference
    def _evidence(self, poi: Dict[str, Any], t: float, ctx: DayContext):
        ev = {"TimeOfDay": tod_bin(t), "DayKind": ctx.daykind, "Season": ctx.season,
              "Event": event_near(self.city, poi, t, ctx.events), "Indoor": int(poi["indoor"]),
              "Kind": kind_index(poi["kind"])}
        soft = None
        if ctx.rain_prob is None:
            ev["Weather"] = ctx.weather
        else:
            soft = {"Weather": [1 - ctx.rain_prob, ctx.rain_prob]}
        return ev, soft

    def demand_probs(self, poi: Dict[str, Any], t: float, ctx: DayContext) -> List[float]:
        ev, soft = self._evidence(poi, t, ctx)
        key = ("D", tuple(sorted(ev.items())), None if soft is None else tuple(soft["Weather"]))
        if key not in self._cache:
            self._cache[key] = self.bn.query("Demand", ev, soft)
        return self._cache[key]

    def crowd_probs(self, poi: Dict[str, Any], t: float, ctx: DayContext, base: float) -> List[float]:
        """P(Crowd | context, baseline): sums over the Demand node."""
        dp = self.demand_probs(poi, t, ctx)
        b = _bin(base, BASELINE_EDGES)
        out = [0.0] * len(CROWD_LEVELS)
        for d, pd in enumerate(dp):
            row = self.bn.cpt_row("Crowd", [d, b])
            for c, pc in enumerate(row):
                out[c] += pd * pc
        return out

    def wait_probs(self, poi: Dict[str, Any], t: float, ctx: DayContext, base: float) -> List[float]:
        crowd = self.crowd_probs(poi, t, ctx, base)
        pop = pop_level(poi["popularity"])
        out = [0.0] * len(WAIT_BUCKETS)
        for c, pc in enumerate(crowd):
            for w, pw in enumerate(self.bn.cpt_row("Wait", [c, pop])):
                out[w] += pc * pw
        return out

    def forecast_day(self, ctx: DayContext) -> DayForecast:
        city = self.city
        load, pvh, probs = [], [], []
        for i, p in enumerate(city.pois):
            lr, pr, pp = [], [], []
            for s in range(city.n_slots):
                t = city.slot_start(s) + city.slot_min / 2
                base = self.baseline.mean_load[i][s]
                if not (p["open_min"] <= t < p["close_min"]) or base <= 0:
                    lr.append(0.0); pr.append(0.0); pp.append([1.0, 0.0, 0.0, 0.0])
                    continue
                dp = self.demand_probs(p, t, ctx)
                lr.append(base * sum(pd * m for pd, m in zip(dp, self.demand_mean)))
                cp = self.crowd_probs(p, t, ctx, base)
                pr.append(cp[3])
                pp.append(cp)
            load.append(lr); pvh.append(pr); probs.append(pp)
        return DayForecast(load, expected_waits(city, load), pvh, probs)

    def delay_probs(self, weather: str, event_on_line: bool, peak: bool) -> List[float]:
        return self.delay_bn.query("Delay", {"Weather": weather, "EventOnLine": int(event_on_line), "Peak": int(peak)})

    def expected_delay(self, weather: str, event_on_line: bool, peak: bool) -> float:
        return sum(p * m for p, m in zip(self.delay_probs(weather, event_on_line, peak), self.delay_mean))

    # ----------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"crowd_bn": self.bn.to_dict(), "delay_bn": self.delay_bn.to_dict(),
                       "baseline": self.baseline.to_dict(), "demand_mean": self.demand_mean,
                       "delay_mean": self.delay_mean}, fh)

    @classmethod
    def load(cls, city: City, path: str) -> "CrowdForecaster":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return cls(city, BayesianNetwork.from_dict(d["crowd_bn"]), BayesianNetwork.from_dict(d["delay_bn"]),
                   HistoricalAverageForecaster.from_dict(city, d["baseline"]), d["demand_mean"], d["delay_mean"])


def evaluate_forecasters(city: City, history_path: str, train_days: int, models: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """MAE of the expected load ratio, multi-class Brier score and accuracy of
    the crowd level on the held-out days."""
    by_day: Dict[int, List[Dict[str, str]]] = {}
    with gzip.open(history_path, "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            d = int(r["day"])
            if d >= train_days:
                by_day.setdefault(d, []).append(r)
    stats = {name: {"mae": 0.0, "brier": 0.0, "accuracy": 0.0, "n": 0} for name in models}
    idx = city.poi_index
    for d, rows in sorted(by_day.items()):
        r0 = rows[0]
        ez = int(r0["event_zone"])
        events = [{"zone": ez, "from": int(r0["event_from"]), "to": int(r0["event_to"])}] if ez >= 0 else []
        ctx = DayContext(daykind=r0["daykind"], season=r0["season"], weather=r0["weather"], events=events)
        fcs = {name: m.forecast_day(ctx) for name, m in models.items()}
        for r in rows:
            i, s, rho = idx[int(r["poi"])], int(r["slot"]), float(r["rho"])
            lvl = crowd_level(rho)
            for name, fc in fcs.items():
                st = stats[name]
                st["mae"] += abs(fc.load[i][s] - rho)
                probs = fc.probs[i][s] if fc.probs else [0.25] * 4
                st["brier"] += sum((probs[l] - (1.0 if l == lvl else 0.0)) ** 2 for l in range(4))
                st["accuracy"] += 1.0 if max(range(4), key=lambda l: probs[l]) == lvl else 0.0
                st["n"] += 1
    out = {}
    for name, st in stats.items():
        n = max(1, st["n"])
        out[name] = {"mae_load": st["mae"] / n, "brier": st["brier"] / n, "accuracy": st["accuracy"] / n,
                     "samples": st["n"]}
    return out
