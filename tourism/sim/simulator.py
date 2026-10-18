"""Simulates one day of the city with every traveller following their plan.

Per 15-minute slot:
  1. events of the scenario timeline happen (closures, delays, preference changes)
  2. live crowd sensors report noisy counts -> Gamma-Poisson update -> surges
  3. travellers advance through their actions; arrivals join POI queues
  4. queues update (background + routed visitors vs capacity) -> waiting times
  5. arrivals are resolved: visit, or fail (closed, sold out, closing time)
Systems with replanning re-plan affected travellers at their next step.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from tourism.core.city import City
from tourism.core.types import PlanResult, ReasonRecord, TravellerProfile, fmt_time
from tourism.data.personas import HiddenTraits
from tourism.m1_search.candidates import utility
from tourism.m3_uncertainty.online import OnlineCrowdUpdater
from tourism.m4_rl.matrix import TravelMatrix
from tourism.sim.truth import TruthModel

VISITS = ("visit", "eat", "local_experience")
TRANSIT = ("metro", "tram", "bus")


def _binomial(rng: random.Random, n: int, p: float) -> int:
    if n <= 30:
        return sum(1 for _ in range(n) if rng.random() < p)
    return max(0, min(n, int(round(rng.gauss(n * p, math.sqrt(n * p * (1 - p)))))))


class TravellerRun:
    def __init__(self, profile: TravellerProfile, traits: HiddenTraits, result: PlanResult):
        self.profile = profile
        self.traits = traits
        self.result = result
        self.actions: List[Dict[str, Any]] = list(result.plan.get("actions", [])) if result.status == "ok" else []
        self.i = 0
        self.node = profile.start_node
        self.t = float(profile.start_min)
        self.end_min = profile.end_min
        self.energy, self.hunger, self.walked, self.lunch = 100.0, 20.0, 0.0, False
        self.money = 0.0
        self.co2 = 0.0
        self.travel_min = 0.0
        self.dist = {"walk": 0.0, "transit": 0.0, "car": 0.0}
        self.visits: List[Dict[str, Any]] = []
        self.failures: List[Dict[str, Any]] = []
        self.pending: Optional[Tuple[Dict[str, Any], int]] = None
        self.replan_request: Optional[Dict[str, Any]] = None
        self.replans: List[Dict[str, Any]] = []
        self.finished = result.status != "ok"
        self.dropped = 0

    def remaining_visits(self) -> List[Dict[str, Any]]:
        return [a for a in self.actions[self.i:] if a["type"] in VISITS]


def satisfaction(visits: List[Dict[str, Any]], travel_min: float) -> float:
    """Realized satisfaction: each visit's utility, reduced when the place is
    over-crowded (up to half its value), minus queueing and travel time."""
    total = 0.0
    for v in visits:
        discount = min(0.5, 0.6 * max(0.0, v["rho"] - 0.9))
        total += v["utility"] * (1.0 - discount) - 0.06 * v["wait"]
    return total - 0.01 * travel_min


def gini_coefficient(values: List[float]) -> float:
    vals = sorted(values)
    n, tot = len(vals), sum(vals)
    if n == 0 or tot <= 0:
        return 0.0
    cum = sum((k + 1) * v for k, v in enumerate(vals))
    return (2 * cum) / (n * tot) - (n + 1) / n


class DaySimulator:
    def __init__(self, city: City, scenario: Dict[str, Any], seed: int, cfg: Dict[str, Any], matrix: TravelMatrix):
        self.city = city
        self.sc = scenario
        self.cfg = cfg
        self.matrix = matrix
        self.weight = cfg["simulation"]["traveller_weight"]
        self.noise = cfg["simulation"]["travel_noise"]
        self.truth = TruthModel(city)
        # common random numbers: every system sees the same background crowds and sensors
        self.bg = self.truth.sample_day(scenario["true"], random.Random(seed * 1000 + 17))
        self.rng = random.Random(seed * 1000 + 29)
        srng = random.Random(seed * 1000 + 41)
        self.sensors = {i for i in range(len(city.pois)) if srng.random() < cfg["simulation"]["sensor_coverage"]}
        P, S = len(city.pois), city.n_slots
        self.routed = [[0.0] * S for _ in range(P)]
        self.wait = [[0.0] * S for _ in range(P)]
        self.load = [[0.0] * S for _ in range(P)]
        self.queue = [0.0] * P
        self.cond: Dict[str, Any] = {"closed_pois": [], "capacity_factor": dict(scenario["initial_conditions"].get("capacity_factor", {})),
                                     "line_delays": [], "car_closed_zones": [], "car_congestion": 1.0}
        self.closures: Dict[int, int] = {}
        self.car_closed: Dict[int, int] = {}
        self.congestion: Tuple[float, int] = (1.0, 0)
        self.log: List[str] = []

    # -------------------------------------------------------------- conditions
    def capfac(self, i: int) -> float:
        return float(self.cond["capacity_factor"].get(str(self.city.pois[i]["node"]), 1.0))

    def closed(self, poi: int, t: float) -> bool:
        return poi in self.closures and t < self.closures[poi]

    def line_state(self, line: str, t: float) -> Tuple[float, bool]:
        delay, susp = 0.0, False
        for d in self.cond["line_delays"]:
            if d["line"] == line and d["from"] <= t < d["to"]:
                delay += d.get("minutes", 0.0)
                susp = susp or d.get("suspended", False)
        return delay, susp

    def _apply_event(self, ev: Dict[str, Any], runs: List[TravellerRun], replan: bool) -> None:
        typ = ev["type"]
        affected: List[TravellerRun] = []
        if typ == "closure":
            self.closures[ev["poi"]] = ev["until"]
            self.cond["closed_pois"] = sorted(set(self.cond["closed_pois"]) | {ev["poi"]})
            affected = [r for r in runs if any(a["poi"] == ev["poi"] for a in r.remaining_visits())]
            desc = "%s closed" % self.city.name_of(ev["poi"])
        elif typ == "car_closure":
            self.car_closed[ev["zone"]] = ev["until"]
            self.cond["car_closed_zones"] = sorted(set(self.cond["car_closed_zones"]) | {ev["zone"]})
            affected = [r for r in runs if self._uses_car_in_zone(r, ev["zone"])]
            desc = "roads closed in %s" % self.city.zone_name(ev["zone"])
        elif typ in ("line_delay", "line_suspension"):
            self.cond["line_delays"].append({"line": ev["line"], "minutes": ev.get("minutes", 0), "from": ev["at"],
                                             "to": ev["until"], "suspended": typ == "line_suspension"})
            affected = [r for r in runs if self._uses_line(r, ev["line"])]
            desc = "%s %s" % (self.city.line_by_id[ev["line"]]["name"], "suspended" if typ == "line_suspension" else "delayed")
        elif typ == "congestion":
            self.congestion = (ev["factor"], ev["until"])
            self.cond["car_congestion"] = ev["factor"]
            desc = "heavy road traffic"
        elif typ == "preference_change":
            pool = [r for r in runs if not r.finished]
            k = int(round(ev["share"] * len(pool)))
            for r in self.rng.sample(pool, min(k, len(pool))):
                r.end_min = max(int(r.t) + 30, r.end_min - ev["end_earlier"])
                affected.append(r)
            desc = "you want to finish earlier"
        else:
            return
        self.log.append("%s %s (%d travellers affected)" % (fmt_time(ev["at"]), desc, len(affected)))
        if replan:
            for r in affected:
                r.replan_request = {"trigger": desc, "time": ev["at"]}

    def _uses_line(self, r: TravellerRun, line: str) -> bool:
        return any(s.get("line") == line for a in r.actions[r.i:] if a["type"] == "travel" for s in a.get("segments", []))

    def _uses_car_in_zone(self, r: TravellerRun, zone: int) -> bool:
        for a in r.actions[r.i:]:
            if a["type"] != "travel":
                continue
            for s in a.get("segments", []):
                if s["mode"] == "car" and (self.city.nodes[s["from"]]["zone"] == zone or self.city.nodes[s["to"]]["zone"] == zone):
                    return True
        return False

    # -------------------------------------------------------------- execution
    def _travel(self, r: TravellerRun, a: Dict[str, Any]) -> None:
        dur = a["end"] - a["start"]
        delay = 0.0
        total = sum(s["dist_m"] for s in a.get("segments", [])) or 1.0
        weather = self.sc["true"].weather
        for s in a.get("segments", []):
            d = s["dist_m"]
            if s["mode"] in TRANSIT:
                extra, susp = self.line_state(s.get("line", ""), r.t)
                if susp:  # falls back to a taxi
                    delay += 20.0
                    r.co2 += d / 1000.0 * self.city.params["car_co2_g_pkm"] / 1000.0
                    r.dist["car"] += d
                    continue
                peak = any(w[0] <= r.t < w[1] for w in self.city.params["peak_windows"])
                delay += extra + self.truth.sample_delay(self.rng, weather, False, peak) * 0.5
                r.co2 += s.get("co2_kg", 0.0)
                r.dist["transit"] += d
            elif s["mode"] == "car":
                f, until = self.congestion
                if f > 1.0 and r.t < until:
                    delay += dur * (f - 1.0) * d / total
                for zone, until_z in self.car_closed.items():
                    if r.t < until_z and (self.city.nodes[s["from"]]["zone"] == zone or self.city.nodes[s["to"]]["zone"] == zone):
                        delay += 15.0
                r.co2 += d / 1000.0 * self.city.params["car_co2_g_pkm"] / 1000.0
                r.dist["car"] += d
            else:
                r.dist["walk"] += d
        actual = dur * math.exp(self.rng.gauss(0, self.noise)) + delay
        r.t += actual
        r.travel_min += actual
        r.node = a["to"]
        r.money += a.get("money", 0.0)
        r.walked += a.get("walk_m", 0.0)
        r.energy -= 0.15 * actual
        r.hunger += 0.1 * actual

    def _go_home(self, r: TravellerRun) -> None:
        green = "car" not in r.profile.modes
        if r.node != r.profile.end_node:
            tt = self.matrix.time(r.node, r.profile.end_node, r.t, green)
            r.co2 += self.matrix.co2(r.node, r.profile.end_node, r.t, green)
            r.dist["transit"] += self.matrix.get("dist", r.node, r.profile.end_node, r.t, green)
            r.t += tt
            r.travel_min += tt
            r.node = r.profile.end_node
        r.finished = True

    def advance(self, r: TravellerRun, t_limit: float, coord, day, replan: bool) -> None:
        while not r.finished and r.pending is None and r.t < t_limit:
            if r.replan_request is not None and replan:
                self._replan(r, coord, day)
            if r.i >= len(r.actions):
                r.finished = True
                break
            a = r.actions[r.i]
            typ = a["type"]
            # a traveller who now wants to stop early heads home (systems that cannot replan)
            if typ in VISITS + ("travel",) and r.t > r.end_min - 20 and not (typ == "travel" and a["to"] == r.profile.end_node):
                r.dropped += len(r.remaining_visits())
                self._go_home(r)
                break
            if typ == "travel":
                self._travel(r, a)
            elif typ in ("shift_visit", "wait"):
                r.t = max(r.t, a["end"])
            elif typ == "rest":
                r.t += a["end"] - a["start"]
                r.energy = min(100.0, r.energy + 40)
            elif typ == "buy_ticket":
                r.t += 5
                r.money += a.get("money", 0.0)
            elif typ in VISITS:
                poi = self.city.poi_by_node[a["poi"]]
                if r.t < poi["open_min"]:
                    r.t = float(poi["open_min"])
                s = self.city.slot_of(r.t)
                i = self.city.poi_index[a["poi"]]
                if not self.closed(a["poi"], r.t):
                    self.routed[i][s] += self.weight
                r.pending = (a, s)
                r.i += 1
                return
            r.i += 1
        if not r.finished and r.i >= len(r.actions) and r.pending is None:
            r.finished = True

    def resolve(self, r: TravellerRun, s: int) -> None:
        a, _ = r.pending  # type: ignore[misc]
        r.pending = None
        poi = self.city.poi_by_node[a["poi"]]
        i = self.city.poi_index[a["poi"]]
        if self.closed(a["poi"], r.t):
            r.failures.append({"poi": a["poi"], "reason": "closed"})
            return
        rho = self.load[i][s]
        if poi["ticketed"] and self.capfac(i) < 1.0 and rho > 1.6:
            r.failures.append({"poi": a["poi"], "reason": "timed entry sold out"})
            return
        wait = self.wait[i][s]
        dur = poi["duration_min"]
        end = r.t + wait + dur
        if end > poi["close_min"] + 1:
            r.failures.append({"poi": a["poi"], "reason": "closed before the visit could finish"})
            r.t += min(wait, 20)
            return
        if a["type"] != "visit" or not poi["ticketed"]:
            r.money += a.get("money", 0.0)
        r.visits.append({"poi": a["poi"], "kind": poi["kind"], "arrive": r.t, "wait": wait, "end": end, "rho": rho,
                         "utility": utility(r.profile, poi, self.city), "independent": poi["independent"]})
        if a["type"] == "eat":
            r.lunch = True
            r.hunger = 0
        r.energy -= 0.08 * dur
        r.t = end

    def _replan(self, r: TravellerRun, coord, day) -> None:
        trig = r.replan_request or {}
        r.replan_request = None
        remaining = r.actions[r.i:]
        if not any(a["type"] in VISITS for a in remaining):
            return
        prof = r.profile
        if r.end_min != prof.end_min:
            d = prof.to_dict()
            d["end_min"] = r.end_min
            prof = TravellerProfile.from_dict(d)
        state = {"node": r.node, "time": r.t, "budget": max(0.0, r.profile.budget - r.money), "energy": r.energy,
                 "hunger": r.hunger, "walked_m": r.walked, "lunch_done": r.lunch,
                 "visited": [v["poi"] for v in r.visits]}
        try:
            res = coord.replan(prof, state, remaining, trig)
        except Exception as exc:  # keep the day going; report it
            self.log.append("replan failed for %s: %s" % (r.profile.id, exc))
            return
        new = res.get("actions", [])
        day.add_plan(remaining, -1.0)
        day.add_plan(new, +1.0)
        r.actions = r.actions[:r.i] + new
        r.replans.append({"time": r.t, "trigger": trig.get("trigger", ""), "strategy": res.get("strategy"),
                          "changes": res.get("stability_changes", 0)})
        r.result.reasons.append(ReasonRecord("M2", "replan", detail={"time": r.t, "trigger": trig.get("trigger", ""),
                                                                     "strategy": res.get("strategy")}))

    # -------------------------------------------------------------- main loop
    def run(self, runs: List[TravellerRun], coord, day, replan: bool,
            updater: Optional[OnlineCrowdUpdater] = None) -> None:
        city = self.city
        timeline = list(self.sc["timeline"])
        surged: Set[int] = set()
        for s in range(city.n_slots):
            t0 = city.slot_start(s)
            t1 = t0 + city.slot_min
            changed = False
            while timeline and timeline[0]["at"] < t1:
                self._apply_event(timeline.pop(0), runs, replan)
                changed = True
            for poi, until in list(self.closures.items()):
                if t0 >= until:
                    del self.closures[poi]
            if changed and replan:
                day.update(conditions=self.cond)
            # live sensors -> Bayesian update -> unannounced crowds trigger replanning
            if updater is not None and s > 0:
                for i in self.sensors:
                    total = self.bg[i][s - 1] + self.routed[i][s - 1]
                    expected = day.forecast.load[i][s - 1] * city.pois[i]["capacity"] + day.planned[i][s - 1]
                    if expected > 0.5:
                        count = _binomial(self.rng, int(total), self.cfg["simulation"]["sensor_detection"])
                        updater.observe(i, s - 1, count, expected)
                new = [i for i in updater.surges(from_slot=s) if i not in surged]
                if new:
                    surged.update(new)
                    day.update(forecast=updater.updated_forecast(s))
                    names = ", ".join(city.pois[i]["name"] for i in new)
                    self.log.append("%s sensors detect an unusual crowd at %s" % (fmt_time(t0), names))
                    nodes = {city.pois[i]["node"] for i in new}
                    for r in runs:
                        if any(a["poi"] in nodes and a["start"] < t0 + 180 for a in r.remaining_visits()):
                            r.replan_request = {"trigger": "an unusual crowd is forming at " + names, "time": t0}
            for r in runs:
                self.advance(r, t1, coord, day, replan)
            for i, p in enumerate(city.pois):
                cap = p["capacity"] * self.capfac(i)
                a = 0.0 if self.closed(p["node"], t0) else self.bg[i][s] + self.routed[i][s]
                self.load[i][s] = a / cap
                self.wait[i][s] = city.slot_min * max(0.0, self.queue[i] + 0.5 * a - 0.5 * cap) / cap
                self.queue[i] = max(0.0, self.queue[i] + a - cap)
            for _ in range(3):
                pending = [r for r in runs if r.pending is not None and r.pending[1] <= s]
                if not pending:
                    break
                for r in pending:
                    self.resolve(r, s)
                    self.advance(r, t1, coord, day, replan)
        for r in runs:
            if not r.finished:
                if r.pending is not None:
                    r.failures.append({"poi": r.pending[0]["poi"], "reason": "day ended"})
                    r.pending = None
                self._go_home(r)

    # -------------------------------------------------------------- metrics
    def metrics(self, runs: List[TravellerRun]) -> Dict[str, Any]:
        city = self.city
        n = max(1, len(runs))
        attractions = [i for i, p in enumerate(city.pois) if p["kind"] == "attraction"]
        per = []
        for r in runs:
            top = sorted((p for p in city.pois if p["kind"] != "restaurant"),
                         key=lambda p: -utility(r.profile, p, city))[:5]
            top_nodes = {p["node"] for p in top}
            acts = [v for v in r.visits if v["kind"] != "restaurant"]
            wait = sum(v["wait"] for v in r.visits)
            sat = satisfaction(r.visits, r.travel_min)
            late = r.t > r.end_min + 5
            feasible = (r.result.status == "ok" and not r.failures and not late
                        and r.money <= r.profile.budget + 1 and r.walked <= r.profile.max_walk_m + 1)
            total_d = sum(r.dist.values()) or 1.0
            per.append({
                "id": r.profile.id, "feasible": feasible, "satisfaction": sat, "activities": len(acts),
                "preferred": sum(1 for v in acts if v["poi"] in top_nodes), "wait": wait, "co2": r.co2,
                "travel_min": r.travel_min, "green_share": (r.dist["walk"] + r.dist["transit"]) / total_d,
                "car_km": r.dist["car"] / 1000.0, "local": sum(1 for v in r.visits if v["independent"]),
                "failures": len(r.failures), "late": late, "dropped": r.dropped, "exposure": [v["rho"] for v in r.visits],
                "proposals": r.result.proposals, "accepted": r.result.accepted, "replans": len(r.replans),
                "replan_changes": sum(x["changes"] for x in r.replans), "plan_ms": r.result.timings.get("total_ms", 0.0),
            })
        exposure = [x for p in per for x in p["exposure"]]
        # crowd distribution: how unevenly the day's load (relative to capacity) is spread over attractions
        core = [s for s in range(city.n_slots) if 600 <= city.slot_start(s) < 1020]
        mean_load = sorted(sum(self.load[i][s] for s in core) / len(core) for i in attractions)
        gini = gini_coefficient(mean_load)
        routed_visits = [0.0] * len(city.pois)
        for r in runs:
            for v in r.visits:
                routed_visits[city.poi_index[v["poi"]]] += 1
        top5 = sorted(attractions, key=lambda i: -city.pois[i]["popularity"])[:5]
        top5_share = sum(routed_visits[i] for i in top5) / max(1.0, sum(routed_visits[i] for i in attractions))
        open_cells = [(i, s) for i in attractions for s in range(city.n_slots)
                      if city.pois[i]["open_min"] <= city.slot_start(s) < city.pois[i]["close_min"]]
        overloaded = sum(1 for i, s in open_cells if self.load[i][s] > 1.0)
        props = sum(p["proposals"] for p in per)
        replans = sum(p["replans"] for p in per)

        def mean(key: str) -> float:
            return sum(float(p[key]) for p in per) / n

        return {
            "travellers": len(runs),
            "feasibility": mean("feasible"),
            "satisfaction": mean("satisfaction"),
            "activities": mean("activities"),
            "preferred_activities": mean("preferred"),
            "avg_wait_min": mean("wait"),
            "co2_kg": mean("co2"),
            "green_share": mean("green_share"),
            "car_km": mean("car_km"),
            "travel_min": mean("travel_min"),
            "local_visits": mean("local"),
            "failures": mean("failures"),
            "dropped_visits": mean("dropped"),
            "crowd_exposure": sum(exposure) / max(1, len(exposure)),
            "overloaded_visit_share": sum(1 for x in exposure if x > 1.0) / max(1, len(exposure)),
            "peak_load": max((self.load[i][s] for i, s in open_cells), default=0.0),
            "overloaded_slot_share": overloaded / max(1, len(open_cells)),
            "gini_load": gini,
            "top5_share": top5_share,
            "proposals": props / n,
            "accepted_alternatives": sum(p["accepted"] for p in per) / n,
            "acceptance_rate": sum(p["accepted"] for p in per) / max(1, props),
            "replans": replans / n,
            "avg_replan_changes": sum(p["replan_changes"] for p in per) / max(1, replans),
            "plan_ms": mean("plan_ms"),
        }
