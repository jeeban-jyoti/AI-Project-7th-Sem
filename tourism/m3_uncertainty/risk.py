"""Monte Carlo risk of a plan and uncertainty of its emissions.

Each sample draws a crowd multiplier per POI (from the online Gamma posterior,
or the prior when nothing has been observed) and transit / road delays (from
the delay network), then replays the plan. A plan "fails" in a sample when an
activity hits capacity, misses its closing time, or the traveller cannot get
back before the end of the day. When P(fail) is above the configured
threshold the coordinator keeps a backup itinerary.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from tourism.core.city import City
from tourism.m3_uncertainty.forecaster import CrowdForecaster, DayForecast
from tourism.m3_uncertainty.online import OnlineCrowdUpdater
from tourism.sim.truth import DayContext

VISIT_TYPES = ("visit", "eat", "local_experience")
CAR_OCCUPANCY = [(1, 0.5), (2, 0.35), (3, 0.15)]


def plan_risk(city: City, plan: Dict[str, Any], forecast: DayForecast, ctx: DayContext, end_min: int,
              rng: random.Random, samples: int = 200, crowd_threshold: float = 0.85, hard_capacity: float = 1.6,
              forecaster: Optional[CrowdForecaster] = None, updater: Optional[OnlineCrowdUpdater] = None,
              car_co2_g_pkm: float = 160.0) -> Dict[str, Any]:
    actions = plan.get("actions", [])
    visits = [a for a in actions if a["type"] in VISIT_TYPES]
    per = {a["poi"]: {"capacity": 0, "crowded": 0, "late": 0} for a in visits}
    fails = crowded_any = 0
    late_return = 0
    co2_samples: List[float] = []
    zones_with_event = {ev["zone"] for ev in ctx.events}
    for _ in range(samples):
        mult: Dict[int, float] = {}
        lateness = 0.0
        failed = False
        crowded = False
        co2 = 0.0
        for a in actions:
            if a["type"] == "travel":
                for seg in a.get("segments", []):
                    if seg["mode"] == "car":
                        occ = _pick(rng, CAR_OCCUPANCY)
                        co2 += seg["dist_m"] / 1000.0 * car_co2_g_pkm / 1000.0 * (1.0 if occ == 1 else 1.0 / occ * 1.4)
                    elif seg["mode"] != "walk":
                        co2 += seg.get("co2_kg", 0.0) * rng.uniform(0.8, 1.2)
                    if seg["mode"] in ("metro", "tram", "bus") and forecaster is not None:
                        line = city.line_by_id.get(seg.get("line", ""), None)
                        on_line = bool(line and zones_with_event & {city.nodes[n]["zone"] for n in line["stops"]})
                        peak = any(w[0] <= a["start"] < w[1] for w in city.params["peak_windows"])
                        probs = forecaster.delay_probs(ctx.weather, on_line, peak)
                        lvl = _pick(rng, list(enumerate(probs)))
                        lateness += forecaster.delay_mean[lvl] * rng.uniform(0.5, 1.5)
                    elif seg["mode"] == "car":
                        lateness += max(0.0, rng.gauss(0, 0.12)) * (a["end"] - a["start"])
                continue
            if a["type"] not in VISIT_TYPES:
                continue
            i = city.poi_index[a["poi"]]
            if i not in mult:
                alpha = updater.alpha[i] if updater else 4.0
                beta = updater.beta[i] if updater else 4.0
                mult[i] = rng.gammavariate(alpha, 1.0 / beta)
            m = mult[i]
            s = city.slot_of(a["start"] + lateness)
            rho = forecast.load[i][s] * m
            poi = city.pois[i]
            extra_wait = forecast.wait[i][s] * max(0.0, m * m - 1.0)
            lateness += extra_wait
            rec = per[a["poi"]]
            if rho > hard_capacity:
                rec["capacity"] += 1
                failed = True
            elif a["end"] + lateness > poi["close_min"]:
                rec["late"] += 1
                failed = True
            if rho > crowd_threshold + 0.25:
                rec["crowded"] += 1
                crowded = True
        if plan.get("totals", {}).get("end_time", 0) + lateness > end_min:
            late_return += 1
            failed = True
        fails += failed
        crowded_any += crowded
        co2_samples.append(co2)
    n = float(max(1, samples))
    by_activity = []
    for poi, rec in per.items():
        by_activity.append({"poi": poi, "name": city.name_of(poi), "p_capacity": rec["capacity"] / n,
                            "p_crowded": rec["crowded"] / n, "p_late": rec["late"] / n})
    riskiest = max(by_activity, key=lambda r: r["p_capacity"] + r["p_late"] + 0.5 * r["p_crowded"], default=None)
    co2_samples.sort()
    return {
        "p_fail": fails / n,
        "p_crowded": crowded_any / n,
        "p_late_return": late_return / n,
        "by_activity": by_activity,
        "riskiest": riskiest,
        "co2_kg": {"mean": sum(co2_samples) / n,
                   "p5": co2_samples[int(0.05 * (len(co2_samples) - 1))] if co2_samples else 0.0,
                   "p95": co2_samples[int(0.95 * (len(co2_samples) - 1))] if co2_samples else 0.0},
    }


def _pick(rng: random.Random, options):
    u = rng.random()
    acc = 0.0
    for value, p in options:
        acc += p
        if u <= acc:
            return value
    return options[-1][0]
