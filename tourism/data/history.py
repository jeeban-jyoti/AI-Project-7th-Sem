"""Generates historical observations (data/history.csv.gz, data/transit_history.csv)
from the hidden truth model. The Bayesian network in Module 3 is trained on these."""
from __future__ import annotations

import csv
import gzip
import os
import random
from typing import Any, Dict, List

from tourism.core.city import City
from tourism.sim.truth import TOD_EDGES, DayContext, TruthModel, queue_waits, tod_bin

HISTORY_FIELDS = ["day", "daykind", "season", "weather", "event_zone", "event_from", "event_to",
                  "poi", "slot", "arrivals", "rho", "wait"]
TRANSIT_FIELDS = ["day", "line", "tod", "weather", "event_on_line", "peak", "delay_min"]


def sample_context(rng: random.Random, day: int, city: City) -> DayContext:
    daykind = "weekday" if day % 7 < 5 else "weekend"
    if daykind == "weekday" and rng.random() < 0.06:
        daykind = "holiday"
    season = ["off", "shoulder", "peak"][(day // 10) % 3]
    rain_p = {"off": 0.30, "shoulder": 0.22, "peak": 0.15}[season]
    weather = "rain" if rng.random() < rain_p else "clear"
    events = []
    if rng.random() < 0.15:
        window = rng.choice([(600, 840), (960, 1260), (1020, 1260)])
        events.append({"type": "public_event", "zone": rng.randrange(len(city.zones)), "from": window[0],
                       "to": window[1], "factor": 1.6})
    return DayContext(daykind=daykind, season=season, weather=weather, events=events)


def generate_history(city: City, days: int, seed: int, out_dir: str) -> Dict[str, Any]:
    rng = random.Random(seed)
    truth = TruthModel(city)
    os.makedirs(out_dir, exist_ok=True)
    hist_path = os.path.join(out_dir, "history.csv.gz")
    transit_path = os.path.join(out_dir, "transit_history.csv")
    rows = 0
    with gzip.open(hist_path, "wt", newline="", encoding="utf-8") as hf, open(transit_path, "w", newline="", encoding="utf-8") as tf:
        hw = csv.writer(hf)
        tw = csv.writer(tf)
        hw.writerow(HISTORY_FIELDS)
        tw.writerow(TRANSIT_FIELDS)
        for d in range(days):
            ctx = sample_context(rng, d, city)
            ev = ctx.events[0] if ctx.events else None
            arrivals = truth.sample_day(ctx, rng)
            for i, p in enumerate(city.pois):
                waits = queue_waits(arrivals[i], p["capacity"], city.slot_min)
                for s in range(city.n_slots):
                    t = city.slot_start(s)
                    if not (p["open_min"] <= t < p["close_min"]):
                        continue
                    a = arrivals[i][s]
                    hw.writerow([d, ctx.daykind, ctx.season, ctx.weather,
                                 ev["zone"] if ev else -1, ev["from"] if ev else 0, ev["to"] if ev else 0,
                                 p["node"], s, a, round(a / p["capacity"], 3), round(waits[s], 1)])
                    rows += 1
            for line in city.lines:
                zones_on_line = {city.nodes[n]["zone"] for n in line["stops"]}
                for b in range(len(TOD_EDGES) - 1):
                    t_mid = (TOD_EDGES[b] + TOD_EDGES[b + 1]) / 2
                    on_line = bool(ev and ev["zone"] in zones_on_line and ev["from"] <= t_mid < ev["to"])
                    peak = any(w[0] <= t_mid < w[1] for w in city.params["peak_windows"])
                    delay = truth.sample_delay(rng, ctx.weather, on_line, peak)
                    tw.writerow([d, line["id"], tod_bin(t_mid), ctx.weather, int(on_line), int(peak), round(delay, 1)])
    return {"history": hist_path, "transit": transit_path, "rows": rows, "days": days}


def read_history(path: str) -> List[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
