"""RL environment: one episode is one simulated day.

Travellers arrive one after another, each with a conventional (satisfaction-
first) itinerary. For every stop the agent picks an action; the simulated
traveller accepts or rejects it; the stop's final POI and time are added to
the city's routed load, so later travellers in the same day face the crowds
earlier decisions created (the collective / social-welfare effect).
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from tourism.core.city import City
from tourism.data.personas import generate_travellers
from tourism.m1_search.candidates import utility
from tourism.m3_uncertainty.forecaster import DayForecast
from tourism.m4_rl.features import (ACTIONS, ASK, KEEP, SHIFT, SUBSTITUTE, SWITCH_MODE, StopOptions,
                                    build_stop_options, make_proposal, outcome, reward, state_index, state_tuple,
                                    valid_actions)
from tourism.m4_rl.matrix import TravelMatrix
from tourism.sim.traveller import SimulatedTraveller


def greedy_itinerary(city: City, profile, matrix: TravelMatrix) -> List[Dict[str, Any]]:
    """Conventional plan: the top-utility attractions, nearest-neighbour order."""
    green = "car" not in profile.modes
    pool = [p for p in city.pois if p["kind"] != "restaurant" and (p["accessible"] or not profile.accessible_only)]
    pool.sort(key=lambda p: -utility(profile, p, city))
    chosen = pool[: profile.max_activities]
    for m in profile.must_visit:
        if m not in [p["node"] for p in chosen]:
            chosen = [city.poi_by_node[m]] + chosen[:-1]
    order, cur, t = [], profile.start_node, float(profile.start_min)
    left = list(chosen)
    while left:
        nxt = min(left, key=lambda p: matrix.time(cur, p["node"], t, green))
        left.remove(nxt)
        arrive = t + matrix.time(cur, nxt["node"], t, green)
        start = max(arrive, nxt["open_min"])
        start = city.slot_start(city.slot_of(start)) + (city.slot_min if start > city.slot_start(city.slot_of(start)) else 0)
        if start + nxt["duration_min"] > min(nxt["close_min"], profile.end_min - 30):
            continue
        order.append({"poi": nxt["node"], "start": float(start)})
        cur, t = nxt["node"], start + nxt["duration_min"]
    return order


class ItineraryEnv:
    def __init__(self, city: City, matrix: TravelMatrix, forecasts: List[DayForecast], rl_cfg: Dict[str, Any],
                 travellers_per_day: int = 40, traveller_weight: float = 8.0, explanation_boost: float = 0.35,
                 willingness: str = "mixed"):
        self.city = city
        self.matrix = matrix
        self.forecasts = forecasts
        self.w = rl_cfg["reward"]
        self.n_travellers = travellers_per_day
        self.weight = traveller_weight
        self.boost = explanation_boost
        self.willingness = willingness
        self.rng = random.Random(0)

    # load model: background forecast + routed travellers of this day
    def load(self, poi: int, t: float) -> Tuple[float, float]:
        i = self.city.poi_index[poi]
        s = self.city.slot_of(t)
        base = self.fc.load[i][s]
        rho = base + self.routed[i][s] / self.city.pois[i]["capacity"]
        wait = self.fc.wait[i][s] + self.city.slot_min * 2.0 * max(0.0, rho - max(base, 0.8))
        return rho, wait

    def reset(self, seed: int) -> Tuple[int, List[int]]:
        self.rng = random.Random(seed)
        self.fc = self.rng.choice(self.forecasts)
        self.routed = [[0.0] * self.city.n_slots for _ in self.city.pois]
        self.travellers = generate_travellers(self.city, self.n_travellers, seed * 7919 + 13, self.willingness)
        self.t_idx = -1
        self.stats = {"decisions": 0, "proposals": 0, "accepted": 0, "asks": 0, "utility": 0.0, "wait": 0.0,
                      "rho": 0.0, "overload": 0, "co2": 0.0, "local": 0, "reward": 0.0, "substitutes": 0,
                      "shifts": 0, "mode_switches": 0}
        self._next_traveller()
        return self._observe()

    def _next_traveller(self) -> None:
        self.t_idx += 1
        if self.t_idx >= len(self.travellers):
            return
        self.profile, traits = self.travellers[self.t_idx]
        self.sim = SimulatedTraveller(self.profile, traits, self.rng, self.boost)
        self.stops = greedy_itinerary(self.city, self.profile, self.matrix)
        while not self.stops and self.t_idx + 1 < len(self.travellers):
            self.t_idx += 1
            self.profile, traits = self.travellers[self.t_idx]
            self.sim = SimulatedTraveller(self.profile, traits, self.rng, self.boost)
            self.stops = greedy_itinerary(self.city, self.profile, self.matrix)
        self.k = 0
        self.will = "unknown"
        self.asked = False
        self.prev_node = self.profile.start_node
        self.prev_end = float(self.profile.start_min)
        self.used = {s["poi"] for s in self.stops}

    def done(self) -> bool:
        return self.t_idx >= len(self.travellers) or not self.stops

    def _observe(self) -> Tuple[int, List[int]]:
        if self.done():
            return 0, [KEEP]
        stop = self.stops[self.k]
        nxt = self.stops[self.k + 1] if self.k + 1 < len(self.stops) else None
        next_node = nxt["poi"] if nxt else self.profile.end_node
        latest_end = (nxt["start"] - 10) if nxt else self.profile.end_min - 30
        last = self.stops[-1]
        last_end = last["start"] + self.city.poi_by_node[last["poi"]]["duration_min"]
        slack = self.profile.end_min - last_end - 30
        start = max(stop["start"], self.prev_end)
        self.opts: StopOptions = build_stop_options(self.city, self.profile, stop["poi"], start, self.prev_node,
                                                    self.prev_end, next_node, latest_end, self.load, self.matrix,
                                                    self.used, slack)
        self.state = state_tuple(self.opts, self.will)
        return state_index(self.state), valid_actions(self.opts, self.asked, stop["poi"] in self.profile.must_visit)

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        o = self.opts
        st = self.stats
        info: Dict[str, Any] = {"traveller_done": False, "action": ACTIONS[action]}
        if action == ASK:
            self.will = self.sim.reveal_willingness()
            self.asked = True
            st["asks"] += 1
            r = -self.w["ask"]
            st["reward"] += r
            s2, valid = self._observe()
            info["valid"] = valid
            return s2, r, False, info
        proposal = make_proposal(action, o, self.city) if action != KEEP else None
        accepted = True
        if proposal is not None:
            st["proposals"] += 1
            accepted = self.sim.respond(proposal)
            st["accepted"] += int(accepted)
        final_action = action if accepted else KEEP
        baseline = reward(self.w, outcome(KEEP, o))
        if accepted:
            st["substitutes"] += int(action == SUBSTITUTE)
            st["shifts"] += int(action == SHIFT)
            st["mode_switches"] += int(action == SWITCH_MODE)
        out = outcome(final_action, o)
        # reward = improvement over keeping the stop as planned (a per-stop baseline that
        # removes the large variance due to how much the traveller likes this stop)
        r = reward(self.w, out, rejected=not accepted) - baseline
        # commit the stop: it now adds to the city's crowd for everyone after
        i = self.city.poi_index[out["poi"]]
        s = self.city.slot_of(out["start"])
        rho_now, wait_now = self.load(out["poi"], out["start"])
        self.routed[i][s] += self.weight
        if out["local"] and out["local_poi"] is not None:
            j = self.city.poi_index[out["local_poi"]]
            self.routed[j][self.city.slot_of(out["start"] + o.duration)] += self.weight
            self.used.add(out["local_poi"])
            st["local"] += 1
        if out["poi"] != o.poi:
            self.used.add(out["poi"])
        st["decisions"] += 1
        st["utility"] += out["utility"]
        st["wait"] += wait_now
        st["rho"] += rho_now
        st["overload"] += int(rho_now > 1.0)
        st["co2"] += out["co2"]
        st["reward"] += r
        self.prev_node = out["poi"]
        self.prev_end = out["start"] + wait_now + self.city.poi_by_node[out["poi"]]["duration_min"] + out["extra"] * 0.3
        self.k += 1
        if self.k >= len(self.stops):
            info["traveller_done"] = True
            self._next_traveller()
        info["accepted"] = accepted
        s2, valid = self._observe()
        info["valid"] = valid
        return s2, r, self.done(), info

    def features(self) -> Tuple[int, ...]:
        return self.state
