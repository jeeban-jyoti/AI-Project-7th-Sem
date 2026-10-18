"""What the RL agent knows about one itinerary stop, the discrete state, the
actions, and the reward. Shared by the training environment and the live
pipeline so the learned policy sees the same inputs in both."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from tourism.core.city import City
from tourism.core.types import Proposal, TravellerProfile
from tourism.m1_search.candidates import similar_alternatives, utility
from tourism.m4_rl.matrix import TravelMatrix

ACTIONS = ["keep", "shift", "substitute", "switch_mode", "add_local", "ask"]
KEEP, SHIFT, SUBSTITUTE, SWITCH_MODE, ADD_LOCAL, ASK = range(6)
WILL_LEVELS = ["unknown", "low", "medium", "high"]
STATE_DIMS = (4, 4, 3, 4, 2, 2, 2)  # time block, crowd, alternative quality, willingness, car leg, local option, slack
N_STATES = 1
for _d in STATE_DIMS:
    N_STATES *= _d
FEATURE_DIM = sum(STATE_DIMS) + 1


@dataclass
class StopOptions:
    poi: int
    start: float
    duration: float
    utility: float
    rho: float
    wait: float
    popularity: float
    leg_car: bool = False
    leg_time: float = 0.0
    leg_co2: float = 0.0
    leg_money: float = 0.0
    green_time: float = 0.0
    green_co2: float = 0.0
    green_money: float = 0.0
    shift_start: Optional[int] = None
    shift_rho: float = 0.0
    shift_wait: float = 0.0
    shift_extra: float = 0.0
    alt: Optional[int] = None
    alt_utility: float = 0.0
    alt_rho: float = 0.0
    alt_wait: float = 0.0
    alt_extra: float = 0.0
    alt_match: float = 0.0
    alt_rating_gap: float = 0.0
    alt_pop_drop: float = 0.0
    alt_popularity: float = 0.0
    alt_co2: float = 0.0
    local: Optional[int] = None
    local_utility: float = 0.0
    local_extra: float = 0.0
    local_match: float = 0.0
    slack: float = 0.0


LoadFn = Callable[[int, float], Tuple[float, float]]  # (poi node, time) -> (rho, wait)


def build_stop_options(city: City, profile: TravellerProfile, poi_node: int, start: float, prev_node: int,
                       prev_end: float, next_node: int, latest_end: float, load_fn: LoadFn, matrix: TravelMatrix,
                       used: Iterable[int], slack: float) -> StopOptions:
    poi = city.poi_by_node[poi_node]
    used = set(used)
    green_only = "car" not in profile.modes
    dur = float(poi["duration_min"])
    rho, wait = load_fn(poi_node, start)
    o = StopOptions(poi=poi_node, start=start, duration=dur, utility=utility(profile, poi, city), rho=rho, wait=wait,
                    popularity=poi["popularity"], slack=slack)
    o.leg_time = matrix.time(prev_node, poi_node, prev_end, green_only)
    o.leg_co2 = matrix.co2(prev_node, poi_node, prev_end, green_only)
    o.leg_car = matrix.uses_car(prev_node, poi_node, prev_end, green_only)
    o.green_time = matrix.time(prev_node, poi_node, prev_end, True)
    o.green_co2 = matrix.co2(prev_node, poi_node, prev_end, True)
    o.leg_money = matrix.money(prev_node, poi_node, prev_end, green_only)
    o.green_money = matrix.money(prev_node, poi_node, prev_end, True)

    # shift: a quieter start within +-2.5 h that still fits opening hours and the next stop
    earliest = prev_end + o.leg_time
    best = None
    s0 = city.slot_start(city.slot_of(earliest)) + (city.slot_min if earliest > city.slot_start(city.slot_of(earliest)) else 0)
    for t in range(int(s0), int(start + 150) + 1, city.slot_min):
        if abs(t - start) < city.slot_min or t < poi["open_min"] or t + dur > poi["close_min"] or t + dur > latest_end:
            continue
        r2, w2 = load_fn(poi_node, t)
        if r2 <= rho - 0.15 and (best is None or r2 + 0.01 * abs(t - start) / 15 < best[1] + 0.01 * abs(best[0] - start) / 15):
            best = (t, r2, w2)
    if best:
        o.shift_start, o.shift_rho, o.shift_wait = best[0], best[1], best[2]
        o.shift_extra = 0.35 * abs(best[0] - start)

    # substitute: a similar, less-crowded POI
    base_cost = o.leg_time + matrix.time(poi_node, next_node, start + dur, green_only)
    best_alt = None
    for q in similar_alternatives(profile, poi, city, k=5, exclude=used):
        r2, w2 = load_fn(q["node"], start)
        if q["open_min"] > start or start + q["duration_min"] > q["close_min"]:
            continue
        extra = matrix.time(prev_node, q["node"], prev_end, green_only) + \
            matrix.time(q["node"], next_node, start + q["duration_min"], green_only) - base_cost
        uq = utility(profile, q, city)
        score = uq - 3.0 * max(0.0, r2 - 0.8) - 0.03 * max(0.0, extra) - 0.05 * w2
        if r2 < rho - 0.1 and (best_alt is None or score > best_alt[0]):
            best_alt = (score, q, uq, r2, w2, extra)
    if best_alt:
        _, q, uq, r2, w2, extra = best_alt
        o.alt, o.alt_utility, o.alt_rho, o.alt_wait, o.alt_extra = q["node"], uq, r2, w2, extra
        o.alt_match = min(1.2, uq / max(0.1, o.utility))
        o.alt_rating_gap = poi["rating"] - q["rating"]
        o.alt_pop_drop = poi["popularity"] - q["popularity"]
        o.alt_popularity = q["popularity"]
        o.alt_co2 = matrix.co2(prev_node, q["node"], prev_end, green_only)

    # a local business near this stop
    best_local = None
    for q in city.nearby(poi_node, 1200, ["shop", "experience"]):
        if not q["independent"] or q["node"] in used or (profile.accessible_only and not q["accessible"]):
            continue
        uq = utility(profile, q, city)
        extra = q["duration_min"] + 2 * city.dist(poi_node, q["node"]) / 75.0
        if extra <= slack and (best_local is None or uq > best_local[1]):
            best_local = (q, uq, extra)
    if best_local:
        q, uq, extra = best_local
        o.local, o.local_utility, o.local_extra = q["node"], uq, extra
        o.local_match = min(1.0, uq / 8.0)
    return o


def crowd_bucket(rho: float) -> int:
    return 0 if rho < 0.6 else (1 if rho < 0.85 else (2 if rho < 1.1 else 3))


def state_tuple(o: StopOptions, willingness: str) -> Tuple[int, ...]:
    tb = 0 if o.start < 660 else (1 if o.start < 840 else (2 if o.start < 1020 else 3))
    alt_q = 0
    if o.alt is not None:
        alt_q = 2 if o.alt_match >= 0.9 else (1 if o.alt_match >= 0.75 else 0)
    return (tb, crowd_bucket(o.rho), alt_q, WILL_LEVELS.index(willingness), int(o.leg_car),
            int(o.local is not None), int(o.slack >= 60))


def state_index(st: Tuple[int, ...]) -> int:
    idx = 0
    for v, d in zip(st, STATE_DIMS):
        idx = idx * d + v
    return idx


def feature_vector(st: Tuple[int, ...]) -> List[float]:
    x = [1.0]
    for v, d in zip(st, STATE_DIMS):
        one = [0.0] * d
        one[v] = 1.0
        x.extend(one)
    return x


def valid_actions(o: StopOptions, asked: bool, must: bool = False) -> List[int]:
    """Must-visits can be moved in time but never replaced."""
    acts = [KEEP]
    if o.shift_start is not None:
        acts.append(SHIFT)
    if o.alt is not None and not must:
        acts.append(SUBSTITUTE)
    if o.leg_car:
        acts.append(SWITCH_MODE)
    if o.local is not None:
        acts.append(ADD_LOCAL)
    if not asked:
        acts.append(ASK)
    return acts


def make_proposal(action: int, o: StopOptions, city: City, explained: bool = True) -> Optional[Proposal]:
    name = city.name_of(o.poi)
    if action == SHIFT:
        return Proposal("shift", o.poi, name, new_start=o.shift_start, extra_minutes=o.shift_extra,
                        wait_saved=o.wait - o.shift_wait, crowd_relief=o.rho - o.shift_rho, explained=explained)
    if action == SUBSTITUTE:
        return Proposal("substitute", o.poi, name, alt_poi=o.alt, alt_name=city.name_of(o.alt), interest_match=o.alt_match,
                        rating_gap=o.alt_rating_gap, popularity_drop=o.alt_pop_drop, extra_minutes=o.alt_extra,
                        wait_saved=o.wait - o.alt_wait, crowd_relief=o.rho - o.alt_rho, explained=explained)
    if action == SWITCH_MODE:
        return Proposal("switch_mode", o.poi, name, extra_minutes=o.green_time - o.leg_time,
                        co2_saved=o.leg_co2 - o.green_co2, explained=explained)
    if action == ADD_LOCAL:
        return Proposal("add_local", o.poi, name, alt_poi=o.local, alt_name=city.name_of(o.local),
                        interest_match=o.local_match, extra_minutes=o.local_extra, explained=explained)
    return None


def outcome(action: int, o: StopOptions) -> Dict[str, Any]:
    """Approximate effect of an (accepted) action on this stop."""
    out = {"poi": o.poi, "start": o.start, "utility": o.utility, "rho": o.rho, "wait": o.wait, "co2": o.leg_co2,
           "money": o.leg_money, "extra": 0.0, "less_visited": o.popularity < 0.5, "local": False, "local_poi": None}
    if action == SHIFT and o.shift_start is not None:
        out.update(start=o.shift_start, rho=o.shift_rho, wait=o.shift_wait, extra=o.shift_extra)
    elif action == SUBSTITUTE and o.alt is not None:
        out.update(poi=o.alt, utility=o.alt_utility, rho=o.alt_rho, wait=o.alt_wait, extra=max(0.0, o.alt_extra),
                   co2=o.alt_co2, less_visited=o.alt_popularity < 0.5)
    elif action == SWITCH_MODE and o.leg_car:
        out.update(co2=o.green_co2, money=o.green_money, extra=max(0.0, o.green_time - o.leg_time))
    elif action == ADD_LOCAL and o.local is not None:
        out.update(local=True, local_poi=o.local, extra=o.local_extra, utility=o.utility + 0.6 * o.local_utility)
    return out


def reward(w: Dict[str, float], out: Dict[str, Any], asked: bool = False, rejected: bool = False) -> float:
    r = w["satisfaction"] * (out["utility"] - 0.02 * out["extra"])
    r += w["completed"]
    r -= w["wait"] * out["wait"]
    r -= w["crowd"] * max(0.0, out["rho"] - 0.8)
    r -= w["externality"] * max(0.0, out["rho"] - 0.7)
    r -= w["co2"] * out["co2"]
    r -= w["cost"] * out.get("money", 0.0)
    r += w["less_visited"] * (1.0 if out["less_visited"] else 0.0)
    r += w["local"] * (1.0 if out["local"] else 0.0)
    if asked:
        r -= w["ask"]
    if rejected:
        r -= w["rejected"]
    return r
