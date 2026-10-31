"""
MODULE 1 - SEARCH and CONSTRAINT SATISFACTION (CSP)

Part A - Search: find the best route between two places.
    The city is a graph. Places are nodes; footpaths, bus links and taxi
    rides are edges. Each edge has a travel time, CO2, fare and distance.
    Algorithms: Uniform Cost Search (UCS), Greedy Best-First Search and A*.
    Wheelchair users never use steep footpaths or pass through places
    without step-free access; a suspended bus line is simply left out.

Part B - CSP: build a day itinerary.
    Variables : the stops of the day (stop 1, stop 2, ...), in time order
    Domains   : the candidate places (sights, local businesses, restaurants)
    Hard constraints:
        - each place at most once (all-different)
        - the place is open for the whole visit
        - the traveller is back at the hotel before the end of the day
        - total spending stays within the budget
        - lunch (if wanted) starts between 11:30 and 14:30
        - wheelchair users only get accessible places
        - every must-visit place is included
    Soft constraints (the score to maximise): interest match, crowds,
        CO2, travel time, support for local businesses.
    Solver: backtracking + forward checking + branch and bound.

This file also holds small helpers (loading the city, time formatting)
that the other modules reuse.
"""
import heapq
import json
import math

INF = float("inf")


# ----------------------------------------------------------------- helpers
def to_min(text):
    """'09:30' -> 570 (minutes after midnight)."""
    h, m = text.split(":")
    return int(h) * 60 + int(m)


def hhmm(minutes):
    """570 -> '09:30'."""
    minutes = int(round(minutes))
    return "%02d:%02d" % (minutes // 60, minutes % 60)


def load_city(path="data/city.json"):
    """Read the city made by generate_data.py and add opening times in minutes."""
    with open(path, encoding="utf-8") as f:
        city = json.load(f)
    city["by_name"] = {}
    for p in city["places"]:
        if "open" in p:
            p["open_min"] = to_min(p["open"])
            p["close_min"] = to_min(p["close"])
        city["by_name"][p["name"]] = p
    return city


def distance_km(a, b):
    """Straight-line distance between two places."""
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


# ======================================================= PART A: SEARCH
def build_graph(city, modes=("walk", "bus", "taxi"), accessible_only=False, suspended_lines=()):
    """Adjacency list: graph[place name] = list of edges leaving that place."""
    t = city["transport"]
    graph = {p["name"]: [] for p in city["places"]}

    def usable(p):                                        # wheelchair users avoid places without access
        return p.get("accessible", True) or not accessible_only

    def add_edge(a, b, mode, line=None):
        km = distance_km(a, b) * t["street_detour"]       # streets are not straight lines
        m = t[mode]
        graph[a["name"]].append({
            "from": a["name"], "to": b["name"], "mode": mode, "line": line, "km": km,
            "minutes": km / m["speed_kmh"] * 60 + m["stop_wait_min"],
            "co2": km * m["co2_g_per_km"],
            "fare": km * m["fare_per_km"] + m["fare_per_ride"],
        })

    if "walk" in modes:
        for name_a, name_b, steep in city["walk_links"]:
            a, b = city["by_name"][name_a], city["by_name"][name_b]
            if usable(a) and usable(b) and not (steep and accessible_only):
                add_edge(a, b, "walk")
                add_edge(b, a, "walk")
    if "taxi" in modes:
        for a in city["places"]:
            for b in city["places"]:
                if a is not b and usable(a) and usable(b) and distance_km(a, b) <= t["taxi_link_km"]:
                    add_edge(a, b, "taxi")
    if "bus" in modes:
        for line, stops in city["bus_lines"].items():
            if line in suspended_lines:
                continue
            for s1, s2 in zip(stops, stops[1:]):
                a, b = city["by_name"][s1], city["by_name"][s2]
                add_edge(a, b, "bus", line)
                add_edge(b, a, "bus", line)
    return graph


def edge_cost(edge, co2_weight):
    """Cost of one edge = minutes + co2_weight * (grams of CO2 / 10).
    co2_weight = 0 -> fastest route; co2_weight = 1 -> greener route."""
    return edge["minutes"] + co2_weight * edge["co2"] / 10.0


def heuristic(city, node, goal):
    """Straight-line distance at the fastest possible speed, in minutes.
    It never over-estimates the real cost, so A* stays optimal (admissible)."""
    fastest_kmh = max(m["speed_kmh"] for m in city["transport"].values() if isinstance(m, dict))
    return distance_km(city["by_name"][node], city["by_name"][goal]) / fastest_kmh * 60


def search(city, graph, start, goal, method="astar", co2_weight=0.0):
    """method: 'ucs' (priority g), 'greedy' (priority h) or 'astar' (priority g + h).
    Returns a route dict, or None if the goal cannot be reached."""
    if start == goal:
        return {"legs": [], "minutes": 0, "co2": 0, "fare": 0, "walk_km": 0, "cost": 0, "expanded": 0}
    best_g = {start: 0.0}
    came_by = {start: None}               # the edge we used to reach each node
    counter = 0                           # tie-breaker so heapq never compares dicts
    frontier = [(0.0, counter, start)]
    expanded = 0
    while frontier:
        _, _, node = heapq.heappop(frontier)
        if node == goal:
            break
        expanded += 1
        for e in graph[node]:
            g2 = best_g[node] + edge_cost(e, co2_weight)
            if g2 < best_g.get(e["to"], INF):
                best_g[e["to"]] = g2
                came_by[e["to"]] = e
                h = heuristic(city, e["to"], goal)
                priority = {"ucs": g2, "greedy": h, "astar": g2 + h}[method]
                counter += 1
                heapq.heappush(frontier, (priority, counter, e["to"]))
    if goal not in came_by:
        return None
    legs = []
    node = goal
    while came_by[node] is not None:      # walk back from the goal to the start
        legs.append(came_by[node])
        node = came_by[node]["from"]
    legs.reverse()
    return {
        "legs": legs,
        "minutes": sum(e["minutes"] for e in legs),
        "co2": sum(e["co2"] for e in legs),
        "fare": sum(e["fare"] for e in legs),
        "walk_km": sum(e["km"] for e in legs if e["mode"] == "walk"),
        "cost": best_g[goal],
        "expanded": expanded,
    }


def route_modes(route):
    """'walk, bus, walk' style summary of a route."""
    modes = []
    for e in route["legs"]:
        if not modes or modes[-1] != e["mode"]:
            modes.append(e["mode"])
    return ", ".join(modes) if modes else "-"


def compare_search(city, pairs, co2_weight=0.0):
    """Run UCS, Greedy and A* on the same start/goal pairs."""
    graph = build_graph(city)
    rows = []
    for a, b in pairs:
        row = {"from": a, "to": b}
        for method in ("ucs", "greedy", "astar"):
            r = search(city, graph, a, b, method, co2_weight)
            row[method] = (r["cost"], r["expanded"])
        rows.append(row)
    return rows


# ========================================================== PART B: CSP
# How each planning system scores a place. "conventional" is the baseline
# that only cares about satisfaction and fame; "sustainable" is our system.
SYSTEMS = {
    "conventional": {"label": "Conventional (satisfaction only)", "popularity": 3.0, "crowd": 0.0,
                     "use_forecast": False, "co2_weight": 0.0, "local_bonus": 0.0, "quiet_bonus": 0.0,
                     "travel": 0.03, "buffer_min": 0},
    "sustainable": {"label": "Sustainable and crowd-aware", "popularity": 0.0, "crowd": 6.0,
                    "use_forecast": True, "co2_weight": 1.0, "local_bonus": 2.0, "quiet_bonus": 1.0,
                    "travel": 0.1, "buffer_min": 30},     # keep 30 min spare for queues and delays
}

LUNCH_FROM, LUNCH_TO = to_min("11:30"), to_min("14:30")
MAX_WAIT_FOR_OPENING = 60
TICKET_MIN = 5


def interest_utility(place, profile):
    """How much this traveller likes this place (0 to about 10)."""
    if place["type"] == "restaurant":
        return 4 * place["rating"] / 5 + 4 * profile["interests"].get("food", 0.3)
    match = sum(profile["interests"].get(c, 0) for c in place["categories"])
    return 10 * min(1.0, match) * place["rating"] / 5


def place_score(place, start, profile, system, forecast=None):
    """Soft-constraint score of visiting `place` at minute `start`."""
    s = interest_utility(place, profile) + system["popularity"] * place.get("popularity", 0)
    if place["type"] == "local" or place.get("local_owned"):
        s += system["local_bonus"]
    if place.get("popularity", 1) < 0.5:
        s += system["quiet_bonus"]
    if forecast is not None and system["use_forecast"]:
        p_high, wait = forecast(place["name"], start)
        s -= system["crowd"] * p_high + wait / 10.0
    return s


def why_impossible(place, profile):
    """Human-readable reason a must-visit place cannot be included."""
    if profile.get("accessible_only") and not place.get("accessible", True):
        return "%s is not wheelchair accessible" % place["name"]
    if place["price"] > profile["budget"]:
        return "%s costs more than the budget" % place["name"]
    if place["open_min"] >= to_min(profile["end"]) or place["close_min"] <= to_min(profile["start"]):
        return "%s is open %s-%s, outside your day" % (place["name"], place["open"], place["close"])
    return "%s does not fit in the day together with the other constraints" % place["name"]


def solve_itinerary(city, profile, system_name, forecast=None, earliest=None, banned=(), node_limit=40000):
    """Solve the itinerary CSP. If the must-visit places cannot all fit, the
    conflict is explained and must-visits are relaxed one at a time."""
    must = [m for m in profile.get("must_visit", []) if m not in banned]
    relaxed, reasons = [], []
    while True:
        result = _solve(city, profile, system_name, forecast, earliest or {}, banned, must, node_limit)
        if result["feasible"] or not must:
            break
        dropped = must.pop(0)
        relaxed.append(dropped)
        reasons.append(why_impossible(city["by_name"][dropped], profile))
    result["relaxed"], result["reasons"] = relaxed, reasons
    return result


def _solve(city, profile, system_name, forecast, earliest, banned, must, node_limit):
    system = SYSTEMS[system_name]
    graph = build_graph(city, profile["modes"], profile.get("accessible_only", False))
    hotel = profile["hotel"]
    day_start, day_end = to_min(profile["start"]), to_min(profile["end"]) - system["buffer_min"]
    use_fc = forecast is not None and system["use_forecast"]

    # 1. Node consistency: remove values that break a unary constraint.
    def allowed(p):
        return (p["name"] not in banned
                and (p.get("accessible", True) or not profile.get("accessible_only"))
                and p["price"] <= profile["budget"]
                and p["close_min"] > day_start and p["open_min"] < day_end)

    def base(p):                          # optimistic score (no crowd penalty, no travel)
        return place_score(p, None, profile, system)

    home = city["by_name"][hotel]

    def nearby_first(p):                  # candidates: good for this traveller, and not too far away
        return base(p) - 0.5 * distance_km(home, p)

    sights = sorted([p for p in city["places"] if p["type"] in ("attraction", "local") and allowed(p)],
                    key=nearby_first, reverse=True)
    sights = sights[:8] + [p for p in sights[8:] if p["name"] in must]
    restaurants = []
    if profile.get("lunch"):
        restaurants = sorted([p for p in city["places"] if p["type"] == "restaurant" and allowed(p)],
                             key=nearby_first, reverse=True)[:4]
    values = sights + restaurants
    if any(m not in {p["name"] for p in values} for m in must):     # a must-visit broke a unary constraint
        return {"feasible": False, "system": system_name, "stops": [], "nodes": 0}

    # 2. Travel table: A* route between every pair of candidate places.
    names = [hotel] + [p["name"] for p in values]
    table = {}
    for a in names:
        for b in names:
            if a != b:
                table[(a, b)] = search(city, graph, a, b, "astar", system["co2_weight"])

    def place_next(cur, t, money, p, limit_wait=True):
        """Check every hard constraint for putting p next. Returns a stop or None."""
        leg = table[(cur, p["name"])]
        back = table[(p["name"], hotel)]
        if leg is None or back is None:
            return None
        arrive = t + leg["minutes"]
        start = max(arrive, p["open_min"], earliest.get(p["name"], 0))
        if p["type"] == "restaurant":
            start = max(start, LUNCH_FROM)
            if start > LUNCH_TO:
                return None
        if limit_wait and start - arrive > MAX_WAIT_FOR_OPENING:
            return None
        wait = forecast(p["name"], start)[1] if use_fc else 0.0
        ticket = TICKET_MIN if p.get("ticketed") else 0
        end = start + ticket + wait + p["duration"]
        spend = p["price"] + leg["fare"]
        if end > p["close_min"] or end + back["minutes"] > day_end or money + spend > profile["budget"]:
            return None
        travel_pen = system["travel"] * (leg["minutes"] + system["co2_weight"] * leg["co2"] / 10)
        return {"name": p["name"], "type": p["type"], "leg": leg, "arrive": arrive, "start": start,
                "wait": wait, "end": end, "spend": spend, "utility": interest_utility(p, profile),
                "gain": place_score(p, start, profile, system, forecast) - travel_pen}

    best = {"score": -INF, "stops": None}
    nodes = [0]
    max_places = profile["max_places"]
    need_lunch = bool(restaurants)

    def backtrack(cur, t, money, stops, used, n_sights, lunch_done, score):
        nodes[0] += 1
        if nodes[0] > node_limit:
            return
        # Is the current partial plan a complete solution?
        if (lunch_done or not need_lunch) and n_sights >= 1 and all(m in used for m in must):
            back = table[(cur, hotel)]
            total = score - system["travel"] * (back["minutes"] + system["co2_weight"] * back["co2"] / 10)
            if total > best["score"]:
                best.update(score=total, stops=list(stops), back=back)

        # Domain of the next stop.
        domain = []
        for p in values:
            if p["name"] in used:
                continue
            if p["type"] == "restaurant" and lunch_done:
                continue
            if p["type"] != "restaurant" and n_sights >= max_places:
                continue
            stop = place_next(cur, t, money, p)
            if stop:
                domain.append(stop)

        # Forward checking: a missing must-visit (or lunch) that can no longer
        # fit at all, even with unlimited waiting, means this branch is dead.
        for m in must:
            if m not in used and place_next(cur, t, money, city["by_name"][m], limit_wait=False) is None:
                return
        if need_lunch and not lunch_done and not any(
                place_next(cur, t, money, r, limit_wait=False) for r in restaurants):
            return

        # Branch and bound: even the best possible continuation cannot win.
        left = sorted((base(p) for p in sights if p["name"] not in used), reverse=True)
        bound = score + sum(max(0.0, x) for x in left[:max_places - n_sights])
        if need_lunch and not lunch_done:
            bound += max(base(r) for r in restaurants)
        if bound <= best["score"]:
            return

        # Value ordering: try the most promising stops first.
        for stop in sorted(domain, key=lambda s: s["gain"], reverse=True):
            is_meal = stop["type"] == "restaurant"
            backtrack(stop["name"], stop["end"], money + stop["spend"], stops + [stop], used | {stop["name"]},
                      n_sights + (0 if is_meal else 1), lunch_done or is_meal, score + stop["gain"])

    backtrack(hotel, day_start, 0.0, [], frozenset(), 0, False, 0.0)

    if best["stops"] is None:
        return {"feasible": False, "system": system_name, "stops": [], "nodes": nodes[0]}
    stops, back = best["stops"], best["back"]
    legs = [s["leg"] for s in stops] + [back]
    return {
        "feasible": True, "system": system_name, "stops": stops, "return_leg": back,
        "back_at": stops[-1]["end"] + back["minutes"],
        "travel_min": sum(l["minutes"] for l in legs),
        "co2_g": sum(l["co2"] for l in legs),
        "spend": sum(s["spend"] for s in stops) + back["fare"],
        "utility": sum(s["utility"] for s in stops),
        "planned_wait": sum(s["wait"] for s in stops),
        "score": best["score"], "nodes": nodes[0],
    }


def print_itinerary(result, title):
    print("\n" + "=" * 72 + "\n " + title + "\n" + "=" * 72)
    if not result["feasible"]:
        print("  No feasible itinerary found.")
        return
    for s in result["stops"]:
        print("  %s-%s  %-28s (travel %2.0f min by %s)"
              % (hhmm(s["start"]), hhmm(s["end"]), s["name"], s["leg"]["minutes"], route_modes(s["leg"])))
    print("  back at hotel %s, travel %.0f min, CO2 %.0f g, spend %.0f"
          % (hhmm(result["back_at"]), result["travel_min"], result["co2_g"], result["spend"]))
    for r in result.get("reasons", []):
        print("  ! relaxed a must-visit: " + r)
