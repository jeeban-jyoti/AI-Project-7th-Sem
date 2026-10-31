"""
MODULE 2 - PLANNING and REPLANNING

The CSP (module 1) decides WHICH places to visit. This module decides HOW:
it turns those goals into a sequence of timed actions, each with
preconditions and effects (classical STRIPS-style planning), and finds the
plan with A* search over world states.

World state: (location, time, energy, money, places visited, lunch eaten, tickets held)

Actions:
    travel(a, b)    pre: at a, enough money for the fare, enough energy to walk,
                         not holding an unused ticket for a
                    eff: at b, time/money/energy change, CO2 added
    buy_ticket(p)   pre: at p, p needs a ticket, no ticket yet, enough money
                    eff: ticket for p, money - price
    wait(p)         pre: at p, p not open yet          eff: time = opening time
    visit(p)        pre: at p, open for the whole visit, has ticket (if needed), energy >= 20,
                         money for the entry fee (places without tickets charge at the door)
                    eff: p visited, energy - 12 per hour, time + queue + duration, money - fee
    eat(r)          pre: at restaurant r, not eaten, 11:30 <= time <= 14:30
                    eff: eaten, energy + 25, time + 50 min
    rest            pre: energy < 40                   eff: energy + 30, time + 20 min

Goal: every goal place visited, lunch eaten (if wanted), back at the hotel
before the end of the day.

Replanning when something goes wrong during the day (a place closes, the bus
is late): 1) keep the plan if it is unaffected, 2) otherwise replan the rest
of the day with A*, 3) if that is impossible, drop the least valuable
remaining place and try again.

validate_plan() checks a finished plan again from scratch, without trusting
the planner, before it is shown to the traveller.
"""
import heapq

from module1_search_csp import (LUNCH_FROM, LUNCH_TO, SYSTEMS, build_graph, hhmm, interest_utility, route_modes,
                                search, to_min)

WALK_ENERGY_PER_KM = 6
VISIT_ENERGY_PER_HOUR = 12
CROWD_COST_MIN = 30          # the sustainable planner treats a surely-crowded visit like 30 extra minutes


def make_problem(city, profile, goals, lunch_place, system_name, forecast=None, earliest=None):
    """Everything the planner needs: goals, places, and a travel table built with A* (module 1)."""
    system = SYSTEMS[system_name]
    graph = build_graph(city, profile["modes"], profile.get("accessible_only", False))
    hotel = profile["hotel"]
    spots = [hotel] + list(goals) + ([lunch_place] if lunch_place else [])
    table = {}
    for a in spots:
        for b in spots:
            if a != b:
                table[(a, b)] = search(city, graph, a, b, "astar", system["co2_weight"])
    return {"city": city, "profile": profile, "goals": frozenset(goals), "lunch": lunch_place,
            "hotel": hotel, "table": table, "co2_weight": system["co2_weight"],
            "forecast": forecast if system["use_forecast"] else None,
            "crowd_cost": CROWD_COST_MIN * system["crowd"] / 6.0,
            "earliest": earliest or {}, "closed": set(),
            "end": to_min(profile["end"])}


def start_state(problem):
    pr = problem["profile"]
    #       location        time                 energy money           visited      ate    tickets
    return (problem["hotel"], to_min(pr["start"]), 100.0, pr["budget"], frozenset(), False, frozenset())


def is_goal(problem, s):
    loc, t, _, _, visited, ate, _ = s
    return (visited == problem["goals"] and (ate or not problem["lunch"])
            and loc == problem["hotel"] and t <= problem["end"])


def successors(problem, s):
    """All applicable actions in state s. Yields (new_state, steps, cost)."""
    city = problem["city"]
    loc, t, energy, money, visited, ate, tickets = s
    place = city["by_name"][loc]

    # rest
    if energy < 40:
        yield ((loc, t + 20, energy + 30, money, visited, ate, tickets),
               [{"action": "rest", "place": loc, "start": t, "end": t + 20}], 20)

    at_goal = loc in problem["goals"] and loc not in visited and loc not in problem["closed"]
    # buy_ticket
    if at_goal and place.get("ticketed") and loc not in tickets and money >= place["price"]:
        yield ((loc, t + 5, energy, money - place["price"], visited, ate, tickets | {loc}),
               [{"action": "ticket", "place": loc, "start": t, "end": t + 5}], 5)

    # (wait) + visit
    fee = 0 if place.get("ticketed") else place.get("price", 0)
    if at_goal and (not place.get("ticketed") or loc in tickets) and energy >= 20 and money >= fee:
        start = max(t, place["open_min"], problem["earliest"].get(loc, 0))
        p_high, queue = problem["forecast"](loc, start) if problem["forecast"] else (0.0, 0.0)
        end = start + queue + place["duration"]
        if start - t <= 90 and end <= place["close_min"]:
            steps = []
            if start > t:
                steps.append({"action": "wait", "place": loc, "start": t, "end": start})
            steps.append({"action": "visit", "place": loc, "start": start, "end": end, "queue": queue})
            used = VISIT_ENERGY_PER_HOUR * place["duration"] / 60
            yield ((loc, end, energy - used, money - fee, visited | {loc}, ate, tickets), steps,
                   end - t + problem["crowd_cost"] * p_high)

    # eat
    if problem["lunch"] == loc and not ate and money >= place["price"]:
        start = max(t, LUNCH_FROM)
        if start <= LUNCH_TO:
            steps = []
            if start > t:
                steps.append({"action": "wait", "place": loc, "start": t, "end": start})
            steps.append({"action": "eat", "place": loc, "start": start, "end": start + 50})
            yield ((loc, start + 50, min(100.0, energy + 25), money - place["price"], visited, True, tickets),
                   steps, start + 50 - t)

    # travel: to an unvisited goal, to lunch, or home when everything is done
    if at_goal and loc in tickets:          # bought a ticket here: visit before leaving
        return
    targets = [g for g in problem["goals"] if g not in visited and g not in problem["closed"]]
    if problem["lunch"] and not ate:
        targets.append(problem["lunch"])
    if not targets:
        targets = [problem["hotel"]]
    for d in targets:
        if d == loc:
            continue
        r = problem["table"][(loc, d)]
        if r is None:
            continue
        tired = WALK_ENERGY_PER_KM * r["walk_km"]
        if money < r["fare"] or energy - tired < 0:
            continue
        step = {"action": "travel", "place": d, "from": loc, "start": t, "end": t + r["minutes"],
                "route": r}
        cost = r["minutes"] + problem["co2_weight"] * r["co2"] / 10
        yield ((d, t + r["minutes"], energy - tired, money - r["fare"], visited, ate, tickets), [step], cost)


def heuristic(problem, s):
    """Time still needed for the remaining visits and lunch. Never too high -> admissible."""
    city = problem["city"]
    _, _, _, _, visited, ate, _ = s
    h = sum(city["by_name"][g]["duration"] for g in problem["goals"] - visited if g not in problem["closed"])
    if problem["lunch"] and not ate:
        h += 50
    return h


def plan(problem, state=None, max_expansions=100000):
    """A* over world states. Returns a list of steps (each step stores the state after it), or None."""
    state = state or start_state(problem)
    counter = 0
    frontier = [(heuristic(problem, state), counter, 0.0, state, [])]
    seen = set()
    expansions = 0
    while frontier and expansions < max_expansions:
        _, _, g, s, steps = heapq.heappop(frontier)
        if is_goal(problem, s):
            return steps
        loc, t, energy, money, visited, ate, tickets = s
        key = (loc, visited, ate, tickets, int(energy // 10), int(t // 5))
        if key in seen:
            continue
        seen.add(key)
        expansions += 1
        for s2, new_steps, cost in successors(problem, s):
            if s2[1] > problem["end"]:                 # past the end of the day: prune
                continue
            new_steps[-1]["state"] = s2
            counter += 1
            heapq.heappush(frontier, (g + cost + heuristic(problem, s2), counter, g + cost, s2,
                                      steps + new_steps))
    return None


def plan_with_relaxation(problem):
    """Plan; if the goals cannot all be reached, drop the least valuable one and retry.
    Returns (steps, dropped places)."""
    dropped = []
    while True:
        steps = plan(problem)
        if steps is not None or not problem["goals"]:
            return steps, dropped
        worst = min(problem["goals"], key=lambda g: interest_utility(problem["city"]["by_name"][g], problem["profile"]))
        problem["goals"] = problem["goals"] - {worst}
        dropped.append(worst)


def plan_summary(steps):
    travel = [s for s in steps if s["action"] == "travel"]
    return {"travel_min": sum(s["route"]["minutes"] for s in travel),
            "co2_g": sum(s["route"]["co2"] for s in travel),
            "queue_min": sum(s.get("queue", 0) for s in steps),
            "back_at": steps[-1]["end"] if steps else None}


def print_plan(steps, title=None):
    if title:
        print("\n" + "=" * 72 + "\n " + title + "\n" + "=" * 72)
    if not steps:
        print("  No plan found.")
        return
    for s in steps:
        line = "  %s-%s  %-7s %s" % (hhmm(s["start"]), hhmm(s["end"]), s["action"], s["place"])
        if s["action"] == "travel":
            line = "  %s-%s  %-7s %s -> %s  [%s]" % (hhmm(s["start"]), hhmm(s["end"]), "travel", s["from"],
                                                     s["place"], route_modes(s["route"]))
        elif s["action"] == "visit" and s.get("queue"):
            line += "  (expected queue %.0f min)" % s["queue"]
        elif s["action"] == "wait":
            line += "  (until it opens)"
        print(line)
    sm = plan_summary(steps)
    print("  travel %.0f min, CO2 %.0f g, back at hotel %s" % (sm["travel_min"], sm["co2_g"], hhmm(sm["back_at"])))


# ------------------------------------------------------- independent validator
def validate_plan(city, profile, steps, closed=(), relaxed=()):
    """Re-check every hard constraint of a finished plan without trusting the planner.
    Returns a list of problems; an empty list means the plan is valid."""
    problems = []
    at, t, spent = profile["hotel"], to_min(profile["start"]), 0.0
    visited, tickets, ate = set(), set(), False
    for s in steps:
        p = city["by_name"][s["place"]]
        if s["start"] < t - 0.5:
            problems.append("%s at %s starts before the previous step ends" % (s["action"], hhmm(s["start"])))
        t = max(t, s["end"])
        if s["action"] == "travel":
            if s["from"] != at:
                problems.append("travel from %s but the traveller is at %s" % (s["from"], at))
            for mode in sorted({leg["mode"] for leg in s["route"]["legs"]} - set(profile["modes"])):
                problems.append("uses %s although the traveller does not want it" % mode)
            spent += s["route"]["fare"]
            at = s["place"]
            continue
        if s["place"] != at:
            problems.append("%s at %s but the traveller is somewhere else" % (s["action"], s["place"]))
        if s["action"] == "ticket":
            tickets.add(s["place"])
            spent += p["price"]
        if s["action"] in ("visit", "eat"):
            if s["start"] < p["open_min"] or s["end"] > p["close_min"]:
                problems.append("%s visited outside its opening hours (%s-%s)" % (s["place"], p["open"], p["close"]))
            if s["place"] in closed:
                problems.append("%s is closed" % s["place"])
            if profile.get("accessible_only") and not p.get("accessible", True):
                problems.append("%s is not wheelchair accessible" % s["place"])
            if s["place"] in visited:
                problems.append("%s visited twice" % s["place"])
            visited.add(s["place"])
        if s["action"] == "visit":
            if p.get("ticketed") and s["place"] not in tickets:
                problems.append("%s visited without a ticket" % s["place"])
            if not p.get("ticketed"):
                spent += p.get("price", 0)
        if s["action"] == "eat":
            ate = True
            spent += p["price"]
            if not LUNCH_FROM <= s["start"] <= LUNCH_TO:
                problems.append("lunch at %s is outside 11:30-14:30" % hhmm(s["start"]))
    if at != profile["hotel"]:
        problems.append("the plan does not end at %s" % profile["hotel"])
    if t > to_min(profile["end"]) + 0.5:
        problems.append("the plan ends at %s, after %s" % (hhmm(t), profile["end"]))
    if spent > profile["budget"] + 0.5:
        problems.append("it costs %.0f, more than the budget of %.0f" % (spent, profile["budget"]))
    if profile.get("lunch") and not ate:
        problems.append("there is no lunch break")
    for m in profile.get("must_visit", []):
        if m not in visited and m not in closed and m not in relaxed:
            problems.append("must-visit %s is missing" % m)
    return problems


# --------------------------------------------------------------- replanning
def replan(problem, steps, now, event):
    """Handle an event at minute `now`.
    event = {"type": "closed", "place": name}  or  {"type": "delay", "minutes": m}
    Returns (strategy, new full plan, list of dropped places)."""
    done = [s for s in steps if s["start"] < now]       # steps already started are finished first
    rest = steps[len(done):]
    state = done[-1]["state"] if done else start_state(problem)
    loc, t, energy, money, visited, ate, tickets = state
    t = max(t, now)

    # 1) keep the plan if the event does not touch it
    if event["type"] == "closed":
        problem["closed"].add(event["place"])
        affected = any(s["place"] == event["place"] and s["action"] in ("visit", "travel") for s in rest)
    else:
        t += event["minutes"]
        affected = True
    if not affected:
        return "keep", steps, []
    state = (loc, t, energy, money, visited, ate, tickets)

    # 2) replan the rest of the day with A*;  3) if impossible, drop the least valuable goal
    dropped = [g for g in problem["goals"] if g in problem["closed"] and g not in visited]
    problem["goals"] = problem["goals"] - set(dropped)
    while True:
        new_steps = plan(problem, state)
        if new_steps is not None:
            return ("replan" if not dropped else "replan + drop"), done + new_steps, dropped
        remaining = [g for g in problem["goals"] if g not in visited]
        if not remaining:
            return "failed", done, dropped
        worst = min(remaining, key=lambda g: interest_utility(problem["city"]["by_name"][g], problem["profile"]))
        problem["goals"] = problem["goals"] - {worst}
        dropped.append(worst)
