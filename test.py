"""
TESTS for every module, shown step by step in the terminal.

    python3 test.py          interactive menu: run one group, step through the tests,
                             or try your own input (plan your own trip, a route, a forecast ...)
    python3 test.py all      run every test without any questions
    python3 test.py 3        run only Module 3's tests (0 = datasets, 1-5 = modules,
                             6 = all modules together)

Every test prints what it checks, the values it found, and PASS or FAIL.
The tests marked "(original test)" are ports of the tests in the first version of the
project (tests/*.py and cpp/tests/test_engine.cpp on the main branch).
To add a test: write a function with a one-line docstring, put @test(module_number)
above it, and use check(...) and show(...) inside it.
"""
import argparse
import contextlib
import io
import itertools
import json
import random
import sys
import traceback
from collections import Counter
from unittest import mock

import main
from generate_data import (CITY_FILE, DAYS, HISTORY_COLUMNS, HISTORY_FILE, HOURS, THEMES, TRANSIT_FILE,
                           TRAVELLERS_FILE, make_city, make_history, make_travellers, queue_waits)
from main import SAMPLE_PROFILES, gini, goals_of, plan_day
from module1_search_csp import (LUNCH_FROM, LUNCH_TO, build_graph, distance_km, heuristic, hhmm, route_modes, search,
                                solve_itinerary, to_min)
from module2_planning import make_problem, plan, plan_with_relaxation, print_plan, replan, validate_plan
from module3_uncertainty import (CROWD, WAIT, BayesNet, _cpt, crowd_distribution, evaluate_forecast,
                                 make_forecast, plan_risk, read_history, sensor_update)
from module4_rl import (ACTIONS, KEEP, SHIFT, SUBSTITUTE, CrowdEnv, TabularAgent, block_index, compare_policies,
                        crowd_level, find_alternative)
import module5_genai
from module5_genai import DEFAULT_PROFILE, ask_profile, explain_plan, parse_request

CONTEXT = {"day": "weekend", "season": "peak", "rain_prob": 0.3}
PAIRS = [("Central Hotel", "Hill Monastery"), ("Lakeside Resort", "Riverside Shrine"),
         ("Riverside Inn", "City Zoo"), ("Lakeside Resort", "Grand Bazaar"), ("Central Hotel", "Botanical Garden")]
MODULE_NAMES = {0: "Datasets (generate_data.py)", 1: "Module 1 - Search and CSP", 2: "Module 2 - Planning and replanning",
                3: "Module 3 - Uncertainty", 4: "Module 4 - Reinforcement learning",
                5: "Module 5 - Generative AI", 6: "All modules together"}


# ------------------------------------------------------------ mini test framework
TESTS = {m: [] for m in MODULE_NAMES}
failures = []


def test(module):
    """Decorator: register a test function under a module number."""
    def register(fn):
        TESTS[module].append(fn)
        return fn
    return register


def check(what, ok):
    print("      %s  %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        failures.append(what)
    return ok


def show(label, value):
    print("      %-28s %s" % (label + ":", value))


def ask(prompt, default=""):
    """input() that also works when nothing can be typed (returns the default)."""
    try:
        answer = input(prompt).strip()
    except EOFError:
        print()
        return default
    return answer or default


class StopStepping(Exception):
    """Raised when the user stops step mode; carries the module's partial result."""
    def __init__(self, module, passed, ran):
        super().__init__()
        self.partial = (module, passed, ran)


def load():
    """Load the city, the Bayesian network and today's forecast once."""
    global city, net, forecast
    if "city" not in globals():
        print("Loading the city and learning the Bayesian network ...")
        city, net = main.setup()
        forecast = make_forecast(net, city, CONTEXT)


data_cache = {}


def history_rows():
    if "history" not in data_cache:
        data_cache["history"] = read_history(HISTORY_FILE)
    return data_cache["history"]


def transit_rows():
    if "transit" not in data_cache:
        data_cache["transit"] = read_history(TRANSIT_FILE)
    return data_cache["transit"]


def run_test(fn, number, total):
    print("\n  [%d/%d] %s" % (number, total, fn.__doc__.strip()))
    before = len(failures)
    try:
        fn()
    except Exception:
        failures.append(fn.__name__)
        print("      ERROR  the test crashed:")
        print("        " + traceback.format_exc().strip().replace("\n", "\n        "))
    return len(failures) == before


def run_module(m, step=False):
    print("\n" + "=" * 72 + "\n  " + MODULE_NAMES[m] + "\n" + "=" * 72)
    passed = 0
    for i, fn in enumerate(TESTS[m], 1):
        passed += run_test(fn, i, len(TESTS[m]))
        if step and ask("      (Enter = next test, q = stop) ").lower() == "q":
            raise StopStepping(m, passed, i)
    print("\n  %s: %d of %d tests passed" % (MODULE_NAMES[m], passed, len(TESTS[m])))
    return passed, len(TESTS[m])


def run_all(step=False):
    load()
    summary = []
    try:
        for m in MODULE_NAMES:
            summary.append((m,) + run_module(m, step))
    except StopStepping as stop:
        summary.append(stop.partial)
        print("\n  stopped.")
    print("\n" + "=" * 72 + "\n  SUMMARY\n" + "=" * 72)
    for m, passed, total in summary:
        print("  %-40s %3d / %-3d %s" % (MODULE_NAMES[m], passed, total, "OK" if passed == total else "FAILED"))
    passed, total = sum(s[1] for s in summary), sum(s[2] for s in summary)
    print("  %-40s %3d / %-3d" % ("Total", passed, total))
    return passed == total


# ------------------------------------------------------------ shared checks
def sustainable_plan(profile=None):
    """CSP itinerary + planning problem + action plan for one traveller."""
    profile = profile or SAMPLE_PROFILES["culture"]
    result = solve_itinerary(city, profile, "sustainable", forecast)
    sights, lunch = goals_of(result)
    problem = make_problem(city, profile, sights, lunch, "sustainable", forecast)
    return result, problem, plan(problem)


def itinerary_checks(result, profile):
    """Check every hard constraint of the CSP on one itinerary."""
    if not check("a feasible itinerary was found", result["feasible"]):
        return
    stops = result["stops"]
    names = [s["name"] for s in stops]
    show("itinerary", ", ".join("%s %s" % (hhmm(s["start"]), s["name"]) for s in stops))
    check("each place at most once", len(names) == len(set(names)))
    check("spends %.0f, within the budget of %d" % (result["spend"], profile["budget"]),
          result["spend"] <= profile["budget"])
    check("back at the hotel at %s, before %s" % (hhmm(result["back_at"]), profile["end"]),
          result["back_at"] <= to_min(profile["end"]))
    sights = [s for s in stops if s["type"] != "restaurant"]
    check("%d sights, at most %d" % (len(sights), profile["max_places"]), len(sights) <= profile["max_places"])
    if profile["lunch"]:
        meals = [s for s in stops if s["type"] == "restaurant"]
        check("one lunch, starting between 11:30 and 14:30 (%s)" % (hhmm(meals[0]["start"]) if meals else "none"),
              len(meals) == 1 and LUNCH_FROM <= meals[0]["start"] <= LUNCH_TO)
    closed = [s["name"] for s in stops if not (city["by_name"][s["name"]]["open_min"] <= s["start"]
                                               and s["end"] <= city["by_name"][s["name"]]["close_min"])]
    check("every place is open for the whole visit", not closed)
    if profile["accessible_only"]:
        check("only wheelchair-accessible places", all(city["by_name"][n]["accessible"] for n in names))
    if profile["must_visit"]:
        missing = [m for m in profile["must_visit"] if m not in result["relaxed"] and m not in names]
        check("must-visit places included (relaxed: %s)" % (", ".join(result["relaxed"]) or "none"), not missing)


def plan_checks(steps, problem):
    """Check that an action plan reaches the goal and every precondition holds."""
    if not check("a plan was found", steps is not None):
        return
    problems = []
    for i, s in enumerate(steps):
        p = city["by_name"][s["place"]]
        if s["action"] == "visit":
            if not (p["open_min"] <= s["start"] and s["end"] <= p["close_min"]):
                problems.append(s["place"] + " is closed during the visit")
            if p.get("ticketed") and not any(t["action"] == "ticket" and t["place"] == s["place"] for t in steps[:i]):
                problems.append(s["place"] + " visited without a ticket")
    for problem_text in problems:
        print("            " + problem_text)
    check("every visit happens while the place is open, with a ticket if needed", not problems)
    check("ends back at %s at %s (deadline %s)" % (steps[-1]["place"], hhmm(steps[-1]["end"]), hhmm(problem["end"])),
          steps[-1]["place"] == problem["hotel"] and steps[-1]["end"] <= problem["end"])


# ===================================================================== DATASETS
def visitable():
    return [p for p in city["places"] if p["type"] in ("attraction", "local", "restaurant")]


@test(0)
def data_deterministic():
    """The generator is repeatable: the same seed always gives the same data"""
    c1, c2 = make_city(7), make_city(7)
    check("same seed -> identical city", c1 == c2)
    check("another seed -> a different city", make_city(8)["places"] != c1["places"])
    h1, t1 = make_history(c1, 11)
    h2, t2 = make_history(c2, 11)
    check("same seed -> identical history (%d rows) and bus delays (%d rows)" % (len(h1), len(t1)),
          h1 == h2 and t1 == t2)
    check("same seed -> identical travellers", make_travellers(c1, seed=4242) == make_travellers(c2, seed=4242))


@test(0)
def data_files_up_to_date():
    """The files in data/ are exactly what generate_data.py makes (JSON round trip, original test)"""
    with open(CITY_FILE) as f:
        on_disk = json.load(f)
    fresh = make_city(on_disk["seed"])
    check("a city survives a JSON round trip", json.loads(json.dumps(fresh)) == fresh)
    check("data/city.json equals a freshly generated city", on_disk == fresh)
    with open(TRAVELLERS_FILE) as f:
        check("data/travellers.json equals freshly generated travellers",
              json.load(f) == make_travellers(fresh, seed=on_disk["seed"] + 4235))


@test(0)
def data_city_size():
    """The city is as large as the original project's"""
    kinds = Counter(p["type"] for p in city["places"])
    show("places", ", ".join("%d %s" % (n, k) for k, n in kinds.items()))
    show("footpaths / bus lines", "%d / %d" % (len(city["walk_links"]), len(city["bus_lines"])))
    check("35 sights, 10 local businesses, 14 restaurants, 6 hotels, 8 bus stations",
          kinds == {"attraction": 35, "local": 10, "restaurant": 14, "hotel": 6, "stop": 8})
    check("at least 300 footpaths", len(city["walk_links"]) >= 300)
    check("4 bus lines, each through at least 4 districts",
          len(city["bus_lines"]) == 4 and all(len(s) >= 4 for s in city["bus_lines"].values()))


@test(0)
def data_place_fields():
    """Every place has sensible values (opening hours, popularity, rating, price, capacity)"""
    problems = []
    for p in visitable():
        if p["open_min"] >= p["close_min"]:
            problems.append(p["name"] + " opens after it closes")
        if not 0 <= p["popularity"] <= 1 or not 1 <= p["rating"] <= 5:
            problems.append(p["name"] + " has a strange popularity or rating")
        if p["price"] < 0 or p["capacity_per_hour"] <= 0 or p["duration"] <= 0 or not p["categories"]:
            problems.append(p["name"] + " has a bad price, capacity, duration or no category")
    for text in problems:
        print("            " + text)
    check("%d visitable places checked, no problems" % len(visitable()), not problems)
    names = [p["name"] for p in city["places"]]
    check("place names are unique", len(names) == len(set(names)))
    check("every footpath and bus stop refers to a real place",
          all(a in city["by_name"] and b in city["by_name"] for a, b, _ in city["walk_links"])
          and all(s in city["by_name"] for stops in city["bus_lines"].values() for s in stops))


@test(0)
def data_connected():
    """Every place can be reached; wheelchair users reach every accessible place by footpath and bus"""
    def reached(modes, wheelchair):
        graph = build_graph(city, modes, accessible_only=wheelchair)
        seen, todo = {"Central Hotel"}, ["Central Hotel"]
        while todo:
            for e in graph[todo.pop()]:
                if e["to"] not in seen:
                    seen.add(e["to"])
                    todo.append(e["to"])
        return seen
    accessible = [p["name"] for p in city["places"] if p.get("accessible", True)]
    on_foot, wheel = reached(("walk",), False), reached(("walk", "bus"), True)
    show("steep footpaths (not for wheelchairs)", sum(1 for link in city["walk_links"] if link[2]))
    check("on foot: %d of %d places" % (len(on_foot), len(city["places"])), len(on_foot) == len(city["places"]))
    check("wheelchair, footpaths and buses: %d of %d accessible places" % (len(wheel & set(accessible)), len(accessible)),
          set(accessible) <= wheel)


@test(0)
def data_history():
    """history.csv: one row per day, opening hour and place, with valid values"""
    rows = history_rows()
    expected = DAYS * sum(len([h for h in HOURS if p["open"] <= "%02d:00" % h < p["close"]]) for p in visitable())
    show("rows", "%d (%d days, every opening hour from 08:00 to 20:00)" % (len(rows), DAYS))
    show("file columns", ", ".join(HISTORY_COLUMNS))
    check("row count is right", len(rows) == expected)
    check("%d different days" % DAYS, len({r["day"] for r in rows}) == DAYS)
    check("the columns are as documented", list(rows[0]) == HISTORY_COLUMNS)
    check("every crowd level and queue label is known",
          all(r["Crowd"] in CROWD and r["Wait"] in WAIT for r in rows))
    check("visitors and queue minutes are never negative",
          all(int(r["visitors"]) >= 0 and float(r["wait_min"]) >= 0 for r in rows))


@test(0)
def data_patterns():
    """The history has realistic patterns the network can learn"""
    rows = history_rows()

    def share_high(**match):
        sel = [r for r in rows if all(r[k] == v for k, v in match.items())]
        return sum(r["Crowd"] == "high" for r in sel) / len(sel)
    famous = share_high(Fame="famous", Block="midday", Day="weekend")
    early = share_high(Fame="famous", Block="morning", Day="weekday")
    quiet = share_high(Fame="quiet", Block="midday", Day="weekend")
    show("crowded: famous, weekend midday", "%.0f%%" % (100 * famous))
    show("crowded: famous, weekday morning", "%.0f%%" % (100 * early))
    show("crowded: quiet, weekend midday", "%.0f%%" % (100 * quiet))
    check("famous places at weekend middays are often crowded (over 40%)", famous > 0.4)
    check("quiet places almost never are (under 10%)", quiet < 0.1)
    check("weekday mornings are calmer than weekend middays", early < famous)

    def visitors(setting, weather):
        sel = [int(r["visitors"]) for r in rows if r["Setting"] == setting and r["Weather"] == weather]
        return sum(sel) / len(sel)
    show("outdoor visitors per hour, clear / rain", "%.0f / %.0f" % (visitors("outdoor", "clear"), visitors("outdoor", "rain")))
    check("rain keeps people away from outdoor places", visitors("outdoor", "rain") < visitors("outdoor", "clear"))


@test(0)
def data_queue():
    """Queues build up when more people arrive than can get in, then drain (original test)"""
    w = queue_waits([0, 150, 150, 0, 0], capacity=100, slot_min=15)
    show("arrivals 0,150,150,0,0, capacity 100", "waits " + ", ".join("%.1f" % x for x in w))
    check("no queue at the start", w[0] == 0)
    check("the backlog carries over, so the wait grows", w[2] > w[1])
    check("and it drains when arrivals stop", w[4] < w[3])


@test(0)
def data_transit():
    """transit_history.csv: buses are late more often in rain and at rush hour"""
    rows = transit_rows()
    check("%d rows = %d days x 4 lines x %d hours" % (len(rows), DAYS, len(HOURS)), len(rows) == DAYS * 4 * len(HOURS))
    check("'late' means a delay of 5 minutes or more",
          all((r["BusDelay"] == "late") == (float(r["delay_min"]) >= 5) for r in rows))

    def late(**match):
        sel = [r for r in rows if all(r[k] == v for k, v in match.items())]
        return sum(r["BusDelay"] == "late" for r in sel) / len(sel)
    show("late in rain / clear", "%.0f%% / %.0f%%" % (100 * late(Weather="rain"), 100 * late(Weather="clear")))
    show("late at rush hour / other hours", "%.0f%% / %.0f%%" % (100 * late(peak="1"), 100 * late(peak="0")))
    check("rain makes buses late more often", late(Weather="rain") > late(Weather="clear"))
    check("rush hour makes buses late more often", late(peak="1") > late(peak="0"))


@test(0)
def data_travellers():
    """travellers.json: 300 valid travellers of 7 kinds"""
    travellers = main.load_travellers()
    problems = []
    for t in travellers:
        if city["by_name"].get(t["hotel"], {}).get("type") != "hotel":
            problems.append(t["name"] + ": unknown hotel")
        if not t["interests"] or any(k not in THEMES or not 0 < v <= 1 for k, v in t["interests"].items()):
            problems.append(t["name"] + ": bad interests")
        for m in t["must_visit"]:
            if m not in city["by_name"] or (t["accessible_only"] and not city["by_name"][m]["accessible"]):
                problems.append(t["name"] + ": impossible must-visit " + m)
        if not {"walk", "bus"} <= set(t["modes"]) or (t["accessible_only"] and "taxi" in t["modes"]):
            problems.append(t["name"] + ": bad transport modes")
        if not (t["start"] < t["end"] and 3 <= t["max_places"] <= 5 and 0 <= t["flexibility"] <= 1):
            problems.append(t["name"] + ": bad times, pace or flexibility")
    for text in problems[:10]:
        print("            " + text)
    kinds = Counter(t["archetype"] for t in travellers)
    show("kinds", ", ".join("%s %d" % kv for kv in sorted(kinds.items())))
    show("wheelchair users / with a must-visit", "%d / %d" % (sum(t["accessible_only"] for t in travellers),
                                                            sum(bool(t["must_visit"]) for t in travellers)))
    check("300 travellers", len(travellers) == 300)
    check("all 7 kinds of traveller appear", len(kinds) == 7)
    check("every traveller profile is valid", not problems)


# ===================================================================== MODULE 1
@test(1)
def time_helpers():
    """Time helpers convert between 'HH:MM' and minutes"""
    show("to_min('09:30')", to_min("09:30"))
    show("hhmm(570)", hhmm(570))
    check("09:30 is 570 minutes after midnight", to_min("09:30") == 570)
    check("570 minutes prints as 09:30", hhmm(570) == "09:30")
    check("17:05 survives a round trip", hhmm(to_min("17:05")) == "17:05")


@test(1)
def reachable_on_foot():
    """Every place in the city can be reached on foot"""
    graph = build_graph(city, ("walk",))
    seen, todo = {"Central Hotel"}, ["Central Hotel"]
    while todo:
        for e in graph[todo.pop()]:
            if e["to"] not in seen:
                seen.add(e["to"])
                todo.append(e["to"])
    show("reached from Central Hotel", "%d of %d places" % (len(seen), len(city["places"])))
    check("the walking graph is connected", len(seen) == len(city["places"]))


@test(1)
def same_start_and_goal():
    """A route from a place to itself is empty and costs nothing"""
    r = search(city, build_graph(city), "Royal Fort", "Royal Fort")
    show("legs, cost", (len(r["legs"]), r["cost"]))
    check("no legs and zero cost", r["legs"] == [] and r["cost"] == 0)


@test(1)
def astar_optimal():
    """A* finds the same cost as UCS and expands no more nodes"""
    graph = build_graph(city)
    for co2_weight in (0.0, 1.0):
        print("      co2_weight = %.0f (%s)" % (co2_weight, "fastest route" if co2_weight == 0 else "greener route"))
        for a, b in PAIRS:
            u = search(city, graph, a, b, "ucs", co2_weight)
            s = search(city, graph, a, b, "astar", co2_weight)
            check("%-38s UCS %6.1f/%3d nodes   A* %6.1f/%3d nodes" % (a + " -> " + b, u["cost"], u["expanded"],
                                                                     s["cost"], s["expanded"]),
                  abs(u["cost"] - s["cost"]) < 1e-6 and s["expanded"] <= u["expanded"])


@test(1)
def greedy_not_cheaper():
    """Greedy best-first is fast but never finds a cheaper route than UCS"""
    graph = build_graph(city)
    for a, b in PAIRS:
        g, u = search(city, graph, a, b, "greedy"), search(city, graph, a, b, "ucs")
        check("%-38s Greedy %6.1f (%d nodes) >= UCS %6.1f" % (a + " -> " + b, g["cost"], g["expanded"], u["cost"]),
              g["cost"] + 1e-9 >= u["cost"])


@test(1)
def heuristic_admissible():
    """The A* heuristic never over-estimates the real cost (admissible)"""
    graph = build_graph(city)
    for a, b in PAIRS:
        h, cost = heuristic(city, a, b), search(city, graph, a, b, "ucs")["cost"]
        check("%-38s h = %5.1f <= real cost %5.1f" % (a + " -> " + b, h, cost), h <= cost)


@test(1)
def route_connected():
    """Each leg of a route starts where the previous one ended"""
    r = search(city, build_graph(city), "Lakeside Resort", "Riverside Shrine", "astar")
    show("route", " ".join([r["legs"][0]["from"]] + ["-%s-> %s" % (e["mode"], e["to"]) for e in r["legs"]]))
    check("starts at Lakeside Resort and ends at Riverside Shrine",
          r["legs"][0]["from"] == "Lakeside Resort" and r["legs"][-1]["to"] == "Riverside Shrine")
    check("no gaps between legs", all(e1["to"] == e2["from"] for e1, e2 in zip(r["legs"], r["legs"][1:])))


@test(1)
def csp_constraints():
    """Every itinerary obeys every hard constraint (4 travellers x 2 systems)"""
    for name, profile in SAMPLE_PROFILES.items():
        for system in ("conventional", "sustainable"):
            print("      --- %s, %s" % (profile["name"], system))
            itinerary_checks(solve_itinerary(city, profile, system, forecast), profile)


@test(1)
def csp_small_budget():
    """With a budget of only 600 the itinerary still stays within budget"""
    profile = dict(SAMPLE_PROFILES["culture"], budget=600)
    itinerary_checks(solve_itinerary(city, profile, "sustainable", forecast), profile)


@test(1)
def csp_must_visit():
    """A must-visit place is always included"""
    profile = dict(SAMPLE_PROFILES["culture"], must_visit=["Riverside Ruins"])
    result = solve_itinerary(city, profile, "sustainable", forecast)
    show("itinerary", ", ".join(s["name"] for s in result["stops"]))
    check("Riverside Ruins is in the plan", "Riverside Ruins" in [s["name"] for s in result["stops"]])
    check("nothing had to be relaxed", result["relaxed"] == [])


@test(1)
def csp_impossible():
    """An impossible must-visit is explained and relaxed"""
    result = solve_itinerary(city, SAMPLE_PROFILES["impossible"], "sustainable", forecast)
    show("must-visit asked", ", ".join(SAMPLE_PROFILES["impossible"]["must_visit"]))
    show("reason given", result["reasons"][0] if result["reasons"] else "-")
    check("a plan is still found", result["feasible"])
    check("only Hill Monastery is relaxed", result["relaxed"] == ["Hill Monastery"])
    check("the reason mentions wheelchair access", bool(result["reasons"]) and "wheelchair" in result["reasons"][0])


@test(1)
def csp_less_co2():
    """The sustainable itinerary emits less CO2 than the conventional one"""
    profile = SAMPLE_PROFILES["culture"]
    c = solve_itinerary(city, profile, "conventional", forecast)
    s = solve_itinerary(city, profile, "sustainable", forecast)
    show("CO2 conventional / sustainable", "%.0f g / %.0f g" % (c["co2_g"], s["co2_g"]))
    check("sustainable emits less", s["co2_g"] < c["co2_g"])


@test(1)
def astar_random_pairs():
    """A* finds UCS's optimal cost on 25 random pairs of places (original test)"""
    rng = random.Random(3)
    names = [p["name"] for p in city["places"] if p["type"] != "stop"]
    graph = build_graph(city)
    ok, ucs_nodes, astar_nodes = True, 0, 0
    for _ in range(25):
        a, b = rng.sample(names, 2)
        u, s = search(city, graph, a, b, "ucs", 1.0), search(city, graph, a, b, "astar", 1.0)
        ok = ok and abs(u["cost"] - s["cost"]) < 1e-6 and s["expanded"] <= u["expanded"]
        ucs_nodes, astar_nodes = ucs_nodes + u["expanded"], astar_nodes + s["expanded"]
    show("nodes expanded in total, UCS / A*", "%d / %d (A* saves %.0f%%)" % (ucs_nodes, astar_nodes,
                                                                           100 - 100 * astar_nodes / ucs_nodes))
    check("same cost on all 25 pairs, never more nodes", ok)


@test(1)
def wheelchair_route():
    """Wheelchair routes never use steep paths or places without step-free access (original test)"""
    steep = {(a, b) for a, b, s in city["walk_links"] if s} | {(b, a) for a, b, s in city["walk_links"] if s}
    normal = build_graph(city, ("walk", "bus"))
    wheel = build_graph(city, ("walk", "bus"), accessible_only=True)
    for a, b in [("Central Hotel", "Hill Lodge"), ("Hill Lodge", "Hilltop Dhaba"), ("Lakeside Resort", "Hill Park Station")]:
        r, n = search(city, wheel, a, b), search(city, normal, a, b)
        bad = [e for e in r["legs"] if (e["mode"] == "walk" and (e["from"], e["to"]) in steep)
               or not city["by_name"][e["to"]].get("accessible", True)]
        check("%-36s %2d legs, %5.1f min (normal route %5.1f min), none steep or inaccessible"
              % (a + " -> " + b, len(r["legs"]), r["minutes"], n["minutes"]), not bad and r["cost"] >= n["cost"] - 1e-9)


@test(1)
def greenest_avoids_taxi():
    """The greenest route never takes a taxi (original test)"""
    graph = build_graph(city)
    for a, b in PAIRS:
        fast, green = search(city, graph, a, b, "astar", 0.0), search(city, graph, a, b, "astar", 5.0)
        check("%-36s fastest: %-16s greenest: %s" % (a + " -> " + b, route_modes(fast), route_modes(green)),
              all(e["mode"] != "taxi" for e in green["legs"]))


@test(1)
def suspended_line():
    """A suspended bus line is never used (original test)"""
    a, b = "Lakeside Resort", "Hill Park Station"
    route = search(city, build_graph(city, ("walk", "bus")), a, b)
    used = sorted({e["line"] for e in route["legs"] if e["mode"] == "bus"})
    show("lines used normally", ", ".join(used))
    detour = search(city, build_graph(city, ("walk", "bus"), suspended_lines=used[:1]), a, b)
    show("lines used with %s suspended" % used[0], ", ".join(sorted({e["line"] for e in detour["legs"] if e["line"]})))
    check("a route is still found", detour is not None)
    check(used[0] + " is not used", all(e["line"] != used[0] for e in detour["legs"]))
    check("the detour is not cheaper than the normal route", detour["cost"] >= route["cost"] - 1e-9)


@test(1)
def csp_opening_hours_conflict():
    """An evening-only must-visit cannot fit a day that ends at 15:00 (original test)"""
    profile = dict(SAMPLE_PROFILES["culture"], must_visit=["Night Food Street"], end="15:00")
    result = solve_itinerary(city, profile, "sustainable", forecast)
    show("reason given", result["reasons"][0] if result["reasons"] else "-")
    check("Night Food Street is relaxed", result["relaxed"] == ["Night Food Street"])
    check("the reason names its opening hours (17:00)", bool(result["reasons"]) and "17:00" in result["reasons"][0])
    check("the rest of the day is still planned", result["feasible"])


# ===================================================================== MODULE 2
@test(2)
def plan_goal():
    """The planner turns the itinerary into actions that reach the goal"""
    result, problem, steps = sustainable_plan()
    print_plan(steps)
    sights, _ = goals_of(result)
    check("every goal place visited: " + ", ".join(sights),
          {s["place"] for s in steps if s["action"] == "visit"} == set(sights))
    check("lunch eaten exactly once", sum(s["action"] == "eat" for s in steps) == 1)
    plan_checks(steps, problem)


@test(2)
def plan_timing():
    """Actions follow each other in time (no overlaps)"""
    _, _, steps = sustainable_plan()
    overlaps = [(a["action"], b["action"]) for a, b in zip(steps, steps[1:]) if b["start"] + 1e-9 < a["end"]]
    show("number of actions", len(steps))
    check("no action starts before the previous one ends", not overlaps)


@test(2)
def plan_preconditions():
    """Preconditions hold: tickets, lunch window, money and energy"""
    _, _, steps = sustainable_plan()
    states = [s["state"] for s in steps if "state" in s]
    show("lowest energy / money", "%.0f / %.0f" % (min(st[2] for st in states), min(st[3] for st in states)))
    for i, s in enumerate(steps):
        p = city["by_name"][s["place"]]
        if s["action"] == "visit" and p.get("ticketed"):
            check("ticket bought before visiting " + s["place"],
                  any(t["action"] == "ticket" and t["place"] == s["place"] for t in steps[:i]))
        if s["action"] == "eat":
            check("lunch at %s, inside 11:30-14:30" % hhmm(s["start"]), LUNCH_FROM <= s["start"] <= LUNCH_TO)
    check("energy never below 0", all(st[2] >= 0 for st in states))
    check("money never below 0", all(st[3] >= 0 for st in states))


@test(2)
def plan_no_unused_ticket():
    """The traveller never leaves a place with a ticket they did not use"""
    for name, profile in SAMPLE_PROFILES.items():
        _, _, steps = sustainable_plan(profile)
        tickets = [i for i, s in enumerate(steps) if s["action"] == "ticket"]
        check("%-34s %d tickets, each followed by its visit" % (profile["name"], len(tickets)),
              all(steps[i + 1]["action"] == "visit" for i in tickets))


@test(2)
def replan_keep():
    """Replanning step 1: keep the plan if the event does not affect it"""
    _, problem, steps = sustainable_plan()
    strategy, new_steps, dropped = replan(problem, steps, steps[1]["end"],
                                          {"type": "closed", "place": "Hill Monastery"})
    show("event", "Hill Monastery closes (not in the plan)")
    show("strategy", strategy)
    check("strategy is 'keep'", strategy == "keep")
    check("the plan is unchanged", new_steps == steps and dropped == [])


@test(2)
def replan_closed():
    """Replanning steps 2-3: a closed place is dropped, the rest is replanned"""
    result, problem, steps = sustainable_plan()
    target = goals_of(result)[0][-1]
    strategy, new_steps, dropped = replan(problem, steps, steps[1]["end"], {"type": "closed", "place": target})
    show("event", "%s closes at %s" % (target, hhmm(steps[1]["end"])))
    show("strategy / dropped", "%s / %s" % (strategy, ", ".join(dropped)))
    check(target + " is dropped", target in dropped)
    check(target + " is not visited", target not in [s["place"] for s in new_steps if s["action"] == "visit"])
    check("still ends at the hotel", new_steps[-1]["place"] == problem["hotel"])


@test(2)
def replan_delay():
    """After a 40-minute delay the rest of the day moves later"""
    _, problem, steps = sustainable_plan()
    now = steps[2]["start"]
    strategy, new_steps, dropped = replan(problem, steps, now, {"type": "delay", "minutes": 40})
    later = [s for s in new_steps if s["start"] >= now]
    show("delay at", hhmm(now))
    show("next action now starts at", hhmm(later[0]["start"]))
    show("strategy / dropped", "%s / %s" % (strategy, ", ".join(dropped) or "nothing"))
    check("the plan was replanned", strategy in ("replan", "replan + drop"))
    check("the next action starts at least 40 minutes later", later[0]["start"] >= now + 40)
    check("back at the hotel at %s, before %s" % (hhmm(new_steps[-1]["end"]), hhmm(problem["end"])),
          new_steps[-1]["place"] == problem["hotel"] and new_steps[-1]["end"] <= problem["end"])


@test(2)
def plan_relaxation():
    """If the day is too short, the least valuable places are dropped"""
    result, _, _ = sustainable_plan()
    sights, lunch = goals_of(result)
    problem = make_problem(city, dict(SAMPLE_PROFILES["culture"], end="13:00"), sights, lunch, "sustainable",
                           forecast)
    steps, dropped = plan_with_relaxation(problem)
    show("goals", ", ".join(sights))
    show("dropped for a day ending 13:00", ", ".join(dropped))
    check("a plan is still found", steps is not None)
    check("at least one place was dropped", len(dropped) > 0)
    check("back by 13:00", steps is not None and steps[-1]["end"] <= to_min("13:00"))


@test(2)
def validator_catches_violations():
    """The independent validator catches every kind of broken plan (original test)"""
    profile = dict(SAMPLE_PROFILES["culture"], budget=100, end="10:00", modes=["walk", "bus"], must_visit=["Old Palace"])
    taxi = search(city, build_graph(city, ("taxi",)), "Central Hotel", "Royal Fort")
    broken = [
        {"action": "travel", "from": "Central Hotel", "place": "Royal Fort", "start": 570, "end": 575, "route": taxi},
        {"action": "visit", "place": "Royal Fort", "start": 500, "end": 1100},
        {"action": "eat", "place": "Fort View Diner", "start": 1100, "end": 1150},
        {"action": "visit", "place": "Royal Fort", "start": 1150, "end": 1160},
    ]
    problems = validate_plan(city, profile, broken)
    for text in problems:
        print("            found: " + text)
    for word in ("taxi", "opening hours", "starts before", "without a ticket", "somewhere else", "outside 11:30",
                 "twice", "does not end", "after 10:00", "budget", "must-visit Old Palace"):
        check("caught: '%s'" % word, any(word in text for text in problems))
    check("a correct plan has no problems", validate_plan(city, SAMPLE_PROFILES["culture"], sustainable_plan()[2]) == [])


@test(2)
def validator_on_dataset():
    """Plans for 6 travellers from the dataset all pass the validator (original test)"""
    for profile in main.load_travellers()[:6]:
        out = plan_day(city, net, None, profile, CONTEXT, show=False)
        check("%-24s %s" % (profile["name"], "; ".join(out.get("violations", [])) or "valid"),
              out["feasible"] and not out["violations"])


# ===================================================================== MODULE 3
@test(3)
def history_size():
    """The network learns from the visitor history and the bus delay history"""
    show("rows learned from", "%d visitor-hours, %d bus-hours" % (len(history_rows()), len(transit_rows())))
    days = {r["day"] for r in history_rows()}
    check("%d days of visitor history" % len(days), len(days) == DAYS)
    late = net.nodes["BusDelay"]["cpt"]
    show("learned P(bus late | clear / rain)", "%.2f / %.2f" % (late[("clear",)]["late"], late[("rain",)]["late"]))
    check("the bus delay table was learned from data (not left at 50/50)", late[("clear",)]["late"] < 0.4)


@test(3)
def cpts_sum_to_one():
    """Every row of every probability table sums to 1"""
    for name, node in net.nodes.items():
        sums = [sum(d.values()) for d in node["cpt"].values()]
        check("%-9s %4d rows, sums between %.6f and %.6f" % (name, len(sums), min(sums), max(sums)),
              all(abs(s - 1) < 1e-9 for s in sums))


@test(3)
def query_distribution():
    """P(Crowd | evidence) is a proper probability distribution"""
    d = crowd_distribution(net, city, CONTEXT, "Royal Fort", 750)
    show("P(crowd) at Royal Fort 12:30", "  ".join("%s %.2f" % (c, d[c]) for c in CROWD))
    check("has the three crowd levels", set(d) == set(CROWD))
    check("sums to 1", abs(sum(d.values()) - 1) < 1e-9)


@test(3)
def famous_more_crowded():
    """A famous place at a busy time is more likely crowded than a quiet place"""
    busy = crowd_distribution(net, city, CONTEXT, "Royal Fort", 750)
    quiet = crowd_distribution(net, city, {"day": "weekday", "season": "off", "weather": "clear"},
                               "Riverside Ruins", 600)
    show("P(high) Royal Fort, peak weekend 12:30", "%.2f" % busy["high"])
    show("P(high) Riverside Ruins, weekday 10:00", "%.2f" % quiet["high"])
    check("the famous place is more likely crowded", busy["high"] > quiet["high"])


@test(3)
def rain_indoors():
    """The network learned that rain moves people indoors"""
    def expected_crowd(name, weather):         # low = 0, medium = 1, high = 2
        d = crowd_distribution(net, city, {"day": "weekday", "season": "off", "weather": weather}, name, 750)
        return d["medium"] + 2 * d["high"]
    for name, setting in (("Royal Fort", "outdoor"), ("National Art Gallery", "indoor")):
        show("%s (%s) clear / rain" % (name, setting),
             "%.2f / %.2f" % (expected_crowd(name, "clear"), expected_crowd(name, "rain")))
    check("outdoor place is quieter in rain", expected_crowd("Royal Fort", "rain") < expected_crowd("Royal Fort", "clear"))
    check("indoor place is busier in rain",
          expected_crowd("National Art Gallery", "rain") > expected_crowd("National Art Gallery", "clear"))


@test(3)
def soft_evidence():
    """A '50% rain' forecast gives an answer between 'rain' and 'clear'"""
    def p_high(extra):
        ctx = dict({"day": "weekend", "season": "peak"}, **extra)
        return crowd_distribution(net, city, ctx, "Royal Fort", 750)["high"]
    rain, clear, half = p_high({"weather": "rain"}), p_high({"weather": "clear"}), p_high({"rain_prob": 0.5})
    show("P(high) rain / 50%% / clear", "%.2f / %.2f / %.2f" % (rain, half, clear))
    check("50% lies between the two", min(rain, clear) < half < max(rain, clear))


@test(3)
def sensor_bayes():
    """The sensor update gives the same answer as Bayes' rule by hand"""
    prior = {"low": 0.5, "medium": 0.3, "high": 0.2}
    post = sensor_update(prior, "high", accuracy=0.8)
    by_hand = 0.2 * 0.8 / (0.2 * 0.8 + 0.5 * 0.1 + 0.3 * 0.1)
    show("prior P(high)", 0.2)
    show("after sensor says 'high'", "%.4f (by hand %.4f)" % (post["high"], by_hand))
    check("matches the hand calculation", abs(post["high"] - by_hand) < 1e-12)
    useless = sensor_update(prior, "high", accuracy=1 / 3)
    check("a sensor that guesses randomly changes nothing", all(abs(useless[c] - prior[c]) < 1e-12 for c in CROWD))


@test(3)
def forecast_values():
    """The forecast gives a probability and a queue length"""
    p_high, wait = forecast("Royal Fort", 750)
    show("forecast('Royal Fort', 12:30)", "P(high) %.2f, queue %.1f min" % (p_high, wait))
    check("P(high) is between 0 and 1", 0 <= p_high <= 1)
    check("queue is not negative", wait >= 0)
    check("same time block gives the same answer (13:00)", forecast("Royal Fort", 780) == (p_high, wait))


@test(3)
def network_beats_baseline():
    """On 30 held-out days the network beats each place's usual crowd levels (original test)"""
    ev = evaluate_forecast(history_rows())
    show("held-out hours", ev["test_rows"])
    show("accuracy: network / per-place average", "%.1f%% / %.1f%%" % (100 * ev["bn_accuracy"],
                                                                     100 * ev["baseline_accuracy"]))
    show("Brier score: network / per-place average", "%.3f / %.3f (lower is better)" % (ev["bn_brier"],
                                                                                     ev["baseline_brier"]))
    check("the network is more accurate", ev["bn_accuracy"] > ev["baseline_accuracy"])
    check("the network's probabilities are better (lower Brier score)", ev["bn_brier"] < ev["baseline_brier"])


@test(3)
def monte_carlo():
    """Monte Carlo: a plan with less spare time is riskier"""
    _, _, steps = sustainable_plan()
    relaxed = plan_risk(net, city, CONTEXT, steps, steps[-1]["end"] + 120)
    tight = plan_risk(net, city, CONTEXT, steps, steps[-1]["end"])
    show("P(fail) with 2 hours spare", "%.2f" % relaxed["p_fail"])
    show("P(fail) with no time spare", "%.2f" % tight["p_fail"])
    show("average queue per day", "%.1f min" % relaxed["mean_queue_min"])
    check("probabilities between 0 and 1", 0 <= relaxed["p_fail"] <= 1 and 0 <= tight["p_fail"] <= 1)
    check("less spare time means more risk", tight["p_fail"] > relaxed["p_fail"])


def sprinkler():
    """The classic textbook network: rain and a sprinkler make the grass wet."""
    net_ = BayesNet()
    net_.add("Rain", ["no", "yes"], [], {(): {"no": 0.8, "yes": 0.2}})
    net_.add("Sprinkler", ["off", "on"], ["Rain"], {("no",): {"off": 0.6, "on": 0.4}, ("yes",): {"off": 0.99, "on": 0.01}})
    net_.add("Wet", ["dry", "wet"], ["Rain", "Sprinkler"],
             {("no", "off"): {"dry": 1.0, "wet": 0.0}, ("no", "on"): {"dry": 0.1, "wet": 0.9},
              ("yes", "off"): {"dry": 0.2, "wet": 0.8}, ("yes", "on"): {"dry": 0.01, "wet": 0.99}})
    return net_


@test(3)
def bn_enumeration_by_hand():
    """Inference matches the joint probability worked out by hand (sprinkler example, original test)"""
    sp = sprinkler()

    def joint(r, s, w):
        return sp.p("Rain", r, {}) * sp.p("Sprinkler", s, {"Rain": r}) * sp.p("Wet", w, {"Rain": r, "Sprinkler": s})
    num = {r: sum(joint(r, s, "wet") for s in ("off", "on")) for r in ("no", "yes")}
    by_hand = num["yes"] / sum(num.values())
    post = sp.query("Rain", {"Wet": "wet"})
    show("P(Rain | grass is wet)", "%.6f (by hand %.6f)" % (post["yes"], by_hand))
    check("matches the hand calculation", abs(post["yes"] - by_hand) < 1e-12)
    check("P(Sprinkler | wet) sums to 1", abs(sum(sp.query("Sprinkler", {"Wet": "wet"}).values()) - 1) < 1e-12)


@test(3)
def bn_soft_evidence_prior():
    """Soft evidence equal to the prior changes nothing (original test)"""
    sp = sprinkler()
    plain = sp.query("Wet", {})
    same = sp.query("Wet", {}, soft={"Rain": {"no": 0.8, "yes": 0.2}})
    rainy = sp.query("Wet", {}, soft={"Rain": {"no": 0.2, "yes": 0.8}})
    show("P(wet): no evidence / prior as soft evidence / 80% rain", "%.4f / %.4f / %.4f"
         % (plain["wet"], same["wet"], rainy["wet"]))
    check("the prior as soft evidence gives the same answer", abs(same["wet"] - plain["wet"]) < 1e-12)
    check("a rainier forecast makes wet grass more likely", rainy["wet"] > plain["wet"])


@test(3)
def bn_learning_recovers():
    """Learning from 20,000 samples recovers the true probabilities (original test)"""
    rng, true = random.Random(1), sprinkler()
    rows = []
    for _ in range(20000):
        r = "yes" if rng.random() < 0.2 else "no"
        s = "on" if rng.random() < (0.01 if r == "yes" else 0.4) else "off"
        w = "wet" if rng.random() < true.p("Wet", "wet", {"Rain": r, "Sprinkler": s}) else "dry"
        rows.append({"Rain": r, "Sprinkler": s, "Wet": w})
    rain = _cpt(rows, "Rain", ["no", "yes"], [], [])[()]["yes"]
    sprinkler_on = _cpt(rows, "Sprinkler", ["off", "on"], ["Rain"], [["no", "yes"]])[("no",)]["on"]
    wet = _cpt(rows, "Wet", ["dry", "wet"], ["Rain", "Sprinkler"], [["no", "yes"], ["off", "on"]])
    show("learned P(rain), true 0.20", "%.3f" % rain)
    show("learned P(sprinkler on | no rain), true 0.40", "%.3f" % sprinkler_on)
    check("P(rain) within 0.02", abs(rain - 0.2) < 0.02)
    check("P(sprinkler on | no rain) within 0.03", abs(sprinkler_on - 0.4) < 0.03)
    check("every learned row sums to 1", all(abs(sum(d.values()) - 1) < 1e-9 for d in wet.values()))


@test(3)
def forecast_context():
    """Forecasts are proper distributions and the kind of day matters (original test)"""
    sights = [p["name"] for p in city["places"] if p["type"] == "attraction"]
    busy_ctx = {"day": "weekend", "season": "peak", "weather": "clear"}
    quiet_ctx = {"day": "weekday", "season": "off", "weather": "rain"}
    dists = {(c, s, m): crowd_distribution(net, city, ctx, s, m) for c, ctx in (("busy", busy_ctx), ("quiet", quiet_ctx))
             for s in sights for m in (600, 750, 960)}
    check("all %d forecasts sum to 1" % len(dists), all(abs(sum(d.values()) - 1) < 1e-9 for d in dists.values()))
    busy = sum(d["high"] for k, d in dists.items() if k[0] == "busy")
    quiet = sum(d["high"] for k, d in dists.items() if k[0] == "quiet")
    show("expected crowded sight-hours: peak weekend / rainy off-season weekday", "%.1f / %.1f" % (busy, quiet))
    check("a peak weekend is at least 1.5 times as crowded", busy > 1.5 * quiet)
    bus = net.nodes["BusDelay"]["cpt"]
    check("bus delay probabilities sum to 1", all(abs(sum(bus[(w,)].values()) - 1) < 1e-9 for w in ("clear", "rain")))
    check("buses are more often late in rain", bus[("rain",)]["late"] > bus[("clear",)]["late"])


@test(3)
def sensor_detects_surge():
    """Sensor readings reveal a sudden crowd on a calm weekday morning, but not on a normal day (original test)"""
    prior = crowd_distribution(net, city, {"day": "weekday", "season": "off", "weather": "clear"}, "Royal Fort", 600)
    surge, normal = dict(prior), dict(prior)
    for _ in range(4):
        surge, normal = sensor_update(surge, "high"), sensor_update(normal, "low")
    show("P(high): forecast / after 4 'high' readings / after 4 'low' readings", "%.3f / %.3f / %.3f"
         % (prior["high"], surge["high"], normal["high"]))
    check("the surge is detected (P(high) above 0.9)", surge["high"] > 0.9)
    check("a normal day raises no alarm", normal["high"] <= prior["high"])


# ===================================================================== MODULE 4
rl_cache = {}


def rl_results():
    if "results" not in rl_cache:
        rl_cache["results"] = compare_policies(city)[1]
    return rl_cache["results"]


@test(4)
def rl_helpers():
    """Crowd levels and time blocks are computed correctly"""
    show("crowd_level(0.5, 0.7, 0.95)", [crowd_level(r) for r in (0.5, 0.7, 0.95)])
    show("block_index(07:00 ... 18:20)", [block_index(m) for m in (420, 600, 700, 900, 1100)])
    check("load ratios map to low / medium / high", [crowd_level(r) for r in (0.5, 0.7, 0.95)] == [0, 1, 2])
    check("times map to morning ... evening",
          [block_index(m) for m in (420, 600, 700, 900, 1100)] == [0, 0, 1, 2, 3])


@test(4)
def rl_alternatives():
    """Alternatives are similar, clearly less popular, and nearby"""
    show("alternative for Grand Temple", find_alternative(city, "Grand Temple"))
    check("Grand Temple -> Riverside Shrine", find_alternative(city, "Grand Temple") == "Riverside Shrine")
    check("restaurants are never replaced", find_alternative(city, "Market Biryani House") is None)
    for p in city["places"]:
        alt = find_alternative(city, p["name"], accessible_only=True)
        if alt:
            b = city["by_name"][alt]
            check("%-22s -> %-26s popularity %.2f -> %.2f, %.1f km" % (p["name"], alt, p["popularity"],
                                                                       b["popularity"], distance_km(p, b)),
                  b["popularity"] <= p["popularity"] - 0.2 + 1e-9 and distance_km(p, b) <= 2.5 and b["accessible"])


@test(4)
def rl_valid_actions():
    """Only actions that make sense are allowed"""
    env = CrowdEnv(city)
    for state, expected in (((2, 3, 0), [KEEP]), ((2, 1, 1), [KEEP, SHIFT, SUBSTITUTE]), ((1, 2, 0), [KEEP, SHIFT])):
        got = env.valid_actions(state)
        check("state %s -> %s" % (state, [ACTIONS[a] for a in got]), got == expected)


@test(4)
def rl_updates():
    """Q-learning and SARSA updates match a hand calculation"""
    s, s2 = (2, 1, 1), (0, 2, 0)
    print("      Q(s', keep) = 2, Q(s', shift) = 4, reward 1, gamma 0.5, next action taken = keep")
    for method, expected, why in (("q_learning", 1 + 0.5 * 4, "uses the best next action"),
                                  ("sarsa", 1 + 0.5 * 2, "uses the action actually taken")):
        agent = TabularAgent(method, gamma=0.5)
        agent.q[(s2, KEEP)], agent.q[(s2, SHIFT)] = 2.0, 4.0
        agent.update(s, SUBSTITUTE, 1.0, s2, KEEP, [KEEP, SHIFT])      # first visit: learning rate 1
        check("%-10s Q = %.1f, expected %.1f (%s)" % (method, agent.Q(s, SUBSTITUTE), expected, why),
              abs(agent.Q(s, SUBSTITUTE) - expected) < 1e-12)
    agent = TabularAgent("q_learning")
    agent.update((0, 0, 0), KEEP, 3.0, None, None, [])
    check("at the end of an episode the target is just the reward", abs(agent.Q((0, 0, 0), KEEP) - 3.0) < 1e-12)


@test(4)
def rl_beats_baselines():
    """The learned policies beat doing nothing and acting randomly"""
    results = rl_results()
    print("      %-12s %8s %12s" % ("policy", "reward", "overcrowded"))
    for name, r in results.items():
        print("      %-12s %8.3f %11.1f%%" % (name, r["reward_per_decision"], 100 * r["overcrowded_share"]))
    keep, rnd = results["always keep"], results["random"]
    for name in ("Q-learning", "SARSA"):
        r = results[name]
        check(name + " earns more reward than always keep and random",
              r["reward_per_decision"] > keep["reward_per_decision"] and r["reward_per_decision"] > rnd["reward_per_decision"])
        check(name + " has fewer overcrowded visits than always keep", r["overcrowded_share"] < keep["overcrowded_share"])


@test(4)
def rl_reproducible():
    """Training again gives exactly the same results (everything is seeded)"""
    check("second run identical to the first", compare_policies(city)[1] == rl_results())


@test(4)
def rl_state_space():
    """There are 24 states and every state allows 'keep' (original state-index test)"""
    env = CrowdEnv(city)
    states = list(itertools.product(range(3), range(4), range(2)))
    check("24 different states", len(set(states)) == 24)
    check("'keep' is allowed in every state", all(KEEP in env.valid_actions(s) for s in states))
    check("'shift' only before the evening", all((SHIFT in env.valid_actions(s)) == (s[1] < 3) for s in states))
    check("'substitute' only when a quieter place exists",
          all((SUBSTITUTE in env.valid_actions(s)) == (s[2] == 1) for s in states))


@test(4)
def rl_bandit():
    """Q-learning finds the best action from noisy rewards (original test)"""
    rng = random.Random(0)
    agent = TabularAgent("q_learning", gamma=0.0, seed=1)
    means = {KEEP: 0.0, SHIFT: 0.5, SUBSTITUTE: -0.3}
    s = (2, 1, 1)
    for _ in range(3000):
        a = agent.act(s, list(means), eps=0.3)
        agent.update(s, a, means[a] + rng.gauss(0, 0.3), None, None, [])
    show("learned Q", ", ".join("%s %.2f (true %.1f)" % (ACTIONS[a], agent.Q(s, a), m) for a, m in means.items()))
    check("the best action (shift) is chosen", agent.act(s, list(means)) == SHIFT)


@test(4)
def rl_crowding_hurts():
    """The same choice earns less reward when the place is over-full (original test)"""
    env = CrowdEnv(city)
    rewards = []
    for load in (0.3, 1.5):
        env.reset(seed=1)
        _, name, block = env.queue[0]
        env.load[name][block] = load
        rewards.append(env.step(KEEP)[1])
    show("reward for keeping a stop: calm / packed", "%.2f / %.2f" % tuple(rewards))
    check("crowding lowers the reward", rewards[0] > rewards[1])


# ===================================================================== MODULE 5
@test(5)
def parse_full():
    """A full request is understood"""
    text = "Family trip from Lakeside Resort, wheelchair user, we love nature and the zoo, budget 3000, from 10 to 6"
    p, notes = parse_request(text, city)
    show("request", text)
    show("understood", "; ".join(notes))
    check("hotel Lakeside Resort", p["hotel"] == "Lakeside Resort")
    check("wheelchair: accessible places only, no taxi", p["accessible_only"] and "taxi" not in p["modes"])
    check("interests include nature and family", "nature" in p["interests"] and "family" in p["interests"])
    check("budget 3000, day 10:00-18:00", (p["budget"], p["start"], p["end"]) == (3000, "10:00", "18:00"))


@test(5)
def parse_details():
    """am/pm times, 'no lunch', number of places and must-visits"""
    text = "from 9 am to 5 pm, no lunch, 2 places, I must see Royal Fort and Old Bridge"
    p, notes = parse_request(text, city)
    show("request", text)
    show("understood", "; ".join(notes))
    check("day 09:00-17:00", (p["start"], p["end"]) == ("09:00", "17:00"))
    check("no lunch", p["lunch"] is False)
    check("2 places", p["max_places"] == 2)
    check("must visit Royal Fort and Old Bridge", sorted(p["must_visit"]) == ["Old Bridge", "Royal Fort"])


@test(5)
def parse_defaults():
    """A request with nothing recognisable gives the default profile"""
    p, notes = parse_request("hello", city)
    show("understood", notes or "nothing")
    check("profile equals the defaults", p == DEFAULT_PROFILE)


@test(5)
def explanation():
    """The explanation contains the important numbers"""
    facts = {"conventional": {"crowded": 1.9, "queue": 70, "co2": 639},
             "sustainable": {"stops": 5, "back_at": "15:29", "crowded": 0.3, "queue": 38, "co2": 275, "p_fail": 0.35},
             "suggestions": [{"text": "visit Riverside Shrine instead of Grand Temple", "accepted": True}],
             "relaxed": ["Hill Monastery is not wheelchair accessible"], "plan_b": "a shorter day"}
    text = explain_plan(facts)
    print("      " + text)
    for piece in ("15:29", "0.3", "275 g", "Riverside Shrine", "wheelchair", "35%", "a shorter day"):
        check("mentions '%s'" % piece, piece in text)


@test(5)
def claude_validated():
    """A bad answer from Claude is cleaned up field by field (fake answer, no internet)"""
    bad = {"hotel": "Fake Hotel", "start": "9am", "end": "17:00", "budget": -5,
           "interests": {"history": 0.9, "aliens": 1}, "max_places": 99, "must_visit": ["Royal Fort", "Moon Base"],
           "accessible_only": "yes", "lunch": False, "modes": ["taxi", "rocket"]}
    show("fake Claude answer", json.dumps(bad))
    with mock.patch.object(module5_genai, "_ask_claude", return_value="Here you go: " + json.dumps(bad)):
        p, notes = parse_request("anything", city, use_llm=True)
    check("unknown hotel ignored -> " + p["hotel"], p["hotel"] == DEFAULT_PROFILE["hotel"])
    check("bad time '9am' ignored -> " + p["start"], p["start"] == DEFAULT_PROFILE["start"])
    check("good time kept -> " + p["end"], p["end"] == "17:00")
    check("negative budget ignored -> %d" % p["budget"], p["budget"] == DEFAULT_PROFILE["budget"])
    check("unknown interest removed -> %s" % p["interests"], p["interests"] == {"history": 0.9})
    check("max_places clipped to 1-6 -> %d" % p["max_places"], p["max_places"] == 6)
    check("unknown place removed -> %s" % p["must_visit"], p["must_visit"] == ["Royal Fort"])
    check("'yes' is not true/false, ignored -> %s" % p["accessible_only"], p["accessible_only"] is False)
    check("unknown mode removed, walking added -> %s" % p["modes"], p["modes"] == ["walk", "taxi"])


@test(5)
def claude_fallback():
    """Without Claude the built-in rules are used"""
    with mock.patch.object(module5_genai, "_ask_claude", return_value=None):
        p, notes = parse_request("budget 1200, we love temples", city, use_llm=True)
    show("understood", "; ".join(notes))
    check("rules found the budget 1200", p["budget"] == 1200)
    with mock.patch.dict(sys.modules, {"anthropic": None}), contextlib.redirect_stdout(io.StringIO()):
        check("a missing anthropic package is handled", module5_genai._ask_claude("hi", "system") is None)


FORM_ANSWERS = ["4", "1", "1", "3", "3,9", "", "2", "5", "2", "1", "2", "1"]


@test(5)
def guided_form():
    """The guided form turns numbered choices into the right profile"""
    with mock.patch("builtins.input", side_effect=FORM_ANSWERS), contextlib.redirect_stdout(io.StringIO()):
        p, ctx = ask_profile(city)
    show("answers typed", " | ".join(a or "Enter" for a in FORM_ANSWERS))
    show("profile", "%s, %s-%s, budget %d, %s, %s" % (p["hotel"], p["start"], p["end"], p["budget"],
                                                    p["interests"], p["modes"]))
    check("hotel 4 = Lakeside Resort", p["hotel"] == "Lakeside Resort")
    check("weekday, off season, rainy forecast (80%)", ctx == {"day": "weekday", "season": "off", "rain_prob": 0.8})
    check("enjoys nature first, then crafts", p["interests"] == {"nature": 1.0, "crafts": 0.8})
    check("09:00 to 20:00, budget 2000", (p["start"], p["end"], p["budget"]) == ("09:00", "20:00", 2000))
    check("wheelchair: accessible places, walking and buses only",
          p["accessible_only"] and p["modes"] == ["walk", "bus"])
    check("relaxed pace: 3 places and lunch, no must-visit",
          p["max_places"] == 3 and p["lunch"] and p["must_visit"] == [])


@test(5)
def guided_form_bad_answers():
    """Wrong answers in the form fall back to the suggested option"""
    answers = ["99", "x", "", "0", "42", "1,1,99", "6", "25:99", "", "5", "abc", "", "", ""]
    with mock.patch("builtins.input", side_effect=answers), contextlib.redirect_stdout(io.StringIO()):
        p, ctx = ask_profile(city)
    show("answers typed", " | ".join(a or "Enter" for a in answers))
    check("unknown hotel number -> Central Hotel", p["hotel"] == "Central Hotel")
    check("'x' and Enter -> weekend, peak season", ctx["day"] == "weekend" and ctx["season"] == "peak")
    check("'0' -> sunny (10%)", ctx["rain_prob"] == 0.1)
    check("no valid interest -> history", p["interests"] == {"history": 1.0})
    check("'1,1,99' -> one must-visit, no duplicates", len(p["must_visit"]) == 1)
    check("'25:99' is not a time -> 09:30", p["start"] == "09:30")
    check("'abc' is not an amount -> budget 3000", p["budget"] == 3000)
    check("Enter -> taxis allowed, no wheelchair, normal pace",
          "taxi" in p["modes"] and not p["accessible_only"] and p["max_places"] == 4)


# ========================================================= ALL MODULES TOGETHER
class AlwaysChange:
    """Stand-in RL agent that always suggests a change."""
    def act(self, s, valid, eps=0.0):
        return valid[-1]


@test(6)
def pipeline_every_traveller():
    """Every sample traveller gets a full plan, risk and explanation"""
    for name, profile in SAMPLE_PROFILES.items():
        out = plan_day(city, net, None, profile, CONTEXT, show=False)
        ok = (out["feasible"] and out["steps"][-1]["place"] == profile["hotel"]
              and out["steps"][-1]["end"] <= to_min(profile["end"]) and isinstance(out["explanation"], str)
              and 0 <= out["sustainable"]["p_fail"] <= 1)
        check("%-34s back %s, risk %.0f%%" % (profile["name"], out["sustainable"]["back_at"],
                                              100 * out["sustainable"]["p_fail"]), ok)


@test(6)
def pipeline_accepted():
    """Accepted suggestions are applied by re-solving the CSP"""
    profile = dict(SAMPLE_PROFILES["culture"], flexibility=1.0)          # accepts everything
    out = plan_day(city, net, AlwaysChange(), profile, CONTEXT, show=False)
    for s in out["suggestions"]:
        show("suggestion", "%s -> %s" % (s["text"], "applied" if s["applied"] else "accepted but did not fit"))
    check("suggestions were made", bool(out["suggestions"]))
    check("at least one was applied", any(s["applied"] for s in out["suggestions"]))
    check("the final plan exists", out["steps"] is not None)


@test(6)
def pipeline_rejected():
    """Rejected suggestions leave the plan unchanged"""
    profile = dict(SAMPLE_PROFILES["culture"], flexibility=0.0)          # refuses everything
    before = plan_day(city, net, None, profile, CONTEXT, show=False)
    after = plan_day(city, net, AlwaysChange(), profile, CONTEXT, show=False)
    show("suggestions rejected", len(after["suggestions"]))
    check("none accepted", not any(s["accepted"] for s in after["suggestions"]))
    check("same places as without suggestions",
          [s["name"] for s in after["result"]["stops"]] == [s["name"] for s in before["result"]["stops"]])


@test(6)
def gini_values():
    """Load inequality (Gini): 0 when visits are even, high when everyone goes to one place (original test)"""
    show("gini(1,1,1,1) / gini(0,0,0,10)", "%.2f / %.2f" % (gini([1, 1, 1, 1]), gini([0, 0, 0, 10])))
    check("even visits give 0", abs(gini([1, 1, 1, 1])) < 1e-12)
    check("everyone at one place gives more than 0.7", gini([0, 0, 0, 10]) > 0.7)


@test(6)
def full_day_dataset():
    """A full day for 15 travellers from the dataset (original simulation test)"""
    feasible, problems, visits = 0, 0, 0
    for i, profile in enumerate(main.load_travellers()[:15]):
        out = plan_day(city, net, None, profile, CONTEXT, show=False, seed=i)
        if out["feasible"] and out["steps"]:
            feasible += 1
            problems += len(out["violations"])
            visits += sum(s["action"] == "visit" for s in out["steps"])
    show("travellers with a plan", "%d of 15" % feasible)
    show("average visits per traveller", "%.1f" % (visits / max(1, feasible)))
    check("at least half get a plan", feasible >= 8)
    check("the validator finds no problems in any plan", problems == 0)
    check("more than one visit per traveller on average", visits / max(1, feasible) > 1)


@test(6)
def interactive_planner():
    """The interactive trip planner: choose options, get a plan, try a closure, finish"""
    answers = [""] * 12 + ["3", "1", "1"]          # suggested options, "a place is closed", the 1st, "keep the plan"
    with mock.patch("builtins.input", side_effect=answers + [""] * 30), \
            contextlib.redirect_stdout(io.StringIO()) as screen:
        out = main.cmd_interactive(city, net, argparse.Namespace(llm=False))
    text = screen.getvalue()
    show("screen output", "%d lines" % len(text.splitlines()))
    check("a plan was made", out["feasible"] and out["steps"] is not None)
    check("the plan passes the independent validator", out["violations"] == [])
    check("the closure was replanned", "and it is closed. Replanning:" in text)
    check("the planner said goodbye", "Enjoy your day!" in text)


# ===================================================== TRY YOUR OWN INPUT
def pick_place(prompt, default):
    names = [p["name"] for p in city["places"]]
    answer = ask("  %s (number or name, Enter = %s): " % (prompt, default), default)
    if answer.isdigit() and 1 <= int(answer) <= len(names):
        return names[int(answer) - 1]
    matches = [n for n in names if answer.lower() in n.lower()]
    if len(matches) != 1:
        print("  not found, using " + default)
        return default
    return matches[0]


def list_places():
    names = [p["name"] for p in city["places"]]
    for i in range(0, len(names), 3):
        print("  " + "".join("%3d %-26s" % (j + 1, names[j]) for j in range(i, min(i + 3, len(names)))))


def ask_time(prompt, default):
    answer = ask("  %s (HH:MM, Enter = %s): " % (prompt, default), default)
    parts = answer.split(":")
    if len(parts) == 2 and all(x.isdigit() for x in parts) and int(parts[0]) < 24 and int(parts[1]) < 60:
        return to_min(answer)
    print("  not a time, using " + default)
    return to_min(default)


def try_route():
    """Your own route: UCS vs Greedy vs A*"""
    list_places()
    a = pick_place("From", "Central Hotel")
    b = pick_place("To", "Hill Monastery")
    w = 1.0 if ask("  Fastest (f) or greener (g) route? [g]: ", "g").lower().startswith("g") else 0.0
    graph = build_graph(city)
    results = {m: search(city, graph, a, b, m, w) for m in ("ucs", "greedy", "astar")}
    for m, r in results.items():
        show(m.upper(), "cost %.1f, %d nodes expanded, %.0f min, %.0f g CO2" % (r["cost"], r["expanded"],
                                                                               r["minutes"], r["co2"]))
    best = results["astar"]
    for e in best["legs"]:
        print("        %-26s -> %-26s %-5s %4.1f min" % (e["from"], e["to"], e["mode"], e["minutes"]))
    check("A* cost equals UCS cost (optimal)", abs(best["cost"] - results["ucs"]["cost"]) < 1e-6)
    check("A* expands no more nodes than UCS", best["expanded"] <= results["ucs"]["expanded"])
    check("Greedy is not cheaper than UCS", results["greedy"]["cost"] + 1e-9 >= results["ucs"]["cost"])
    check("heuristic %.1f <= real cost" % heuristic(city, a, b), heuristic(city, a, b) <= results["ucs"]["cost"])


def try_request():
    """Your own trip in plain words: full plan + constraint checks"""
    default = "Riverside Inn, I love history and temples, budget 2000, from 9 to 5, no taxi"
    text = ask('  Describe your trip (Enter = "%s"):\n  > ' % default, default)
    profile, notes = parse_request(text, city)
    show("understood", "; ".join(notes) or "nothing specific, using defaults")
    out = plan_day(city, net, main.get_agent(city), profile, CONTEXT, interactive=True)
    if not out["feasible"]:
        check("a plan exists", False)
        return
    print("\n  Checking your plan:")
    itinerary_checks(out["result"], dict(profile, must_visit=[m for m in profile["must_visit"]
                                                              if m not in out["result"]["relaxed"]]))
    plan_checks(out["steps"], out["problem"])


def try_forecast():
    """Your own crowd forecast, then a live sensor reading"""
    list_places()
    name = pick_place("Place", "Royal Fort")
    minute = ask_time("Time", "12:30")
    day = "weekday" if ask("  weekday or weekend? [weekend]: ", "weekend").startswith("weekd") else "weekend"
    season = "off" if ask("  off or peak season? [peak]: ", "peak").startswith("o") else "peak"
    try:
        rain = min(1.0, max(0.0, float(ask("  chance of rain 0-1 [0.3]: ", "0.3"))))
    except ValueError:
        rain = 0.3
    ctx = {"day": day, "season": season, "rain_prob": rain}
    d = crowd_distribution(net, city, ctx, name, minute)
    p_high, wait = make_forecast(net, city, ctx)(name, minute)
    show("P(low / medium / high)", "%.2f / %.2f / %.2f" % (d["low"], d["medium"], d["high"]))
    show("expected queue", "%.1f min" % wait)
    check("probabilities sum to 1", abs(sum(d.values()) - 1) < 1e-9)
    check("queue is not negative", wait >= 0)
    reading = ask("  What does the live sensor say? low / medium / high (Enter = skip): ", "")
    if reading in CROWD:
        post = sensor_update(d, reading)
        show("after the sensor", "%.2f / %.2f / %.2f" % (post["low"], post["medium"], post["high"]))
        check("P(%s) went up (or was already almost 1)" % reading, post[reading] >= d[reading] - 1e-12)


def try_rl():
    """Ask the learned policy what it would do"""
    agent = main.get_agent(city)
    env = CrowdEnv(city)
    level = {"l": 0, "m": 1, "h": 2}.get(ask("  Crowd level: low / medium / high [h]: ", "h")[:1].lower(), 2)
    block = block_index(ask_time("Time of the visit", "12:30"))
    has_alt = int(ask("  Is there a similar quieter place nearby? y/n [y]: ", "y").lower().startswith("y"))
    s = (level, block, has_alt)
    valid = env.valid_actions(s)
    for a in valid:
        show("Q(%s)" % ACTIONS[a], "%.3f" % agent.Q(s, a))
    choice = agent.act(s, valid)
    show("learned action", ACTIONS[choice])
    check("the action is allowed in this state", choice in valid)
    check("it is the action with the highest Q value", agent.Q(s, choice) == max(agent.Q(s, a) for a in valid))


def try_replan():
    """Make something go wrong during the day and watch the replanning"""
    names = list(SAMPLE_PROFILES)
    key = ask("  Traveller (%s) [culture]: " % ", ".join(names), "culture")
    profile = SAMPLE_PROFILES.get(key, SAMPLE_PROFILES["culture"])
    result, problem, steps = sustainable_plan(profile)
    print_plan(steps, profile["name"])
    visits = [s["place"] for s in steps if s["action"] == "visit"]
    kind = ask("  Event: (c) a place closes or (d) a delay? [c]: ", "c").lower()
    now = ask_time("When", hhmm(steps[2]["start"]))
    if kind.startswith("d"):
        try:
            minutes = int(ask("  How many minutes late? [40]: ", "40"))
        except ValueError:
            minutes = 40
        event = {"type": "delay", "minutes": minutes}
    else:
        print("  places in the plan: " + ", ".join(visits))
        target = ask("  Which place closes? [%s]: " % visits[-1], visits[-1])
        event = {"type": "closed", "place": target if target in visits else visits[-1]}
    strategy, new_steps, dropped = replan(problem, steps, now, event)
    show("strategy", strategy)
    show("dropped", ", ".join(dropped) or "nothing")
    print_plan(new_steps)
    if event["type"] == "closed":
        check("%s is not visited after it closed" % event["place"],
              not any(s["place"] == event["place"] and s["action"] == "visit" and s["start"] >= now
                      for s in new_steps))
    if new_steps:
        check("ends back at the hotel before %s" % profile["end"],
              new_steps[-1]["place"] == profile["hotel"] and new_steps[-1]["end"] <= to_min(profile["end"]))


def try_trip_planner():
    """Plan my whole trip: choose from options, get a plan (the interactive trip planner)"""
    out = main.cmd_interactive(city, net, argparse.Namespace(llm=False))
    if out and out.get("feasible"):
        print("\n  Checking your final plan:")
        check("the plan passes the independent validator", not out["violations"])
        plan_checks(out["steps"], out["problem"])


TRY = [try_trip_planner, try_route, try_request, try_forecast, try_rl, try_replan]


def try_menu():
    while True:
        print("\n  TRY YOUR OWN INPUT")
        for i, fn in enumerate(TRY, 1):
            print("  %d  %s" % (i, fn.__doc__))
        print("  b  back")
        choice = ask("  choose: ", "b").lower()
        if choice == "b":
            return
        if choice.isdigit() and 1 <= int(choice) <= len(TRY):
            before = len(failures)
            TRY[int(choice) - 1]()
            print("\n  result: %s" % ("all checks passed" if len(failures) == before else
                                     "%d check(s) failed" % (len(failures) - before)))


def menu():
    load()
    while True:
        print("\n" + "=" * 72 + "\n  TESTS - choose what to run\n" + "=" * 72)
        for m, name in MODULE_NAMES.items():
            print("  %d  %-40s (%d tests)" % (m, name, len(TESTS[m])))
        print("  a  run all tests")
        print("  s  step through all tests, one at a time")
        print("  t  try your own input")
        print("  q  quit")
        choice = ask("  choose: ", "q").lower()
        if choice in [str(m) for m in MODULE_NAMES]:
            run_module(int(choice))
        elif choice == "a":
            run_all()
        elif choice == "s":
            run_all(step=True)
        elif choice == "t":
            try_menu()
        elif choice == "q":
            return


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg is None:
        menu()
    elif arg == "all":
        sys.exit(0 if run_all() else 1)
    elif arg in [str(m) for m in MODULE_NAMES]:
        load()
        passed, total = run_module(int(arg))
        sys.exit(0 if passed == total else 1)
    else:
        print(__doc__)
        sys.exit(2)
