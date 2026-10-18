"""The planning domain: actions with preconditions, effects, duration and
resources, mirrored as PDDL 2.1 (numeric fluents + durative actions) so the
model can also be checked with an off-the-shelf numeric planner (e.g. ENHSP
or OPTIC). The C++ engine implements exactly these actions."""
from __future__ import annotations

from typing import Any, Dict, List

from tourism.core.city import City
from tourism.core.types import TravellerProfile

ACTIONS: List[Dict[str, str]] = [
    {"name": "travel(a, b)", "pre": "at a; route a->b exists for allowed modes; walked + walk(a,b) <= max walk; budget >= fare",
     "eff": "at b; clock += time(a,b); budget -= fare; CO2 += co2(a,b); energy -= walk effort", "dur": "route time (A*)",
     "res": "money, CO2, walking distance, energy"},
    {"name": "buy_ticket(p)", "pre": "at p; p ticketed; budget >= price(p)", "eff": "has ticket; budget -= price",
     "dur": "5 min", "res": "money"},
    {"name": "visit(p)", "pre": "at p; p open; not closed; expected load <= capacity limit; energy >= 20; hunger < 85",
     "eff": "visited(p); clock += wait + duration; energy -= 0.08/min", "dur": "queue wait + visit duration",
     "res": "time, energy, capacity slot"},
    {"name": "eat(r)", "pre": "at restaurant r; inside the lunch window; budget >= meal price",
     "eff": "hunger = 0; energy += 15; lunch done", "dur": "50 min + wait", "res": "money"},
    {"name": "local_experience(x)", "pre": "at independent business x; open; budget >= price",
     "eff": "visited(x); local support +1", "dur": "activity duration", "res": "money, time"},
    {"name": "shift_visit(p)", "pre": "at p; current slot above crowd threshold; a slot >= 0.05 cheaper exists before closing (within 2.5 h)",
     "eff": "clock moves to the off-peak slot; energy recovers slightly", "dur": "until the quieter slot", "res": "time"},
    {"name": "rest", "pre": "energy < 40", "eff": "energy += 40", "dur": "30 min", "res": "time"},
    {"name": "skip(p)", "pre": "p is not a must-visit", "eff": "p dropped; cost += its utility", "dur": "0", "res": "-"},
    {"name": "finish", "pre": "every goal visited or skipped", "eff": "travel to the end point before the day ends",
     "dur": "route time", "res": "money, CO2"},
]

PDDL_DOMAIN = """(define (domain crowd-aware-tourism)
  (:requirements :strips :typing :fluents :durative-actions :negative-preconditions)
  (:types place - object poi - place restaurant - poi)
  (:predicates (at ?p - place) (visited ?p - poi) (open ?p - poi) (closed ?p - poi)
               (ticketed ?p - poi) (has-ticket ?p - poi) (lunch-done))
  (:functions (clock) (budget) (energy) (hunger) (walked) (co2) (max-walk)
              (travel-time ?a ?b - place) (fare ?a ?b - place) (walk-dist ?a ?b - place) (trip-co2 ?a ?b - place)
              (duration ?p - poi) (expected-wait ?p - poi) (price ?p - poi) (load ?p - poi) (capacity-limit)
              (opens ?p - poi) (closes ?p - poi) (lunch-from) (lunch-to))
  (:durative-action travel
    :parameters (?a ?b - place)
    :duration (= ?duration (travel-time ?a ?b))
    :condition (and (at start (at ?a)) (at start (>= (budget) (fare ?a ?b)))
                    (at start (<= (+ (walked) (walk-dist ?a ?b)) (max-walk))))
    :effect (and (at start (not (at ?a))) (at end (at ?b))
                 (at end (decrease (budget) (fare ?a ?b))) (at end (increase (walked) (walk-dist ?a ?b)))
                 (at end (increase (co2) (trip-co2 ?a ?b))) (at end (increase (clock) (travel-time ?a ?b)))))
  (:durative-action buy-ticket
    :parameters (?p - poi)
    :duration (= ?duration 5)
    :condition (and (over all (at ?p)) (at start (ticketed ?p)) (at start (>= (budget) (price ?p))))
    :effect (and (at end (has-ticket ?p)) (at end (decrease (budget) (price ?p)))))
  (:durative-action visit
    :parameters (?p - poi)
    :duration (= ?duration (+ (expected-wait ?p) (duration ?p)))
    :condition (and (over all (at ?p)) (at start (not (closed ?p))) (at start (>= (clock) (opens ?p)))
                    (at end (<= (clock) (closes ?p))) (at start (<= (load ?p) (capacity-limit)))
                    (at start (>= (energy) 20)) (at start (< (hunger) 85)))
    :effect (and (at end (visited ?p)) (at end (decrease (energy) (* 0.08 (duration ?p))))
                 (at end (increase (clock) (+ (expected-wait ?p) (duration ?p))))))
  (:durative-action eat
    :parameters (?r - restaurant)
    :duration (= ?duration 50)
    :condition (and (over all (at ?r)) (at start (>= (clock) (lunch-from))) (at start (<= (clock) (lunch-to)))
                    (at start (>= (budget) (price ?r))))
    :effect (and (at end (lunch-done)) (at end (visited ?r)) (at end (assign (hunger) 0))
                 (at end (increase (energy) 15)) (at end (decrease (budget) (price ?r)))))
  (:durative-action shift-visit
    :parameters (?p - poi)
    :duration (>= ?duration 15)
    :condition (and (over all (at ?p)) (at start (> (load ?p) 0.85)))
    :effect (at end (increase (clock) ?duration)))
  (:durative-action rest
    :parameters (?p - place)
    :duration (= ?duration 30)
    :condition (and (over all (at ?p)) (at start (< (energy) 40)))
    :effect (at end (increase (energy) 40))))
"""


def pddl_problem(city: City, profile: TravellerProfile, goals: List[Dict[str, Any]], load: Dict[int, float]) -> str:
    """A PDDL problem for one traveller and the goals chosen by the CSP."""
    def obj(n: int) -> str:
        return "n%d" % n
    places = {profile.start_node, profile.end_node} | {g["poi"] for g in goals}
    lines = ["(define (problem day-%s) (:domain crowd-aware-tourism)" % profile.id.lower(), "  (:objects"]
    for g in goals:
        p = city.poi_by_node[g["poi"]]
        lines.append("    %s - %s ; %s" % (obj(g["poi"]), "restaurant" if p["kind"] == "restaurant" else "poi", p["name"]))
    for n in places - {g["poi"] for g in goals}:
        lines.append("    %s - place ; %s" % (obj(n), city.name_of(n)))
    lines.append("  )")
    lines.append("  (:init (at %s) (= (clock) %d) (= (budget) %d) (= (energy) 100) (= (hunger) 20) (= (walked) 0)"
                 % (obj(profile.start_node), profile.start_min, profile.budget))
    lines.append("    (= (co2) 0) (= (max-walk) %d) (= (capacity-limit) 1.6) (= (lunch-from) 690) (= (lunch-to) 900)"
                 % profile.max_walk_m)
    for g in goals:
        p = city.poi_by_node[g["poi"]]
        n = obj(g["poi"])
        lines.append("    (= (duration %s) %d) (= (price %s) %d) (= (opens %s) %d) (= (closes %s) %d) (= (load %s) %.2f) (= (expected-wait %s) 0)%s"
                     % (n, p["duration_min"], n, p["price"], n, p["open_min"], n, p["close_min"], n,
                        load.get(g["poi"], 0.0), n, " (ticketed %s)" % n if p["ticketed"] else ""))
    lines.append("    ; travel-time / fare / walk-dist / trip-co2 come from the engine's A* routes")
    lines.append("  )")
    lines.append("  (:goal (and (at %s) %s))" % (obj(profile.end_node), " ".join("(visited %s)" % obj(g["poi"])
                                                                              for g in goals if g.get("must"))))
    lines.append("  (:metric minimize (+ (* 0.02 (total-time)) (co2))))")
    return "\n".join(lines)
