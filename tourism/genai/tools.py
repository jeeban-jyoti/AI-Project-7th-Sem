"""Tool registry for the future GenAI agents.

Each tool is a plain function over the shared types with a JSON schema, so an
LLM with function calling can use it directly. Everything here already works
(try `./run.sh tools`); only the agents that decide *when* to call these are
missing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from tourism.coordinator.base import FormProfileBuilder
from tourism.coordinator.rule_based import RuleCoordinator
from tourism.coordinator.validator import validate_plan
from tourism.core.types import TravellerProfile
from tourism.m1_search.candidates import select_candidates, utility
from tourism.m1_search.search import CHEAPEST, FASTEST, GREENEST, compare_modes
from tourism.m4_rl.features import ACTIONS, build_stop_options, make_proposal, state_index, state_tuple, valid_actions


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    fn: Callable[..., Any]
    agent: str

    def schema(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def _obj(props: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {"type": "object", "properties": props, "required": required}


class ToolBox:
    def __init__(self, coordinator: RuleCoordinator):
        self.c = coordinator
        self.city = coordinator.city
        self.engine = coordinator.engine
        self.builder = FormProfileBuilder(self.city)
        S = {"type": "string"}
        N = {"type": "number"}
        place = {"type": ["string", "integer"], "description": "place name or node id"}
        self.tools: Dict[str, Tool] = {}
        self._add("poi_info", "Opening hours, capacity, price, themes and zone of a place.", _obj({"poi": place}, ["poi"]),
                  self.poi_info, "attraction")
        self._add("crowd_forecast", "Expected load ratio, queue wait and P(over capacity) at a place and time.",
                  _obj({"poi": place, "time": S}, ["poi", "time"]), self.crowd_forecast, "attraction")
        self._add("compare_modes", "Compare fastest / public transport / walking / lowest-CO2 / cheapest routes.",
                  _obj({"from": place, "to": place, "time": S}, ["from", "to", "time"]), self.compare_modes, "transport")
        self._add("route", "One route (A*) for an objective: fastest, greenest or cheapest.",
                  _obj({"from": place, "to": place, "time": S, "objective": S, "modes": {"type": "array", "items": S}},
                       ["from", "to", "time"]), self.route, "transport")
        self._add("search_local", "Independent restaurants, shops, guides and community experiences near a place.",
                  _obj({"near": place, "radius_m": N}, ["near"]), self.search_local, "local_business")
        self._add("impact", "Environmental and social impact of visiting a place at a time via a route.",
                  _obj({"poi": place, "time": S, "from": place}, ["poi", "time", "from"]), self.impact, "sustainability")
        self._add("plan_day", "Run the full pipeline (CSP, RL suggestions, planner, validator, risk) for a profile.",
                  _obj({"profile": {"type": "object"}}, ["profile"]), self.plan_day, "coordinator")
        self._add("validate", "Check a plan against every hard constraint.",
                  _obj({"profile": {"type": "object"}, "plan": {"type": "object"}}, ["profile", "plan"]), self.validate,
                  "coordinator")
        self._add("rl_suggest", "What the learned policy would suggest for one stop of an itinerary.",
                  _obj({"profile": {"type": "object"}, "poi": place, "time": S, "from": place}, ["profile", "poi", "time", "from"]),
                  self.rl_suggest, "coordinator")

    def _add(self, name, desc, params, fn, agent):
        self.tools[name] = Tool(name, desc, params, fn, agent)

    def list(self) -> List[Dict[str, Any]]:
        return [dict(t.schema(), agent=t.agent) for t in self.tools.values()]

    def call(self, name: str, args: Dict[str, Any]) -> Any:
        return self.tools[name].fn(**args)

    # ------------------------------------------------------------------ helpers
    def _node(self, x) -> int:
        return int(x) if isinstance(x, int) or (isinstance(x, str) and x.isdigit()) else self.builder._node_by_name(x)

    @staticmethod
    def _time(s) -> int:
        if isinstance(s, (int, float)):
            return int(s)
        hh, mm = str(s).split(":")
        return int(hh) * 60 + int(mm)

    def _profile(self, d) -> TravellerProfile:
        return d if isinstance(d, TravellerProfile) else self.builder.build_profile(d)

    # ------------------------------------------------------------------ tools
    def poi_info(self, poi) -> Dict[str, Any]:
        p = dict(self.city.poi_by_node[self._node(poi)])
        p["themes"] = {t: w for t, w in zip(self.city.themes, p["themes"]) if w > 0}
        p["zone_name"] = self.city.zone_name(p["zone"])
        return p

    def crowd_forecast(self, poi, time) -> Dict[str, Any]:
        n, t = self._node(poi), self._time(time)
        rho, wait = self.c.day.load_fn(n, t)
        i = self.city.poi_index[n]
        return {"poi": self.city.name_of(n), "time": time, "load_ratio": round(rho, 3), "expected_wait_min": round(wait, 1),
                "p_over_capacity": round(self.c.day.forecast.pvh[i][self.city.slot_of(t)], 3)}

    def compare_modes(self, **kw) -> Dict[str, Any]:
        a, b, t = self._node(kw["from"]), self._node(kw["to"]), self._time(kw["time"])
        out = {}
        for name, r in compare_modes(self.engine, a, b, t, context=self.c.day.key).items():
            out[name] = {k: r.get(k) for k in ("time_min", "money", "co2_kg", "walk_m", "transfers")}
            out[name]["modes"] = [l["mode"] for l in r.get("legs", [])]
        return out

    def route(self, objective: str = "fastest", modes=None, **kw) -> Dict[str, Any]:
        w = {"fastest": FASTEST, "greenest": GREENEST, "cheapest": CHEAPEST}.get(objective, FASTEST)
        return self.engine.route(self._node(kw["from"]), self._node(kw["to"]), self._time(kw["time"]),
                                 modes or ["walk", "transit", "car"], w, "astar", False, self.c.day.key)

    def search_local(self, near, radius_m: float = 1500) -> List[Dict[str, Any]]:
        n = self._node(near)
        return [{"name": q["name"], "kind": q["kind"], "zone": self.city.zone_name(q["zone"]),
                 "distance_m": round(self.city.dist(n, q["node"])), "rating": q["rating"]}
                for q in self.city.nearby(n, radius_m) if q["independent"]]

    def impact(self, poi, time, **kw) -> Dict[str, Any]:
        n, t, a = self._node(poi), self._time(time), self._node(kw["from"])
        rho, wait = self.c.day.load_fn(n, t)
        modes = self.compare_modes(**{"from": a, "to": n, "time": t})
        q = self.city.poi_by_node[n]
        return {"crowd_load_ratio": round(rho, 3), "adds_to_overcrowding": rho > 1.0,
                "co2_kg_by_option": {k: v["co2_kg"] for k, v in modes.items()},
                "less_visited_place": q["popularity"] < 0.5, "independent_business": q["independent"],
                "zone": self.city.zone_name(q["zone"])}

    def plan_day(self, profile) -> Dict[str, Any]:
        res = self.c.plan(self._profile(profile))
        return {"status": res.status, "explanation": res.explanation, "actions": res.plan.get("actions", []),
                "violations": res.violations}

    def validate(self, profile, plan) -> List[str]:
        return validate_plan(self.city, self._profile(profile), plan, self.c.day.conditions)

    def rl_suggest(self, profile, poi, time, **kw) -> Dict[str, Any]:
        prof = self._profile(profile)
        n, t, a = self._node(poi), self._time(time), self._node(kw["from"])
        opts = build_stop_options(self.city, prof, n, t, a, t - 20, prof.end_node, prof.end_min - 30,
                                  self.c.day.load_fn, self.c.matrix, {n}, 60)
        st = state_tuple(opts, "unknown")
        act = self.c.policy.act(state_index(st), st, valid_actions(opts, True, n in prof.must_visit), opts, 0.0)
        prop = make_proposal(act, opts, self.city)
        return {"action": ACTIONS[act], "proposal": prop.to_dict() if prop else None}


def describe_tools(toolbox: ToolBox) -> str:
    return json.dumps(toolbox.list(), indent=1)
