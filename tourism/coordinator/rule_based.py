"""RuleCoordinator: the fixed pipeline that runs modules 1-4 for one traveller.

    profile -> M3 forecast (shared day state, incl. travellers already routed)
            -> M1 candidates + CSP (A* legs inside)          [relax must-visits if unsolvable]
            -> M4 policy per stop -> traveller accepts/rejects -> CSP re-solve with accepted changes
            -> M2 planner (actions, resources, off-peak shifts)
            -> validator (hard constraints, independent of the engine)
            -> M3 risk + plan B when the plan is likely to fail
            -> explainer (templates over ReasonRecords)
            -> routed load added to the day state (collective awareness)

The LLM coordinator that replaces it later implements the same `Coordinator`
interface and calls the same functions as tools (see tourism/genai)."""
from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple

from tourism.coordinator.base import Coordinator, TravellerInterface
from tourism.coordinator.explainer import TemplateExplainer
from tourism.coordinator.validator import validate_plan
from tourism.core.city import City
from tourism.core.types import PlanResult, ReasonRecord, TravellerProfile, fmt_time
from tourism.data.personas import willingness_level
from tourism.engine.client import EngineClient
from tourism.m1_search.candidates import select_candidates, similar_alternatives, to_candidate, utility
from tourism.m3_uncertainty.forecaster import CrowdForecaster, DayForecast
from tourism.m3_uncertainty.risk import plan_risk
from tourism.m4_rl.agents import KeepPolicy, Policy
from tourism.m4_rl.features import (ACTIONS, ADD_LOCAL, ASK, KEEP, SHIFT, SUBSTITUTE, SWITCH_MODE, build_stop_options,
                                    make_proposal, state_index, state_tuple, valid_actions)
from tourism.m4_rl.matrix import TravelMatrix
from tourism.sim.truth import DayContext, queue_waits

VISITS = ("visit", "eat", "local_experience")


class DayState:
    """Per-strategy shared state for one day: the forecast the strategy uses,
    the live conditions, and the load of travellers it has already routed.
    Mirrors the engine context `key`."""

    def __init__(self, city: City, engine: EngineClient, key: str, forecast: Optional[DayForecast], ctx: DayContext,
                 conditions: Dict[str, Any], collective: bool, weight: float):
        self.city = city
        self.engine = engine
        self.key = key
        self.ctx = ctx
        self.collective = collective
        self.weight = weight
        P, S = len(city.pois), city.n_slots
        self.blind = forecast is None
        self.forecast = forecast or DayForecast([[0.0] * S for _ in range(P)], [[0.0] * S for _ in range(P)],
                                                [[0.0] * S for _ in range(P)])
        self.conditions = dict(conditions)
        self.planned = [[0.0] * S for _ in range(P)]
        self.forecast = self._effective(self.forecast)
        engine.set_context(key, self.forecast.load, self.forecast.wait, self.forecast.pvh, self.conditions, reset_planned=True)

    def _effective(self, fc: DayForecast) -> DayForecast:
        """Queue waits under reduced capacity: with timed-entry limits the
        backlog accumulates through the day, so waits are recomputed with the
        effective service capacity rather than scaled."""
        caps = self.conditions.get("capacity_factor", {})
        if not caps or self.blind:
            return fc
        fc = fc.copy()
        for node, f in caps.items():
            i = self.city.poi_index.get(int(node))
            if i is None:
                continue
            cap = self.city.pois[i]["capacity"]
            fc.wait[i] = queue_waits([l * cap for l in fc.load[i]], cap * float(f), self.city.slot_min)
        return fc

    def capfac(self, poi: int) -> float:
        return float(self.conditions.get("capacity_factor", {}).get(str(poi), 1.0))

    def load_fn(self, poi: int, t: float) -> Tuple[float, float]:
        i = self.city.poi_index[poi]
        s = self.city.slot_of(t)
        f = self.capfac(poi)
        base = self.forecast.load[i][s] / f
        rho = base + (self.planned[i][s] / (self.city.pois[i]["capacity"] * f) if self.collective else 0.0)
        wait = self.forecast.wait[i][s] + self.city.slot_min * 2.0 * max(0.0, rho - max(base, 0.8))
        return rho, wait

    def _items(self, actions: List[Dict[str, Any]]) -> List[List[float]]:
        return [[a["poi"], a["start"], a["start"] + 1, self.weight] for a in actions if a["type"] in VISITS]

    def add_plan(self, actions: List[Dict[str, Any]], sign: float = 1.0) -> None:
        items = self._items(actions)
        for poi, start, _, w in items:
            i = self.city.poi_index[int(poi)]
            self.planned[i][self.city.slot_of(start)] += sign * w
        if self.collective and items:
            self.engine.add_load(self.key, [[p, s, e, sign * w] for p, s, e, w in items])

    def update(self, forecast: Optional[DayForecast] = None, conditions: Optional[Dict[str, Any]] = None) -> None:
        if conditions is not None:
            self.conditions = dict(conditions)
        if forecast is not None and not self.blind:
            self.forecast = self._effective(forecast)
        self.engine.set_context(self.key, self.forecast.load, self.forecast.wait, self.forecast.pvh, self.conditions,
                                reset_planned=False)


class RuleCoordinator(Coordinator):
    def __init__(self, city: City, engine: EngineClient, cfg: Dict[str, Any], strategy: Dict[str, Any], day: DayState,
                 matrix: TravelMatrix, policy: Optional[Policy] = None, forecaster: Optional[CrowdForecaster] = None,
                 seed: int = 0):
        self.city = city
        self.engine = engine
        self.cfg = cfg
        self.strategy = strategy
        self.day = day
        self.matrix = matrix
        self.policy = policy or KeepPolicy()
        self.forecaster = forecaster
        self.rng = random.Random(seed)
        self.planning = cfg["planning"]
        self.crowd_aware = bool(strategy.get("crowd_aware"))
        self.hard_capacity = self.planning["hard_capacity"] if self.crowd_aware else 99.0
        self.explainer = TemplateExplainer(city)

    # ---------------------------------------------------------------- engine calls
    def _solve(self, profile: TravellerProfile, cands: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.engine.call("solve", context=self.day.key, profile=profile.to_engine(self.hard_capacity),
                                candidates=cands, weights=self.strategy["weights"], options=self.planning["csp"])

    def _plan_actions(self, profile: TravellerProfile, sol: Dict[str, Any], cands: List[Dict[str, Any]],
                      state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        by_poi = {c["poi"]: c for c in cands}
        goals = [by_poi[a["poi"]] for a in sol["activities"] if a["poi"] in by_poi]
        opts = dict(self.planning["planner"])
        opts["allow_shift"] = self.crowd_aware
        st = state or {"node": profile.start_node, "time": profile.start_min, "budget": profile.budget}
        return self.engine.call("plan", context=self.day.key, profile=profile.to_engine(self.hard_capacity),
                                weights=self.strategy["weights"], goals=goals,
                                hint_order=[a["poi"] for a in sol["activities"]], state=st, options=opts)

    # ---------------------------------------------------------------- main pipeline
    def plan(self, profile: TravellerProfile, traveller: Optional[TravellerInterface] = None) -> PlanResult:
        t0 = time.time()
        timings: Dict[str, float] = {}
        reasons: List[ReasonRecord] = []
        cands = select_candidates(profile, self.city, self.planning)
        sol = self._solve(profile, cands)
        timings["csp_ms"] = sol["stats"]["runtime_ms"]
        if sol["status"] != "ok":
            reasons.append(ReasonRecord("M1", "unsolvable", detail={"conflicts": sol.get("conflicts", [])}))
            if profile.must_visit:
                # relax the must-visits named in the conflict first, then all of them if needed
                named = {c.get("poi") for c in sol.get("conflicts", []) if c.get("poi") is not None}
                for drop in ([m for m in profile.must_visit if m in named], list(profile.must_visit)):
                    if not drop:
                        continue
                    relaxed = dict(profile.to_dict())
                    relaxed["must_visit"] = [m for m in profile.must_visit if m not in drop]
                    trial = TravellerProfile.from_dict(relaxed)
                    trial_cands = [dict(c, must=c["poi"] in trial.must_visit) for c in cands]
                    sol = self._solve(trial, trial_cands)
                    if sol["status"] == "ok":
                        profile, cands = trial, trial_cands
                        reasons.append(ReasonRecord("M1", "relax", detail={
                            "relaxed": ["must-visit " + self.city.name_of(m) for m in drop]}))
                        break
        n_prop = n_acc = 0
        if sol["status"] == "ok" and traveller is not None and not isinstance(self.policy, KeepPolicy):
            t1 = time.time()
            sol, cands, n_prop, n_acc = self._apply_policy(profile, cands, sol, traveller, reasons)
            timings["policy_ms"] = (time.time() - t1) * 1000
        if sol["status"] != "ok":
            plan = {"status": "unsolvable", "actions": [], "totals": {}, "visited": [], "skipped": []}
            res = PlanResult(profile.id, self.strategy["name"], "unsolvable", sol, plan, reasons,
                             self.explainer.explain(plan, reasons), proposals=n_prop, accepted=n_acc)
            res.timings = timings
            return res
        for a in sol["activities"]:
            if a["rho"] > profile.crowd_threshold and self.crowd_aware:
                reasons.append(ReasonRecord("M1", "soft_violation", a["poi"], a["name"], detail={"rho": a["rho"]}))
        plan = self._plan_actions(profile, sol, cands)
        timings["planner_ms"] = plan["stats"]["runtime_ms"]
        violations = validate_plan(self.city, profile, plan, self.day.conditions)
        if violations:
            # The validator rejected the plan (e.g. a must-visit the planner could not fit once queues,
            # tickets and resources were added). Relax the must-visits it could not keep, re-solve and re-plan.
            visited = set(plan.get("visited", []))
            missing = [m for m in profile.must_visit if m not in visited]
            if missing:
                d = profile.to_dict()
                d["must_visit"] = [m for m in profile.must_visit if m not in missing]
                trial = TravellerProfile.from_dict(d)
                trial_cands = [dict(c, must=c["poi"] in trial.must_visit) for c in cands]
                sol2 = self._solve(trial, trial_cands)
                if sol2["status"] == "ok":
                    plan2 = self._plan_actions(trial, sol2, trial_cands)
                    v2 = validate_plan(self.city, trial, plan2, self.day.conditions)
                    if len(v2) < len(violations):
                        profile, cands, sol, plan, violations = trial, trial_cands, sol2, plan2, v2
                        reasons.append(ReasonRecord("M1", "relax", detail={
                            "relaxed": ["must-visit " + self.city.name_of(m) + " (no feasible time once queues are included)"
                                        for m in missing]}))
        risk = backup = None
        if self.strategy.get("contingency") and self.forecaster is not None:
            risk = plan_risk(self.city, plan, self.day.forecast, self.day.ctx, profile.end_min, self.rng,
                             self.cfg["risk"]["samples"], profile.crowd_threshold, self.hard_capacity, self.forecaster)
            if risk["p_fail"] > self.cfg["risk"]["threshold"] and risk["riskiest"]:
                backup = self._backup(profile, cands, risk["riskiest"]["poi"])
                if backup is not None:
                    r = risk["riskiest"]
                    reasons.append(ReasonRecord("M3", "contingency", r["poi"], r["name"],
                                                detail={"risk": r["p_capacity"] + r["p_late"] + 0.5 * r["p_crowded"]}))
        res = PlanResult(profile.id, self.strategy["name"], "ok", sol, plan, reasons, "", backup, risk, n_prop, n_acc,
                         violations, timings)
        res.explanation = self.explainer.explain(plan, reasons, risk, backup)
        self.day.add_plan(plan["actions"])
        res.timings["total_ms"] = (time.time() - t0) * 1000
        return res

    # ---------------------------------------------------------------- M4: per-stop decisions
    def _apply_policy(self, profile: TravellerProfile, cands: List[Dict[str, Any]], sol: Dict[str, Any],
                      traveller: TravellerInterface, reasons: List[ReasonRecord]):
        acts = sol["activities"]
        legs = sol["legs"]
        will = willingness_level(profile.willingness) if profile.willingness is not None else "unknown"
        asked = will != "unknown"
        used = {a["poi"] for a in acts}
        mods = {"forbid": {}, "exclude": set(), "extra": [], "modes": {}}
        accepted: List[Tuple[int, ReasonRecord]] = []
        n_prop = 0
        checked: List[float] = []
        end_time = sol["totals"]["end_time"]
        for k, a in enumerate(acts):
            if a["kind"] == "restaurant":
                continue
            prev_node = legs[k]["from_node"]
            prev_end = legs[k]["depart"]
            nxt = acts[k + 1] if k + 1 < len(acts) else None
            next_node = nxt["poi"] if nxt else profile.end_node
            latest_end = (nxt["start"] - 10) if nxt else profile.end_min - 30
            slack = profile.end_min - end_time - 30
            opts = build_stop_options(self.city, profile, a["poi"], a["start"], prev_node, prev_end, next_node,
                                      latest_end, self.day.load_fn, self.matrix, used, slack)
            st = state_tuple(opts, will)
            must = a["poi"] in profile.must_visit
            valid = valid_actions(opts, asked, must)
            act = self.policy.act(state_index(st), st, valid, opts, 0.0)
            if act == ASK:
                will = traveller.reveal_willingness()
                asked = True
                reasons.append(ReasonRecord("M4", "ask", detail={"answer": will}))
                st = state_tuple(opts, will)
                act = self.policy.act(state_index(st), st, valid_actions(opts, True, must), opts, 0.0)
            checked.append(opts.rho)
            if act in (KEEP, ASK):
                continue
            prop = make_proposal(act, opts, self.city)
            n_prop += 1
            ok = traveller.respond(prop)
            rec = self._proposal_record(prop, opts, a, ok)
            reasons.append(rec)
            if not ok:
                continue
            accepted.append((act, rec))
            if act == SHIFT:
                windows = []
                for s in range(self.city.n_slots):
                    t = self.city.slot_start(s)
                    if self.day.load_fn(a["poi"], t)[0] > max(opts.shift_rho + 0.1, profile.crowd_threshold):
                        windows.append([t, t + self.city.slot_min])
                mods["forbid"][a["poi"]] = windows
            elif act == SUBSTITUTE:
                mods["exclude"].add(a["poi"])
                mods["extra"].append(opts.alt)
                used.add(opts.alt)
            elif act == SWITCH_MODE:
                mods["modes"][a["poi"]] = ["walk", "transit"]
            elif act == ADD_LOCAL:
                mods["extra"].append(opts.local)
                used.add(opts.local)
        if checked:
            reasons.append(ReasonRecord("M4", "review", detail={"stops": len(checked), "max_rho": max(checked),
                                                                "suggested": n_prop}))
        if not accepted:
            return sol, cands, n_prop, 0
        # re-solve with the accepted changes; drop the latest change until the CSP is feasible again
        while True:
            new_cands = self._apply_mods(profile, cands, mods)
            sol2 = self._solve(profile, new_cands)
            if sol2["status"] == "ok":
                return sol2, new_cands, n_prop, len(accepted)
            if not accepted:
                return sol, cands, n_prop, 0
            act, rec = accepted.pop()
            rec.accepted = None
            reasons.append(ReasonRecord("M1", "reverted", rec.poi, rec.poi_name))
            if act == SHIFT:
                mods["forbid"].pop(rec.poi, None)
            elif act == SUBSTITUTE:
                mods["exclude"].discard(rec.poi)
                mods["extra"].remove(rec.detail["alt"])
            elif act == SWITCH_MODE:
                mods["modes"].pop(rec.poi, None)
            elif act == ADD_LOCAL:
                mods["extra"].remove(rec.detail["alt"])

    def _apply_mods(self, profile: TravellerProfile, cands: List[Dict[str, Any]], mods: Dict[str, Any]) -> List[Dict[str, Any]]:
        out = [dict(c) for c in cands if c["poi"] not in mods["exclude"]]
        have = {c["poi"] for c in out}
        for node in mods["extra"]:
            if node not in have:
                out.append(to_candidate(profile, self.city.poi_by_node[node], self.city, must=True))
                have.add(node)
            else:
                for c in out:
                    if c["poi"] == node:
                        c["must"] = True
        for c in out:
            if c["poi"] in mods["forbid"]:
                c["forbid"] = mods["forbid"][c["poi"]]
                c["must"] = True
            if c["poi"] in mods["modes"]:
                c["modes"] = mods["modes"][c["poi"]]
                c["must"] = True
        return out

    def _proposal_record(self, prop, opts, a, ok) -> ReasonRecord:
        city = self.city
        d: Dict[str, Any] = {"old_start": a["start"], "rho_old": opts.rho}
        deltas: Dict[str, float] = {}
        if prop.action == "shift":
            d.update(new_start=opts.shift_start, rho_new=opts.shift_rho)
            deltas = {"wait_min": opts.shift_wait - opts.wait, "rho": opts.shift_rho - opts.rho}
        elif prop.action == "substitute":
            q = city.poi_by_node[opts.alt]
            d.update(alt=opts.alt, alt_name=q["name"], rho_new=opts.alt_rho, match=opts.alt_match,
                     less_visited=q["popularity"] < 0.5, zone=city.zone_name(q["zone"]))
            deltas = {"wait_min": opts.alt_wait - opts.wait, "rho": opts.alt_rho - opts.rho, "travel_min": opts.alt_extra,
                      "utility": opts.alt_utility - opts.utility}
        elif prop.action == "switch_mode":
            deltas = {"travel_min": opts.green_time - opts.leg_time, "co2_kg": opts.green_co2 - opts.leg_co2}
        elif prop.action == "add_local":
            q = city.poi_by_node[opts.local]
            d.update(alt=opts.local, alt_name=q["name"], kind="shop" if q["kind"] == "shop" else "experience")
            deltas = {"travel_min": opts.local_extra}
        return ReasonRecord("M4", prop.action, a["poi"], a["name"], d, "policy", deltas, ok)

    # ---------------------------------------------------------------- contingency (M3)
    def _backup(self, profile: TravellerProfile, cands: List[Dict[str, Any]], risky: int) -> Optional[Dict[str, Any]]:
        if risky in profile.must_visit:
            return None
        alts = similar_alternatives(profile, self.city.poi_by_node[risky], self.city, k=2,
                                    exclude=[c["poi"] for c in cands])
        new = [c for c in cands if c["poi"] != risky] + [to_candidate(profile, q, self.city) for q in alts]
        sol = self._solve(profile, new)
        if sol["status"] != "ok":
            return None
        plan = self._plan_actions(profile, sol, new)
        plan["instead_of"] = self.city.name_of(risky)
        plan["risky_poi"] = risky
        return plan

    # ---------------------------------------------------------------- M2: replanning during the day
    def replan(self, profile: TravellerProfile, state: Dict[str, Any], remaining: List[Dict[str, Any]],
               trigger: Dict[str, Any], previous: Optional[PlanResult] = None) -> Dict[str, Any]:
        goals, seen = [], set()
        for a in remaining:
            if a["type"] in VISITS and a["poi"] not in seen:
                goals.append(to_candidate(profile, self.city.poi_by_node[a["poi"]], self.city,
                                          must=a["poi"] in profile.must_visit))
                seen.add(a["poi"])
        exclude = set(state.get("visited", [])) | seen
        alts: List[Dict[str, Any]] = []
        for g in goals:
            for q in similar_alternatives(profile, self.city.poi_by_node[g["poi"]], self.city, k=2, exclude=exclude):
                c = to_candidate(profile, q, self.city)
                c["replaces"] = g["poi"]
                alts.append(c)
                exclude.add(q["node"])
        opts = dict(self.planning["planner"])
        opts["allow_shift"] = self.crowd_aware
        res = self.engine.call("replan", context=self.day.key, profile=profile.to_engine(self.hard_capacity),
                               weights=self.strategy["weights"], goals=goals, alternatives=alts,
                               previous_order=[g["poi"] for g in goals], state=state, options=opts)
        return res
