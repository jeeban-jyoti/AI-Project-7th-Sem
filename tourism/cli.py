"""Command-line entry point: python -m tourism.cli <command> [options]
(run.sh wraps these; `./run.sh help` lists them)."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

from tourism.core.config import ROOT, ensure_dir, load_config, path


def log(msg: str = "") -> None:
    print(msg, flush=True)


def header(title: str) -> None:
    log("")
    log("=" * 78)
    log(" " + title)
    log("=" * 78)


# ---------------------------------------------------------------------- data
def cmd_data(args) -> None:
    from tourism.core.city import City
    from tourism.data.city_generator import write_city
    from tourism.data.history import generate_history
    cfg = load_config()
    data_dir = ensure_dir(path(cfg, "data_dir"))
    header("Generating the synthetic city and its crowd history")
    city_d = write_city(os.path.join(data_dir, "city.json"), cfg["city"]["seed"])
    city = City(city_d)
    log("city.json: %d places (%d POIs, %d hotels), %d edges, %d transit lines, %d zones"
        % (len(city.nodes), len(city.pois), len(city.hotels), len(city_d["edges"]), len(city.lines), len(city.zones)))
    t0 = time.time()
    info = generate_history(city, cfg["history"]["days"], cfg["history"]["seed"], data_dir)
    log("history.csv.gz: %d days, %d POI-slot observations (%.1fs)" % (info["days"], info["rows"], time.time() - t0))


# ---------------------------------------------------------------------- train
def cmd_train(args) -> None:
    from tourism.core.city import City
    from tourism.engine.client import EngineClient
    from tourism.core.config import engine_path
    from tourism.m3_uncertainty.forecaster import CrowdForecaster, evaluate_forecasters
    from tourism.m4_rl.matrix import TravelMatrix
    from tourism.m4_rl.train import train_all
    cfg = load_config()
    data_dir = path(cfg, "data_dir")
    models = ensure_dir(path(cfg, "outputs_dir", "models"))
    reports = ensure_dir(path(cfg, "outputs_dir", "reports"))
    city = City.load(os.path.join(data_dir, "city.json"))
    train_days = cfg["history"]["days"] - cfg["history"]["test_days"]
    header("Module 3: learning the Bayesian crowd and delay networks")
    t0 = time.time()
    fc = CrowdForecaster.fit(city, os.path.join(data_dir, "history.csv.gz"), os.path.join(data_dir, "transit_history.csv"),
                             train_days)
    fc.save(os.path.join(models, "forecaster.json"))
    log("trained on %d days in %.1fs -> outputs/models/forecaster.json" % (train_days, time.time() - t0))
    ev = evaluate_forecasters(city, os.path.join(data_dir, "history.csv.gz"), train_days,
                              {"Bayesian network": fc, "Historical average": fc.baseline})
    for k, v in ev.items():
        log("  %-20s MAE(load) %.4f  Brier %.4f  accuracy %.1f%%" % (k, v["mae_load"], v["brier"], 100 * v["accuracy"]))
    online = online_demo(cfg, city, fc)
    log("  online update: unannounced crowd at %s from %s detected at %s; demand multiplier estimate %.2f at "
        "detection, up to %.2f during the surge (true %.1f)" % (online["poi"], online["surge_starts"], online["detected_at"],
                                                                online["multiplier"], online["multiplier_during_surge"],
                                                                online["true_factor"]))
    with open(os.path.join(reports, "forecast_eval.json"), "w", encoding="utf-8") as fh:
        json.dump({"models": ev, "online": online}, fh, indent=1)

    header("Travel matrices from the C++ engine")
    eng = EngineClient(engine_path(cfg), os.path.join(data_dir, "city.json"))
    matrix = TravelMatrix.build(eng, city)
    matrix.save(os.path.join(models, "matrix.json"))
    eng.close()
    log("outputs/models/matrix.json (%d places, peak / off-peak, any mode / walk+transit)" % len(matrix.nodes))

    header("Module 4: training Q-learning, SARSA and a LinUCB bandit")
    res = train_all(cfg, city, matrix, fc, models, reports, args.episodes, 0, log)
    log("")
    log("  %-14s %10s %9s %9s %9s %8s" % ("policy", "reward/dec", "utility", "overload%", "CO2/trav", "accept%"))
    for name, v in sorted(res["evaluation"].items(), key=lambda kv: -kv[1]["reward_per_decision"]):
        log("  %-14s %10.3f %9.2f %9.1f %9.2f %8.0f" % (name, v["reward_per_decision"], v["utility_per_stop"],
                                                        100 * v["overloaded_visits"], v["co2_kg_per_traveller"],
                                                        100 * v["acceptance_rate"]))


def online_demo(cfg, city, fc) -> Dict[str, Any]:
    """Feeds the Gamma-Poisson updater the noisy sensor counts of a day with an
    unannounced crowd and reports when it notices."""
    from tourism.core.types import fmt_time
    from tourism.m3_uncertainty.online import OnlineCrowdUpdater
    from tourism.sim.scenarios import resolve
    from tourism.sim.truth import TruthModel
    sc = resolve(city, "S2_sudden_crowd")
    truth = TruthModel(city)
    arrivals = truth.sample_day(sc["true"], random.Random(5))
    fcd = fc.forecast_day(sc["known"])
    up = OnlineCrowdUpdater(city, fcd, detection=cfg["simulation"]["sensor_detection"])
    target = sc["true"].surges[0]
    i = city.poi_index[target["poi"]]
    rng = random.Random(9)
    detected, at_detection, peak = None, 0.0, 0.0
    for s in range(city.n_slots):
        if city.slot_start(s) >= target["to"]:
            break
        k = sum(1 for _ in range(arrivals[i][s]) if rng.random() < cfg["simulation"]["sensor_detection"])
        up.observe(i, s, k)
        if city.slot_start(s) >= target["from"]:
            peak = max(peak, up.multiplier(i))
        if detected is None and i in up.surges():
            detected = city.slot_start(s + 1)
            at_detection = up.multiplier(i)
    return {"poi": city.pois[i]["name"], "surge_starts": fmt_time(target["from"]),
            "detected_at": fmt_time(detected) if detected else "not detected", "multiplier": at_detection,
            "multiplier_during_surge": peak, "true_factor": target["factor"]}


# ---------------------------------------------------------------------- bench
def cmd_bench(args) -> None:
    from tourism.eval.workspace import Workspace
    from tourism.data.personas import generate_travellers
    from tourism.m1_search.candidates import select_candidates
    from tourism.m1_search.search import benchmark
    from tourism.sim.scenarios import resolve
    ws = Workspace()
    cfg = ws.cfg
    header("Module 1: UCS vs Greedy best-first vs A* on the multimodal graph")
    b = benchmark(ws.engine, pairs=args.pairs)
    log("  %-8s %14s %14s %12s %10s" % ("", "nodes expanded", "runtime (us)", "cost vs UCS", "optimal"))
    for name, v in b["algorithms"].items():
        log("  %-8s %14.1f %14.1f %12.3f %9.0f%%" % (name.upper(), v["avg_expanded"], v["avg_runtime_us"],
                                                    v["avg_cost_ratio_vs_ucs"], 100 * v["optimal_share"]))
    header("Module 1: itinerary CSP statistics (peak weekend, sustainable weights)")
    sc = resolve(ws.city, "S1_peak_weekend")
    coord = ws.coordinator("sustainable", sc, "bench", 1)
    stats = {"solves": 0, "solved": 0, "nodes": 0, "backtracks": 0, "pruned": 0, "fc_removed": 0, "runtime_ms": 0,
             "complete": 0, "greedy_equal": 0}
    for profile, _ in generate_travellers(ws.city, args.travellers, 99):
        r = coord._solve(profile, select_candidates(profile, ws.city, cfg["planning"]))
        st = r["stats"]
        stats["solves"] += 1
        stats["solved"] += r["status"] == "ok"
        for k in ("nodes", "backtracks", "pruned", "fc_removed", "runtime_ms"):
            stats[k] += st[k]
        stats["complete"] += bool(st["complete_search"])
        stats["greedy_equal"] += abs(st.get("greedy_score", 0) - r.get("score", 0)) < 1e-6
    n = max(1, stats["solves"])
    for k in ("solved", "nodes", "backtracks", "pruned", "fc_removed", "runtime_ms", "complete", "greedy_equal"):
        stats[k] = stats[k] / n
    log("  %d solves, %.0f%% solved, %.0f nodes, %.0f backtracks, %.0f B&B prunes, %.1f ms on average;"
        % (stats["solves"], 100 * stats["solved"], stats["nodes"], stats["backtracks"], stats["pruned"], stats["runtime_ms"]))
    log("  branch & bound improved on the greedy start in %.0f%% of cases" % (100 * (1 - stats["greedy_equal"])))
    out = os.path.join(ensure_dir(path(cfg, "outputs_dir", "reports")), "search_bench.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"routing": b, "csp": stats}, fh, indent=1)
    ws.close()


# ---------------------------------------------------------------------- plan / demo
def print_plan(city, res, title: str, crowd_blind: bool = False) -> None:
    from tourism.core.types import fmt_time
    header(title)
    if crowd_blind:
        log("  (this planner does not look at crowds, so it shows no crowd estimates)")
    if res.status != "ok":
        log("No feasible itinerary.")
        log(res.explanation)
        return
    for a in res.plan["actions"]:
        t = "%s-%s" % (fmt_time(a["start"]), fmt_time(a["end"]))
        if a["type"] == "travel":
            modes = []
            for s in a.get("segments", []):
                m = s["mode"] + (" " + s["line"] if "line" in s else "")
                if not modes or modes[-1] != m:
                    modes.append(m)
            log("  %s  travel   %s -> %s  [%s]" % (t, city.name_of(a["from"]), city.name_of(a["to"]), ", ".join(modes)))
        elif a["type"] in ("visit", "eat", "local_experience"):
            kind = {"visit": "visit", "eat": "lunch", "local_experience": "local"}[a["type"]]
            if crowd_blind:
                log("  %s  %-8s %s" % (t, kind, a.get("name", "")))
            else:
                log("  %s  %-8s %s  (expected load %d%% of capacity, queue %d min)"
                    % (t, kind, a.get("name", ""), round(100 * a.get("rho", 0)), round(a.get("wait", 0))))
        elif a["type"] == "buy_ticket":
            log("  %s  ticket   %s" % (t, a.get("name", "")))
        else:
            log("  %s  %-8s %s %s" % (t, a["type"].replace("_visit", ""), a.get("name", ""), a.get("note", "")))
    log("")
    for line in res.explanation.splitlines():
        log("  " + line)
    if res.proposals:
        log("  Suggestions offered: %d, accepted: %d" % (res.proposals, res.accepted))
    if res.violations:
        log("  VALIDATOR: " + "; ".join(res.violations))


def _load_profile(ws, path_or_dict):
    from tourism.coordinator.base import FormProfileBuilder
    if isinstance(path_or_dict, str):
        with open(path_or_dict, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    else:
        d = path_or_dict
    return FormProfileBuilder(ws.city).build_profile(d)


def ask_profile(ws) -> Dict[str, Any]:
    """Console form (the stand-in for the GenAI traveller agent)."""
    city = ws.city
    log("Hotels: " + ", ".join(city.name_of(h) for h in city.hotels))
    hotel = input("Your hotel [Central Hotel]: ").strip() or "Central Hotel"
    start = input("Start time [09:30]: ").strip() or "09:30"
    end = input("Be back by [19:00]: ").strip() or "19:00"
    budget = float(input("Budget [3000]: ").strip() or 3000)
    log("Interests (0-1) for: " + ", ".join(city.themes))
    interests = {}
    for t in city.themes:
        v = input("  %s [0.3]: " % t).strip()
        interests[t] = float(v) if v else 0.3
    car = (input("Use taxis? [y/N]: ").strip().lower() or "n").startswith("y")
    wheel = (input("Wheelchair access needed? [y/N]: ").strip().lower() or "n").startswith("y")
    must = input("Must-see place (optional): ").strip()
    return {"id": "U0001", "name": "You", "start": hotel, "start_time": start, "end_time": end, "budget": budget,
            "interests": interests, "modes": ["walk", "transit"] + (["car"] if car else []), "accessible_only": wheel,
            "must_visit_names": [must] if must else []}


def cmd_plan(args) -> None:
    from tourism.coordinator.base import ConsoleTraveller
    from tourism.data.personas import HiddenTraits
    from tourism.eval.workspace import Workspace
    from tourism.sim.scenarios import resolve
    from tourism.sim.traveller import SimulatedTraveller
    ws = Workspace()
    src = ask_profile(ws) if args.interactive else (args.profile or os.path.join(ROOT, "examples", "profiles", "culture_lover.json"))
    profile = _load_profile(ws, src)
    sc = resolve(ws.city, args.scenario)
    # other travellers that day are planned first, so the plan sees the crowds they create
    from tourism.data.personas import generate_travellers
    coord = ws.coordinator(args.system, sc, "plan", 1)
    others = generate_travellers(ws.city, args.background, 4242)
    rng = random.Random(1)
    for p, tr in others:
        coord.plan(p, SimulatedTraveller(p, tr, rng))
    traveller = ConsoleTraveller() if args.interactive else SimulatedTraveller(
        profile, HiddenTraits(args.willingness, 1.0, 1.0, 0.5, 0.3), rng)
    res = coord.plan(profile, traveller)
    print_plan(ws.city, res, "%s - %s plan (%s)" % (profile.name, ws.cfg["strategies"][args.system]["label"],
                                                   sc["label"]), ws.cfg["strategies"][args.system]["forecast"] == "none")
    out = os.path.join(ensure_dir(path(ws.cfg, "outputs_dir", "plans")), "%s_%s.json" % (profile.id, args.system))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res.to_dict(), fh, indent=1, default=str)
    log("\nSaved %s" % os.path.relpath(out, ROOT))
    ws.close()


def cmd_demo(args) -> None:
    from tourism.data.personas import HiddenTraits, generate_travellers
    from tourism.eval.workspace import Workspace
    from tourism.sim.scenarios import resolve
    from tourism.sim.traveller import SimulatedTraveller
    from tourism.core.types import fmt_time
    ws = Workspace()
    city = ws.city
    profile = _load_profile(ws, os.path.join(ROOT, "examples", "profiles", "culture_lover.json"))
    sc = resolve(city, "S1_peak_weekend")
    for system in ("conventional", "sustainable"):
        coord = ws.coordinator(system, sc, "demo-" + system, 1)
        rng = random.Random(3)
        for p, tr in generate_travellers(city, 80, 4242):
            coord.plan(p, SimulatedTraveller(p, tr, rng))
        res = coord.plan(profile, SimulatedTraveller(profile, HiddenTraits(0.7, 1.0, 1.0, 0.4, 0.3), rng))
        print_plan(city, res, "%s on a peak-season weekend (after 80 other travellers were planned)"
                   % ws.cfg["strategies"][system]["label"], ws.cfg["strategies"][system]["forecast"] == "none")
        if system == "sustainable":
            # show a traveller whose plan still had crowded stops, so the RL policy made suggestions
            for p, tr in generate_travellers(city, 40, 777):
                r2 = coord.plan(p, SimulatedTraveller(p, tr, rng))
                if r2.proposals >= 2 and r2.accepted >= 1:
                    print_plan(city, r2, "Another traveller (%s) planned later the same day: suggestions in action"
                               % p.name)
                    break
    unsolvable = _load_profile(ws, os.path.join(ROOT, "examples", "profiles", "impossible_request.json"))
    coord = ws.coordinator("sustainable", sc, "demo-unsolvable", 1)
    res = coord.plan(unsolvable)
    header("An impossible request: the CSP explains the conflict and relaxes it")
    for line in res.explanation.splitlines():
        log("  " + line)

    header("A simulated public-event day (S3) with replanning, 80 travellers")
    out = ws.run_day("S3_public_event", "sustainable", 2, 80, keep_runs=True)
    for line in out["log"]:
        log("  event: " + line)
    replanned = [r for r in out["runs"] if r.replans]
    log("  %d travellers were re-planned; example:" % len(replanned))
    if replanned:
        r = replanned[0]
        for rp in r.replans:
            log("    %s %s at %s -> %s (%d changes)" % (r.profile.id, rp["trigger"], fmt_time(rp["time"]), rp["strategy"], rp["changes"]))
        log("    visits: " + ", ".join("%s %s" % (fmt_time(v["arrive"]), city.name_of(v["poi"])) for v in r.visits))
    m = out["metrics"]
    log("  day metrics: satisfaction %.1f, wait %.1f min, overcrowded visits %.1f%%, CO2 %.2f kg, feasible %.0f%%"
        % (m["satisfaction"], m["avg_wait_min"], 100 * m["overloaded_visit_share"], m["co2_kg"], 100 * m["feasibility"]))
    ws.close()


def cmd_simulate(args) -> None:
    from tourism.eval.workspace import Workspace
    ws = Workspace()
    header("Simulating %s with %s (%d travellers, seed %d)" % (args.scenario, args.system, args.travellers, args.seed))
    out = ws.run_day(args.scenario, args.system, args.seed, args.travellers, args.willingness)
    for line in out["log"]:
        log("  event: " + line)
    for k, v in out["metrics"].items():
        log("  %-24s %s" % (k, ("%.3f" % v) if isinstance(v, float) else v))
    ws.close()


def cmd_evaluate(args) -> None:
    from tourism.eval.experiments import run_experiments
    header("Evaluation (%s)" % args.mode)
    run_experiments(args.mode, args.workers, log, args.scenarios.split(",") if args.scenarios else None)


def cmd_report(args) -> None:
    from tourism.eval.report import build_report
    out = build_report(args.mode)
    log("Report written to %s" % out)


def cmd_tools(args) -> None:
    from tourism.eval.workspace import Workspace
    from tourism.genai.tools import ToolBox
    from tourism.sim.scenarios import resolve
    ws = Workspace()
    coord = ws.coordinator("sustainable", resolve(ws.city, "S1_peak_weekend"), "tools", 1)
    tb = ToolBox(coord)
    header("Tools the GenAI agents will call (already working)")
    for t in tb.list():
        log("  %-16s [%s agent] %s" % (t["name"], t["agent"], t["description"]))
    log("")
    log("crowd_forecast(Royal Fort, 11:30) -> %s" % json.dumps(tb.call("crowd_forecast", {"poi": "Royal Fort", "time": "11:30"})))
    log("search_local(near=Grand Bazaar)  -> %s" % json.dumps(tb.call("search_local", {"near": "Grand Bazaar"})[:3]))
    cm = tb.call("compare_modes", {"from": "Central Hotel", "to": "Botanical Garden", "time": "10:00"})
    for k, v in cm.items():
        log("compare_modes %-18s %5.1f min  %6.1f money  %.2f kg CO2  %s" % (k, v["time_min"] or 0, v["money"] or 0,
                                                                          v["co2_kg"] or 0, "+".join(v["modes"])))
    ws.close()


def cmd_pddl(args) -> None:
    from tourism.m2_planning.domain import ACTIONS, PDDL_DOMAIN, pddl_problem
    from tourism.eval.workspace import Workspace
    from tourism.m1_search.candidates import select_candidates
    from tourism.sim.scenarios import resolve
    ws = Workspace()
    profile = _load_profile(ws, args.profile or os.path.join(ROOT, "examples", "profiles", "culture_lover.json"))
    coord = ws.coordinator("sustainable", resolve(ws.city, "S1_peak_weekend"), "pddl", 1)
    cands = select_candidates(profile, ws.city, ws.cfg["planning"])
    sol = coord._solve(profile, cands)
    goals = [c for c in cands if c["poi"] in {a["poi"] for a in sol["activities"]}]
    load = {a["poi"]: a["rho"] for a in sol["activities"]}
    out_dir = ensure_dir(path(ws.cfg, "outputs_dir", "pddl"))
    with open(os.path.join(out_dir, "domain.pddl"), "w", encoding="utf-8") as fh:
        fh.write(PDDL_DOMAIN)
    with open(os.path.join(out_dir, "problem_%s.pddl" % profile.id), "w", encoding="utf-8") as fh:
        fh.write(pddl_problem(ws.city, profile, goals, load))
    header("Planning domain (Module 2)")
    for a in ACTIONS:
        log("  %-20s pre: %s" % (a["name"], a["pre"]))
        log("  %-20s eff: %s | duration: %s | resources: %s" % ("", a["eff"], a["dur"], a["res"]))
    log("\nPDDL written to outputs/pddl/ (domain.pddl, problem_%s.pddl)" % profile.id)
    ws.close()


def cmd_info(args) -> None:
    from tourism.core.city import City
    cfg = load_config()
    city = City.load(os.path.join(path(cfg, "data_dir"), "city.json"))
    header(city.name)
    for z in city.zones:
        pois = [p for p in city.pois if p["zone"] == z["id"]]
        log("  %-16s %s" % (z["name"], ", ".join(p["name"] for p in pois)))
    log("  Lines: " + "; ".join("%s (%s)" % (l["name"], l["kind"]) for l in city.lines))
    log("  Hotels: " + ", ".join(city.name_of(h) for h in city.hotels))


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="tourism", description="Sustainable and Crowd-Aware Tourism Planner")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("data", help="generate the synthetic city and its crowd history")
    p = sub.add_parser("train", help="train the Bayesian networks (M3) and the RL policies (M4)")
    p.add_argument("--episodes", type=int, default=0)
    p = sub.add_parser("bench", help="search algorithm and CSP benchmarks (M1)")
    p.add_argument("--pairs", type=int, default=300)
    p.add_argument("--travellers", type=int, default=40)
    sub.add_parser("demo", help="plan example travellers and simulate an event day")
    p = sub.add_parser("plan", help="plan a day for one traveller")
    p.add_argument("--profile", default=None)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--scenario", default="S1_peak_weekend")
    p.add_argument("--system", default="sustainable")
    p.add_argument("--background", type=int, default=80, help="other travellers planned before you")
    p.add_argument("--willingness", type=float, default=0.6)
    p = sub.add_parser("simulate", help="simulate one day for one system")
    p.add_argument("--scenario", default="S1_peak_weekend")
    p.add_argument("--system", default="sustainable")
    p.add_argument("--travellers", type=int, default=120)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--willingness", default="mixed")
    p = sub.add_parser("evaluate", help="run the scenario experiments")
    p.add_argument("--mode", default="quick", choices=["quick", "full"])
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--scenarios", default="")
    p = sub.add_parser("report", help="build outputs/report/report.html")
    p.add_argument("--mode", default="quick", choices=["quick", "full"])
    sub.add_parser("tools", help="list the GenAI tool registry and call a few tools")
    p = sub.add_parser("pddl", help="print the planning domain and export PDDL")
    p.add_argument("--profile", default=None)
    sub.add_parser("info", help="describe the city")
    args = ap.parse_args(argv)
    cmds = {"data": cmd_data, "train": cmd_train, "bench": cmd_bench, "demo": cmd_demo, "plan": cmd_plan,
            "simulate": cmd_simulate, "evaluate": cmd_evaluate, "report": cmd_report, "tools": cmd_tools,
            "pddl": cmd_pddl, "info": cmd_info}
    if args.cmd not in cmds:
        ap.print_help()
        sys.exit(1)
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()
