"""Runs the evaluation grid in parallel worker processes.

Each worker owns one Workspace (and therefore one C++ engine process). Every
system sees the same travellers, background crowds and sensor noise for a
given seed (common random numbers), so differences come from the planner.
"""
from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from tourism.core.config import ensure_dir, load_config, path

_WS = None


def _init_worker() -> None:
    global _WS
    from tourism.eval.workspace import Workspace
    _WS = Workspace()


def _run_job(job: Tuple[str, str, int, int, str]) -> Dict[str, Any]:
    scenario, system, seed, travellers, willingness = job
    assert _WS is not None
    out = _WS.run_day(scenario, system, seed, travellers, willingness)
    m = out["metrics"]
    m["log"] = " | ".join(out["log"])
    return m


def build_jobs(cfg: Dict[str, Any], mode: str) -> List[Tuple[str, str, int, int, str]]:
    ev = cfg["evaluation"][mode]
    n = ev["travellers"]
    jobs = []
    for sc in ev["scenarios"]:
        for system in ev["systems"]:
            for seed in ev["seeds"]:
                jobs.append((sc, system, seed, n, ev["willingness"]))
    sw = ev["willingness_sweep"]
    for level in sw["levels"]:
        for system in sw["systems"]:
            for seed in ev["seeds"]:
                job = (sw["scenario"], system, seed, n, level)
                if job not in jobs:
                    jobs.append(job)
    ab = ev["ablation"]
    for sc in ab.get("scenarios", [ab.get("scenario")]):
        for system in ["sustainable"] + ab["systems"]:
            for seed in ev["seeds"]:
                job = (sc, system, seed, n, ev["willingness"])
                if job not in jobs:
                    jobs.append(job)
    return jobs


def run_experiments(mode: str = "quick", workers: int = 0, log: Callable[[str], None] = print,
                    only_scenarios: Optional[List[str]] = None) -> str:
    cfg = load_config()
    jobs = build_jobs(cfg, mode)
    if only_scenarios:
        jobs = [j for j in jobs if j[0] in only_scenarios]
    workers = workers or max(1, min(6, (os.cpu_count() or 2) - 1))
    out_dir = ensure_dir(path(cfg, "outputs_dir", "results"))
    out_csv = os.path.join(out_dir, "results_%s.csv" % mode)
    log("Running %d simulated days (%s mode) on %d worker process(es)..." % (len(jobs), mode, workers))
    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    if workers == 1:
        _init_worker()
        for k, job in enumerate(jobs):
            rows.append(_run_job(job))
            log("  [%3d/%d] %-20s %-24s seed %d %-6s %5.1fs" % (k + 1, len(jobs), job[0], job[1], job[2], job[4],
                                                               rows[-1]["seconds"]))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            futs = {pool.submit(_run_job, job): job for job in jobs}
            for k, fut in enumerate(as_completed(futs)):
                job = futs[fut]
                rows.append(fut.result())
                log("  [%3d/%d] %-20s %-24s seed %d %-6s %5.1fs" % (k + 1, len(jobs), job[0], job[1], job[2], job[4],
                                                                   rows[-1]["seconds"]))
    rows.sort(key=lambda r: (r["scenario"], r["system"], r["willingness"], r["seed"]))
    fields = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    with open(os.path.join(out_dir, "summary_%s.json" % mode), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    log("Finished in %.0fs -> %s" % (time.time() - t0, out_csv))
    return out_csv


METRICS = ["feasibility", "satisfaction", "activities", "preferred_activities", "avg_wait_min", "co2_kg", "green_share",
           "car_km", "travel_min", "local_visits", "failures", "crowd_exposure", "overloaded_visit_share", "peak_load",
           "overloaded_slot_share", "gini_load", "top5_share", "proposals", "accepted_alternatives", "acceptance_rate",
           "replans", "avg_replan_changes", "plan_ms", "violations"]


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["scenario"], r["system"], r["willingness"]), []).append(r)
    out = []
    for (sc, system, will), rs in sorted(groups.items()):
        d: Dict[str, Any] = {"scenario": sc, "system": system, "willingness": will, "runs": len(rs)}
        for m in METRICS:
            vals = [float(r[m]) for r in rs]
            mean = sum(vals) / len(vals)
            sd = (sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
            d[m] = mean
            d[m + "_ci95"] = 1.96 * sd / max(1.0, len(vals) ** 0.5)
        out.append(d)
    return {"groups": out}


def load_rows(csv_path: str) -> List[Dict[str, Any]]:
    with open(csv_path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k in ("scenario", "system", "willingness", "log"):
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows
