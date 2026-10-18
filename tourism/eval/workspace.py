"""Loads everything a process needs once: config, city, engine, travel
matrix, forecasters and policies; and runs one (scenario, system, seed) day."""
from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, Optional

from tourism.coordinator.rule_based import DayState, RuleCoordinator
from tourism.core.city import City
from tourism.core.config import engine_path, ensure_dir, load_config, path
from tourism.data.personas import generate_travellers
from tourism.engine.client import EngineClient
from tourism.m3_uncertainty.forecaster import CrowdForecaster
from tourism.m3_uncertainty.online import OnlineCrowdUpdater
from tourism.m4_rl.matrix import TravelMatrix
from tourism.m4_rl.policy import load_policy
from tourism.sim.scenarios import resolve
from tourism.sim.simulator import DaySimulator, TravellerRun
from tourism.sim.traveller import SimulatedTraveller


class Workspace:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None, need_models: bool = True):
        self.cfg = cfg or load_config()
        self.data_dir = path(self.cfg, "data_dir")
        self.models_dir = ensure_dir(path(self.cfg, "outputs_dir", "models"))
        self.city_path = os.path.join(self.data_dir, "city.json")
        if not os.path.exists(self.city_path):
            raise FileNotFoundError("data/city.json not found - run ./run.sh data first")
        self.city = City.load(self.city_path)
        self.engine = EngineClient(engine_path(self.cfg), self.city_path)
        mpath = os.path.join(self.models_dir, "matrix.json")
        if os.path.exists(mpath):
            self.matrix = TravelMatrix.load(self.city, mpath)
        else:
            self.matrix = TravelMatrix.build(self.engine, self.city)
            self.matrix.save(mpath)
        self.forecaster: Optional[CrowdForecaster] = None
        fpath = os.path.join(self.models_dir, "forecaster.json")
        if os.path.exists(fpath):
            self.forecaster = CrowdForecaster.load(self.city, fpath)
        elif need_models:
            raise FileNotFoundError("outputs/models/forecaster.json not found - run ./run.sh train first")
        self._policies: Dict[str, Any] = {}

    def policy(self, name: str):
        if name not in self._policies:
            self._policies[name] = load_policy(name, self.models_dir)
        return self._policies[name]

    def forecast(self, kind: str, ctx):
        if kind == "bayes" and self.forecaster is not None:
            return self.forecaster.forecast_day(ctx)
        if kind == "historical" and self.forecaster is not None:
            return self.forecaster.baseline.forecast_day(ctx)
        return None

    def coordinator(self, system: str, scenario: Dict[str, Any], key: str, seed: int) -> RuleCoordinator:
        strat = self.cfg["strategies"][system]
        fc = self.forecast(strat["forecast"], scenario["known"])
        day = DayState(self.city, self.engine, key, fc, scenario["known"], scenario["initial_conditions"],
                       bool(strat["collective"]), self.cfg["simulation"]["traveller_weight"])
        policy = self.policy(strat["policy"])
        return RuleCoordinator(self.city, self.engine, self.cfg, strat, day, self.matrix, policy, self.forecaster, seed)

    def run_day(self, scenario_name: str, system: str, seed: int, travellers: int, willingness: str = "mixed",
                keep_runs: bool = False) -> Dict[str, Any]:
        t0 = time.time()
        sc = resolve(self.city, scenario_name)
        strat = self.cfg["strategies"][system]
        key = "%s|%s|%d|%s" % (system, scenario_name, seed, willingness)
        coord = self.coordinator(system, sc, key, seed)
        people = generate_travellers(self.city, travellers, seed * 100 + 7, willingness)
        rng = random.Random(seed * 31 + 1)
        boost = self.cfg["acceptance"]["explanation_boost"]
        runs = []
        for profile, traits in people:
            res = coord.plan(profile, SimulatedTraveller(profile, traits, rng, boost))
            runs.append(TravellerRun(profile, traits, res))
        t_plan = time.time() - t0
        sim = DaySimulator(self.city, sc, seed, self.cfg, self.matrix)
        updater = None
        if strat["replan"] and strat["forecast"] == "bayes" and coord.day.forecast is not None:
            updater = OnlineCrowdUpdater(self.city, coord.day.forecast, detection=self.cfg["simulation"]["sensor_detection"])
        sim.run(runs, coord, coord.day, bool(strat["replan"]), updater)
        m = sim.metrics(runs)
        m.update({"scenario": scenario_name, "system": system, "seed": seed, "willingness": willingness,
                  "plan_seconds": t_plan, "seconds": time.time() - t0,
                  "violations": sum(len(r.result.violations) for r in runs)})
        self.engine.call("drop_context", key=key)
        out = {"metrics": m, "log": sim.log}
        if keep_runs:
            out["runs"] = runs
            out["sim"] = sim
        return out

    def close(self) -> None:
        self.engine.close()
