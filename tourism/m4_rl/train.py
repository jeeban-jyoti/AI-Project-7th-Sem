"""Training and evaluation of the RL agents against the comparison policies."""
from __future__ import annotations

import itertools
import json
import os
import random
import time
from typing import Any, Callable, Dict, List

from tourism.core.city import City
from tourism.m3_uncertainty.forecaster import CrowdForecaster, DayForecast
from tourism.m4_rl.agents import (FIXED, LinUCBAgent, Policy, QLearningAgent, SarsaAgent, save_agent)
from tourism.m4_rl.env import ItineraryEnv
from tourism.m4_rl.matrix import TravelMatrix
from tourism.sim.truth import DayContext


def training_forecasts(forecaster: CrowdForecaster, seed: int = 3) -> List[DayForecast]:
    """A spread of day contexts (calendar x season x weather, some with events)."""
    rng = random.Random(seed)
    out = []
    combos = list(itertools.product(["weekend", "holiday", "weekday"], ["shoulder", "peak"], ["clear", "rain"]))
    combos += [("weekday", "off", "clear"), ("weekend", "off", "clear")]
    for daykind, season, weather in combos:
        events = []
        if rng.random() < 0.25:
            events = [{"type": "public_event", "zone": rng.randrange(len(forecaster.city.zones)), "from": 600,
                       "to": 900, "factor": 1.6}]
        out.append(forecaster.forecast_day(DayContext(daykind, season, weather, events=events)))
    return out


def run_episode(env: ItineraryEnv, agent: Policy, seed: int, eps: float, learn: bool) -> Dict[str, float]:
    s, valid = env.reset(seed)
    if env.done():
        return dict(env.stats)
    a = agent.act(s, env.state, valid, env.opts, eps)
    while not env.done():
        st_prev = env.state
        s2, r, done, info = env.step(a)
        a2 = agent.act(s2, env.state, info["valid"], env.opts, eps) if not done else None
        if learn:
            if isinstance(agent, LinUCBAgent):
                agent.update_bandit(st_prev, a, r)
            else:
                agent.update(s, a, r, s2, info["valid"], done or info["traveller_done"], a2)  # type: ignore[attr-defined]
        s, a = s2, a2
    return dict(env.stats)


def summarize(stats: List[Dict[str, float]], travellers: int) -> Dict[str, float]:
    tot = {k: sum(s[k] for s in stats) for k in stats[0]}
    d = max(1, tot["decisions"])
    n = max(1, travellers * len(stats))
    return {
        "reward_per_decision": tot["reward"] / d,
        "acceptance_rate": tot["accepted"] / max(1, tot["proposals"]),
        "proposals_per_traveller": tot["proposals"] / n,
        "utility_per_stop": tot["utility"] / d,
        "wait_per_stop_min": tot["wait"] / d,
        "crowd_exposure": tot["rho"] / d,
        "overloaded_visits": tot["overload"] / d,
        "co2_kg_per_traveller": tot["co2"] / n,
        "local_per_traveller": tot["local"] / n,
        "asks_per_traveller": tot["asks"] / n,
        "shifts_per_traveller": tot["shifts"] / n,
        "substitutes_per_traveller": tot["substitutes"] / n,
        "mode_switches_per_traveller": tot["mode_switches"] / n,
    }


def train_all(cfg: Dict[str, Any], city: City, matrix: TravelMatrix, forecaster: CrowdForecaster, models_dir: str,
              reports_dir: str, episodes: int = 0, eval_episodes: int = 0, log: Callable[[str], None] = print) -> Dict[str, Any]:
    rl = cfg["rl"]
    episodes = episodes or rl["episodes"]
    eval_episodes = eval_episodes or rl["eval_episodes"]
    forecasts = training_forecasts(forecaster)
    env = ItineraryEnv(city, matrix, forecasts, rl, rl["travellers_per_episode"], cfg["simulation"]["traveller_weight"],
                       cfg["acceptance"]["explanation_boost"])
    agents: Dict[str, Policy] = {
        "q_learning": QLearningAgent(rl["alpha"], rl["gamma"], seed=1),
        "sarsa": SarsaAgent(rl["alpha"], rl["gamma"], seed=2),
        "linucb": LinUCBAgent(rl["linucb_alpha"], seed=3),
    }
    curves: Dict[str, List[float]] = {}
    for name, agent in agents.items():
        t0 = time.time()
        curve = []
        for ep in range(episodes):
            frac = min(1.0, ep / max(1.0, 0.7 * episodes))
            eps = rl["epsilon_start"] + (rl["epsilon_end"] - rl["epsilon_start"]) * frac
            st = run_episode(env, agent, 1000 + ep, eps, learn=True)
            curve.append(st["reward"] / max(1, st["decisions"]))
        curves[name] = curve
        save_agent(agent, os.path.join(models_dir, "rl_%s.json" % name))
        log("  trained %-10s %4d episodes in %5.1fs, last-50 reward/decision %.3f"
            % (name, episodes, time.time() - t0, sum(curve[-50:]) / max(1, len(curve[-50:]))))
    policies: Dict[str, Policy] = dict(agents)
    for name, cls in FIXED.items():
        policies[name] = cls()
    evaluation = {}
    for name, pol in policies.items():
        stats = [run_episode(env, pol, 50000 + ep, 0.0, learn=False) for ep in range(eval_episodes)]
        evaluation[name] = summarize(stats, rl["travellers_per_episode"])
    window = max(1, episodes // 40)
    smooth = {k: [sum(v[i:i + window]) / len(v[i:i + window]) for i in range(0, len(v), window)] for k, v in curves.items()}
    result = {"episodes": episodes, "eval_episodes": eval_episodes, "curves": smooth, "curve_window": window,
              "evaluation": evaluation}
    with open(os.path.join(reports_dir, "rl_results.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    return result
