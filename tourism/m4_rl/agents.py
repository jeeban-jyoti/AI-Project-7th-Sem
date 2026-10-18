"""RL agents (Q-learning, SARSA, LinUCB contextual bandit) and the fixed
comparison policies (satisfaction only, minimum distance, environment only,
a simple crowd rule, random)."""
from __future__ import annotations

import json
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from tourism.m4_rl.features import (ACTIONS, ADD_LOCAL, FEATURE_DIM, KEEP, N_STATES, SHIFT, SUBSTITUTE, SWITCH_MODE,
                                    StopOptions, feature_vector)


class Policy:
    name = "policy"
    learns = False

    def act(self, s: int, st: Tuple[int, ...], valid: List[int], opts: Optional[StopOptions], eps: float = 0.0) -> int:
        raise NotImplementedError

    def to_dict(self) -> Dict:
        return {"name": self.name}


class QLearningAgent(Policy):
    name = "q_learning"
    learns = True

    def __init__(self, alpha: float = 0.15, gamma: float = 0.9, seed: int = 0):
        self.q = [[0.0] * len(ACTIONS) for _ in range(N_STATES)]
        self.visits = [[0] * len(ACTIONS) for _ in range(N_STATES)]
        self.alpha, self.gamma = alpha, gamma
        self.rng = random.Random(seed)

    def greedy(self, s: int, valid: Sequence[int]) -> int:
        return max(valid, key=lambda a: (self.q[s][a], -a))

    def act(self, s, st, valid, opts, eps=0.0):
        if self.rng.random() < eps:
            return self.rng.choice(valid)
        return self.greedy(s, valid)

    def step_size(self, s: int, a: int) -> float:
        """Decaying learning rate per (state, action): large early, then averaging."""
        return max(0.02, self.alpha / (1.0 + self.visits[s][a] / 40.0))

    def update(self, s: int, a: int, r: float, s2: int, valid2: Sequence[int], terminal: bool, a2: Optional[int] = None) -> None:
        target = r if terminal else r + self.gamma * max(self.q[s2][b] for b in valid2)
        self.visits[s][a] += 1
        self.q[s][a] += self.step_size(s, a) * (target - self.q[s][a])

    def to_dict(self):
        return {"name": self.name, "alpha": self.alpha, "gamma": self.gamma,
                "q": [[round(v, 4) for v in row] for row in self.q], "visits": self.visits}

    @classmethod
    def from_dict(cls, d):
        a = cls(d["alpha"], d["gamma"])
        a.q = d["q"]
        a.visits = d.get("visits", a.visits)
        return a


class SarsaAgent(QLearningAgent):
    name = "sarsa"

    def update(self, s, a, r, s2, valid2, terminal, a2=None):
        nxt = 0.0 if terminal or a2 is None else self.q[s2][a2]
        self.visits[s][a] += 1
        self.q[s][a] += self.step_size(s, a) * (r + (0.0 if terminal else self.gamma * nxt) - self.q[s][a])


class LinUCBAgent(Policy):
    """Disjoint LinUCB: one ridge regression per action on the one-hot state
    features, choosing argmax(theta.x + alpha * sqrt(x' A^-1 x)). A^-1 is kept
    up to date with the Sherman-Morrison formula."""
    name = "linucb"
    learns = True

    def __init__(self, alpha: float = 0.6, seed: int = 0):
        d = FEATURE_DIM
        self.alpha = alpha
        self.ainv = [[[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)] for _ in ACTIONS]
        self.b = [[0.0] * d for _ in ACTIONS]
        self.rng = random.Random(seed)

    def _theta(self, a: int) -> List[float]:
        A, b = self.ainv[a], self.b[a]
        return [sum(A[i][j] * b[j] for j in range(len(b))) for i in range(len(b))]

    def score(self, a: int, x: List[float], explore: bool) -> float:
        A = self.ainv[a]
        theta = self._theta(a)
        mean = sum(t * v for t, v in zip(theta, x))
        if not explore:
            return mean
        ax = [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(x))]
        return mean + self.alpha * math.sqrt(max(0.0, sum(v * w for v, w in zip(x, ax))))

    def act(self, s, st, valid, opts, eps=0.0):
        x = feature_vector(st)
        explore = eps > 0
        return max(valid, key=lambda a: (self.score(a, x, explore), -a))

    def update_bandit(self, st: Tuple[int, ...], a: int, r: float) -> None:
        x = feature_vector(st)
        A = self.ainv[a]
        ax = [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(x))]
        denom = 1.0 + sum(v * w for v, w in zip(x, ax))
        for i in range(len(x)):
            for j in range(len(x)):
                A[i][j] -= ax[i] * ax[j] / denom
        for i in range(len(x)):
            self.b[a][i] += r * x[i]

    def to_dict(self):
        return {"name": self.name, "alpha": self.alpha, "ainv": self.ainv, "b": self.b}

    @classmethod
    def from_dict(cls, d):
        a = cls(d["alpha"])
        a.ainv, a.b = d["ainv"], d["b"]
        return a


class KeepPolicy(Policy):
    """Satisfaction only: never deviates from the traveller's favourite plan."""
    name = "keep"

    def act(self, s, st, valid, opts, eps=0.0):
        return KEEP


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, s, st, valid, opts, eps=0.0):
        return self.rng.choice(valid)


class RulePolicy(Policy):
    """A simple hand-written crowd rule: when the stop is crowded, shift it if a
    quieter slot exists, otherwise substitute; else keep."""
    name = "rule"

    def act(self, s, st, valid, opts, eps=0.0):
        crowded = st[1] >= 2
        if crowded and SHIFT in valid:
            return SHIFT
        if crowded and SUBSTITUTE in valid:
            return SUBSTITUTE
        return KEEP


class GreenPolicy(Policy):
    """Environmental impact only: drop the car whenever possible, otherwise
    prefer a substitute that needs less travel."""
    name = "green"

    def act(self, s, st, valid, opts, eps=0.0):
        if SWITCH_MODE in valid:
            return SWITCH_MODE
        if opts is not None and SUBSTITUTE in valid and opts.alt_co2 < opts.leg_co2 - 0.05:
            return SUBSTITUTE
        return KEEP


class NearestPolicy(Policy):
    """Minimum distance only: substitute whenever the alternative is closer."""
    name = "min_distance"

    def act(self, s, st, valid, opts, eps=0.0):
        if opts is not None and SUBSTITUTE in valid and opts.alt_extra < -2:
            return SUBSTITUTE
        return KEEP


FIXED = {"keep": KeepPolicy, "random": RandomPolicy, "rule": RulePolicy, "green": GreenPolicy,
         "min_distance": NearestPolicy}


def save_agent(agent: Policy, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(agent.to_dict(), fh)


def load_agent(path: str) -> Policy:
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    name = d["name"]
    if name == "q_learning":
        return QLearningAgent.from_dict(d)
    if name == "sarsa":
        a = SarsaAgent(d["alpha"], d["gamma"])
        a.q = d["q"]
        return a
    if name == "linucb":
        return LinUCBAgent.from_dict(d)
    return FIXED[name]()
