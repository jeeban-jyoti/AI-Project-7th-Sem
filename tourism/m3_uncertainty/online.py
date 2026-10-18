"""Online Bayesian updating of crowd forecasts from noisy live observations.

For each POI the true arrivals are modelled as Poisson(m * lambda_s), where
lambda_s is the forecast's expected arrivals in slot s and m is an unknown
multiplier (1 = the forecast is right). Sensors only detect a fraction d of
visitors, so an observed count k ~ Poisson(d * m * lambda_s). With a Gamma(a, b)
prior on m the posterior after each observation is conjugate:

    a <- a + k        b <- b + d * lambda_s

Crowds change during the day, so old evidence is discounted before each
update (a power prior / exponential forgetting with factor f): a <- f*a + k,
b <- f*b + d*lambda. With f = 0.8 the posterior effectively remembers the last
~5 slots, which lets it notice a surge within about half an hour.

A posterior that moves well above 1 means an unannounced crowd is forming;
the planner then re-plans the travellers heading there.
"""
from __future__ import annotations

import math
from typing import List, Optional

from tourism.core.city import City
from tourism.m3_uncertainty.forecaster import DayForecast, expected_waits


def _gammainc_lower_reg(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) (Numerical Recipes)."""
    if x <= 0:
        return 0.0
    gln = math.lgamma(a)
    if x < a + 1:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(500):
            ap += 1
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-10:
                break
        return s * math.exp(-x + a * math.log(x) - gln)
    b = x + 1 - a
    c, d = 1e300, 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-10:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - gln) * h


class OnlineCrowdUpdater:
    def __init__(self, city: City, forecast: DayForecast, prior_strength: float = 4.0, detection: float = 0.6,
                 forget: float = 0.8):
        self.city = city
        self.base = forecast
        self.detection = detection
        self.forget = forget
        self.prior = prior_strength
        n = len(city.pois)
        self.alpha = [prior_strength] * n
        self.beta = [prior_strength] * n
        self.observations = [0] * n

    def expected_arrivals(self, i: int, s: int) -> float:
        return self.base.load[i][s] * self.city.pois[i]["capacity"]

    def observe(self, i: int, s: int, count: int, expected: Optional[float] = None) -> None:
        """`expected` overrides the forecast's arrivals (e.g. to include the
        travellers the planner itself routed there)."""
        lam = self.expected_arrivals(i, s) if expected is None else expected
        if lam <= 0.5:
            return
        # discount old evidence, but never below the prior's strength
        f = self.forget
        if self.alpha[i] * f >= 1.0 and self.beta[i] * f >= 1.0:
            self.alpha[i] *= f
            self.beta[i] *= f
        self.alpha[i] += count
        self.beta[i] += self.detection * lam
        self.observations[i] += 1

    def multiplier(self, i: int) -> float:
        return self.alpha[i] / self.beta[i]

    def prob_above(self, i: int, x: float) -> float:
        """Posterior P(m > x)."""
        return 1.0 - _gammainc_lower_reg(self.alpha[i], self.beta[i] * x)

    def surges(self, from_slot: Optional[int] = None, min_multiplier: float = 1.3, min_prob: float = 0.9,
               min_obs: int = 2, min_load: float = 0.9, horizon: int = 8) -> List[int]:
        """POIs whose demand is clearly above forecast. With `from_slot`, only
        those whose projected load in the next `horizon` slots reaches
        `min_load` (a surge at a half-empty place needs no replanning)."""
        out = []
        for i in range(len(self.alpha)):
            if self.observations[i] < min_obs or self.multiplier(i) < min_multiplier or self.prob_above(i, 1.15) < min_prob:
                continue
            if from_slot is not None:
                row = self.base.load[i][from_slot:from_slot + horizon]
                if not row or max(row) * self.multiplier(i) < min_load:
                    continue
            out.append(i)
        return out

    def updated_forecast(self, from_slot: int) -> DayForecast:
        fc = self.base.copy()
        for i, p in enumerate(self.city.pois):
            m = self.multiplier(i)
            if abs(m - 1.0) < 1e-6:
                continue
            cap = p["capacity"]
            for s in range(from_slot, self.city.n_slots):
                mu = fc.load[i][s] * m * cap
                fc.load[i][s] *= m
                # negative-binomial predictive, normal approximation of P(arrivals >= capacity)
                var = mu + mu * mu / max(1.0, self.alpha[i])
                if mu <= 0:
                    fc.pvh[i][s] = 0.0
                else:
                    z = (cap - 0.5 - mu) / math.sqrt(max(var, 1e-9))
                    fc.pvh[i][s] = 0.5 * math.erfc(z / math.sqrt(2))
        fc.wait = expected_waits(self.city, fc.load)
        return fc
