"""Simulated travellers: how a person reacts to a sustainable suggestion.

The planner never sees `HiddenTraits`; it can only ask (the RL `ask` action
reveals a coarse willingness level) and observe accept / reject. A real user
(or the GenAI traveller agent, later) implements the same interface.
"""
from __future__ import annotations

import math
import random

from tourism.coordinator.base import TravellerInterface
from tourism.core.types import Proposal, TravellerProfile
from tourism.data.personas import HiddenTraits, willingness_level


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def acceptance_probability(traits: HiddenTraits, p: Proposal, explanation_boost: float = 0.35) -> float:
    x = -0.4 + 3.2 * (traits.willingness - 0.5)
    relief = min(1.0, p.wait_saved / 30.0) + 0.8 * min(1.0, max(0.0, p.crowd_relief))
    if p.action == "shift":
        x += 0.5 + 0.8 * relief - traits.time_sensitivity * p.extra_minutes / 45.0
    elif p.action == "substitute":
        x += (-0.3 + 2.5 * (p.interest_match - 0.8) - traits.quality_sensitivity * max(0.0, p.rating_gap) * 1.5
              - traits.fame_attachment * p.popularity_drop * 1.8 + 0.8 * relief
              - traits.time_sensitivity * max(0.0, p.extra_minutes) / 40.0)
    elif p.action == "switch_mode":
        x += 0.4 - traits.time_sensitivity * max(0.0, p.extra_minutes) / 25.0 - 1.4 * traits.car_preference
    elif p.action == "add_local":
        x += 0.3 + 1.5 * (p.interest_match - 0.6) - traits.time_sensitivity * max(0.0, p.extra_minutes) / 60.0
    if p.explained:
        x += explanation_boost
    return _sigmoid(x)


class SimulatedTraveller(TravellerInterface):
    def __init__(self, profile: TravellerProfile, traits: HiddenTraits, rng: random.Random,
                 explanation_boost: float = 0.35):
        self.profile = profile
        self.traits = traits
        self.rng = rng
        self.boost = explanation_boost
        self.asked = 0
        self.history = []

    def respond(self, proposal: Proposal) -> bool:
        ok = self.rng.random() < acceptance_probability(self.traits, proposal, self.boost)
        self.history.append((proposal.action, ok))
        return ok

    def reveal_willingness(self) -> str:
        self.asked += 1
        return willingness_level(self.traits.willingness)
