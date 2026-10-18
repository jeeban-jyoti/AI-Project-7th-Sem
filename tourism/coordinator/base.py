"""The three seams where GenAI plugs in later.

1. ProfileBuilder      build_profile(input) -> TravellerProfile
                        now: FormProfileBuilder (JSON / interactive form)
                        later: the traveller agent turns a chat into the same profile
2. TravellerInterface  respond(proposal) / reveal_willingness()  (the RL `ask` hook)
                        now: SimulatedTraveller or a console prompt
                        later: the traveller agent or a real user in chat
3. Coordinator         plan(profile, day) -> PlanResult, replan(...)
                        now: RuleCoordinator (fixed pipeline + weighted scoring)
                        later: LLMCoordinator negotiating between specialist agents
"""
from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional

from tourism.core.types import PlanResult, Proposal, TravellerProfile


class ProfileBuilder(abc.ABC):
    @abc.abstractmethod
    def build_profile(self, source: Any) -> TravellerProfile:
        ...


class TravellerInterface(abc.ABC):
    @abc.abstractmethod
    def respond(self, proposal: Proposal) -> bool:
        """Accept (True) or reject (False) a suggested alternative."""

    @abc.abstractmethod
    def reveal_willingness(self) -> str:
        """Answer the RL 'ask' action: low / medium / high."""


class Coordinator(abc.ABC):
    @abc.abstractmethod
    def plan(self, profile: TravellerProfile, traveller: Optional[TravellerInterface] = None) -> PlanResult:
        ...

    @abc.abstractmethod
    def replan(self, profile: TravellerProfile, state: Dict[str, Any], remaining: List[Dict[str, Any]],
               trigger: Dict[str, Any], previous: Optional[PlanResult] = None) -> Dict[str, Any]:
        ...


class FormProfileBuilder(ProfileBuilder):
    """Builds a profile from a dict (a JSON file or answers to a console form)."""

    def __init__(self, city):
        self.city = city

    def build_profile(self, source: Dict[str, Any]) -> TravellerProfile:
        d = dict(source)
        d.setdefault("id", "U0001")
        d.setdefault("name", "Traveller")
        if isinstance(d.get("start"), str):
            d["start_node"] = self._node_by_name(d.pop("start"))
        if isinstance(d.get("end"), str):
            d["end_node"] = self._node_by_name(d.pop("end"))
        d.setdefault("end_node", d.get("start_node", self.city.hotels[0]))
        d.setdefault("start_node", self.city.hotels[0])
        for key in ("start_time", "end_time"):
            if isinstance(d.get(key), str):
                hh, mm = d.pop(key).split(":")
                d["start_min" if key == "start_time" else "end_min"] = int(hh) * 60 + int(mm)
        if "must_visit_names" in d:
            d["must_visit"] = [self._node_by_name(n) for n in d.pop("must_visit_names")]
        interests = {t: 0.1 for t in self.city.themes}
        interests.update({k: float(v) for k, v in d.get("interests", {}).items() if k in interests})
        d["interests"] = interests
        return TravellerProfile.from_dict(d)

    def _node_by_name(self, name: str) -> int:
        low = name.strip().lower()
        for n in self.city.nodes:
            if n["name"].lower() == low:
                return n["id"]
        for n in self.city.nodes:
            if low in n["name"].lower():
                return n["id"]
        raise ValueError("unknown place: %s" % name)


class ConsoleTraveller(TravellerInterface):
    """Asks a real person in the terminal (used by `run.sh plan --interactive`)."""

    def respond(self, proposal: Proposal) -> bool:
        from tourism.coordinator.explainer import describe_proposal
        ans = input("\n%s\nAccept? [y/n] " % describe_proposal(proposal)).strip().lower()
        return ans.startswith("y")

    def reveal_willingness(self) -> str:
        ans = input("\nHow open are you to quieter or greener alternatives? [low/medium/high] ").strip().lower()
        return ans if ans in ("low", "medium", "high") else "medium"
