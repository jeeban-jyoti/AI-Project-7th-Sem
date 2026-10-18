"""The six GenAI agents from the brief, as typed stubs.

Each stub states the stand-in it replaces and the tools (tools.py) it will
call. They are intentionally not implemented yet; the rest of the project
runs without them. When implementing, keep the guard-rails:
  * agents decide what to ask and how to phrase it; modules 1-4 compute;
  * every itinerary must pass `validate` before the traveller sees it;
  * explanations may only use numbers present in ReasonRecords / tool results.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from tourism.coordinator.base import Coordinator, ProfileBuilder, TravellerInterface
from tourism.core.types import PlanResult, Proposal, ReasonRecord, TravellerProfile

NOT_YET = "The GenAI layer is added later - see docs/GENAI_INTEGRATION.md"


class LLMClient(Protocol):
    """Whatever model is used later (local or API) must offer chat with tools."""

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        ...


class TravellerAgent(ProfileBuilder, TravellerInterface):
    """Replaces FormProfileBuilder (chat -> TravellerProfile, asking clarifying
    questions, detecting unsolvable requests) and answers the RL `ask` action
    and proposals on behalf of the user in conversation."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def build_profile(self, source: Any) -> TravellerProfile:
        raise NotImplementedError(NOT_YET)

    def respond(self, proposal: Proposal) -> bool:
        raise NotImplementedError(NOT_YET)

    def reveal_willingness(self) -> str:
        raise NotImplementedError(NOT_YET)


class AttractionAgent:
    """Monitors opening hours, capacity and crowd forecasts. Tools: poi_info, crowd_forecast."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def assess(self, itinerary: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError(NOT_YET)


class TransportAgent:
    """Compares travel modes for each leg. Tools: compare_modes, route."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def propose(self, itinerary: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError(NOT_YET)


class SustainabilityAgent:
    """Estimates environmental and social impact of options. Tools: impact, crowd_forecast."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def score(self, options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError(NOT_YET)


class LocalBusinessAgent:
    """Finds restaurants, shops, guides and community experiences outside busy areas. Tools: search_local, poi_info."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def suggest(self, itinerary: Dict[str, Any], profile: TravellerProfile) -> List[Dict[str, Any]]:
        raise NotImplementedError(NOT_YET)


class CoordinatorAgent(Coordinator):
    """Replaces RuleCoordinator and TemplateExplainer: negotiates between the
    specialist agents (max 3 rounds), runs every candidate plan through
    `validate`, and writes the explanation from ReasonRecords only.
    Tools: plan_day, rl_suggest, validate + everything the specialists use."""

    def __init__(self, llm: LLMClient, toolbox, agents: Dict[str, Any]):
        self.llm = llm
        self.toolbox = toolbox
        self.agents = agents

    def plan(self, profile: TravellerProfile, traveller: Optional[TravellerInterface] = None) -> PlanResult:
        raise NotImplementedError(NOT_YET)

    def replan(self, profile: TravellerProfile, state: Dict[str, Any], remaining: List[Dict[str, Any]],
               trigger: Dict[str, Any], previous: Optional[PlanResult] = None) -> Dict[str, Any]:
        raise NotImplementedError(NOT_YET)

    def explain(self, plan: Dict[str, Any], reasons: List[ReasonRecord]) -> str:
        raise NotImplementedError(NOT_YET)
