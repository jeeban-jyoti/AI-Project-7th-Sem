"""Shared data types. Every module talks through these (as plain JSON-able
dicts at the engine boundary), which is what lets the GenAI agents call the
same modules as tools later without changes."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def fmt_time(minutes: float) -> str:
    m = int(round(minutes))
    return "%02d:%02d" % (m // 60, m % 60)


@dataclass
class TravellerProfile:
    id: str
    name: str
    interests: Dict[str, float]
    budget: float
    start_node: int
    end_node: int
    start_min: int = 540
    end_min: int = 1170
    max_walk_m: float = 7000.0
    accessible_only: bool = False
    modes: List[str] = field(default_factory=lambda: ["walk", "transit", "car"])
    max_activities: int = 4
    lunch: bool = True
    must_visit: List[int] = field(default_factory=list)
    willingness: Optional[float] = None  # stated willingness (None = unknown to the planner)
    archetype: str = ""
    crowd_threshold: float = 0.85
    co2_target_kg: float = 3.0
    party_size: int = 1

    def to_engine(self, hard_capacity: float = 1.6, lunch_window=(720, 870)) -> Dict[str, Any]:
        return {
            "start_node": self.start_node,
            "end_node": self.end_node,
            "start_min": self.start_min,
            "end_min": self.end_min,
            "budget": self.budget,
            "max_walk_m": self.max_walk_m,
            "accessible_only": self.accessible_only,
            "modes": list(self.modes),
            "max_activities": self.max_activities,
            "lunch": self.lunch,
            "lunch_window": list(lunch_window),
            "crowd_threshold": self.crowd_threshold,
            "co2_target_kg": self.co2_target_kg,
            "hard_capacity": hard_capacity,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TravellerProfile":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ReasonRecord:
    """Why a module changed the plan. The explainer (templates now, an LLM
    later) only phrases these records; it never invents facts."""
    module: str
    change: str
    poi: Optional[int] = None
    poi_name: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)
    why: str = ""
    deltas: Dict[str, float] = field(default_factory=dict)
    accepted: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Proposal:
    """A sustainable alternative put to the traveller (M4 action)."""
    action: str  # shift | substitute | switch_mode | add_local | ask
    poi: Optional[int]
    poi_name: str = ""
    alt_poi: Optional[int] = None
    alt_name: str = ""
    new_start: Optional[int] = None
    interest_match: float = 1.0
    rating_gap: float = 0.0
    popularity_drop: float = 0.0
    extra_minutes: float = 0.0
    wait_saved: float = 0.0
    crowd_relief: float = 0.0
    co2_saved: float = 0.0
    explained: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlanResult:
    traveller_id: str
    strategy: str
    status: str
    itinerary: Dict[str, Any]
    plan: Dict[str, Any]
    reasons: List[ReasonRecord] = field(default_factory=list)
    explanation: str = ""
    backup: Optional[Dict[str, Any]] = None
    risk: Optional[Dict[str, Any]] = None
    proposals: int = 0
    accepted: int = 0
    violations: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)

    def visits(self) -> List[Dict[str, Any]]:
        return [a for a in self.plan.get("actions", []) if a["type"] in ("visit", "eat", "local_experience")]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["reasons"] = [r.to_dict() for r in self.reasons]
        return d
