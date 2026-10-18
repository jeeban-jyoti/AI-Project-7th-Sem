"""Independent validator: re-checks every hard constraint of a plan in Python.

It does not trust the C++ solver: any plan from any coordinator (rule-based
now, an LLM later) must pass here before it reaches the traveller."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from tourism.core.city import City
from tourism.core.types import TravellerProfile, fmt_time

VISITS = ("visit", "eat", "local_experience")
TOL = 1.5  # minutes / metres / money tolerance for rounding


def validate_plan(city: City, profile: TravellerProfile, plan: Dict[str, Any], conditions: Optional[Dict[str, Any]] = None,
                  lunch_window=(720, 870), start_state: Optional[Dict[str, Any]] = None) -> List[str]:
    cond = conditions or {}
    closed = set(cond.get("closed_pois", []))
    actions = plan.get("actions", [])
    out: List[str] = []
    st = start_state or {}
    t = st.get("time", profile.start_min)
    node = st.get("node", profile.start_node)
    money = profile.budget - st.get("budget", profile.budget)
    walked = st.get("walked_m", 0.0)
    visited = set(st.get("visited", []))
    for a in actions:
        if a["start"] < t - TOL:
            out.append("%s at %s starts before the previous step ends" % (a["type"], fmt_time(a["start"])))
        t = max(t, a["end"])
        if a["type"] == "travel":
            if a["from"] != node:
                out.append("travel from %s but traveller is at %s" % (city.name_of(a["from"]), city.name_of(node)))
            node = a["to"]
            money += a.get("money", 0.0)
            walked += a.get("walk_m", 0.0)
            modes = {s["mode"] for s in a.get("segments", [])}
            if "car" in modes and "car" not in profile.modes:
                out.append("uses a car although the traveller does not want one")
        elif a["type"] in VISITS or a["type"] == "buy_ticket":
            poi = city.poi_by_node[a["poi"]]
            if node != a["poi"]:
                out.append("%s at %s but traveller is elsewhere" % (a["type"], poi["name"]))
            money += a.get("money", 0.0)
            if a["type"] in VISITS:
                if a["start"] < poi["open_min"] - TOL or a["end"] > poi["close_min"] + TOL:
                    out.append("%s outside opening hours (%s-%s)" % (poi["name"], fmt_time(poi["open_min"]), fmt_time(poi["close_min"])))
                if a["poi"] in closed:
                    out.append("%s is closed" % poi["name"])
                if profile.accessible_only and not poi["accessible"]:
                    out.append("%s is not wheelchair accessible" % poi["name"])
                if a["type"] == "eat" and profile.lunch and not (lunch_window[0] - 30 - TOL <= a["start"] <= lunch_window[1] + 30 + TOL):
                    out.append("lunch at %s is outside the lunch window" % fmt_time(a["start"]))
                if a["poi"] in visited:
                    out.append("%s visited twice" % poi["name"])
                visited.add(a["poi"])
    if node != profile.end_node:
        out.append("plan does not end at %s" % city.name_of(profile.end_node))
    if t > profile.end_min + TOL:
        out.append("plan ends at %s, after %s" % (fmt_time(t), fmt_time(profile.end_min)))
    if money > profile.budget + TOL:
        out.append("cost %.0f exceeds the budget %.0f" % (money, profile.budget))
    if walked > profile.max_walk_m + TOL:
        out.append("walking %.1f km exceeds the limit %.1f km" % (walked / 1000, profile.max_walk_m / 1000))
    for m in profile.must_visit:
        if m not in visited and m not in closed:
            out.append("must-visit %s is missing" % city.name_of(m))
    return out
