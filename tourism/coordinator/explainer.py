"""Template explainer (stand-in for the GenAI coordinator's explanations).

It only phrases ReasonRecords produced by modules 1-4; every number it prints
comes from those records. Sustainable changes are worded as options with
their trade-offs, never as restrictions."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from tourism.core.city import City
from tourism.core.types import Proposal, ReasonRecord, fmt_time


def crowd_word(rho: float) -> str:
    if rho < 0.4:
        return "quiet"
    if rho < 0.7:
        return "moderately busy"
    if rho < 1.0:
        return "busy"
    return "very crowded"


def crowd_change(before: float, after: float) -> str:
    if crowd_word(before) != crowd_word(after):
        return "%s instead of %s" % (crowd_word(after), crowd_word(before))
    return "expected load falls from %d%% to %d%% of capacity" % (round(100 * before), round(100 * after))


def describe_proposal(p: Proposal) -> str:
    """One-line description of a suggestion (used when asking the traveller)."""
    if p.action == "shift":
        return ("Visit %s at %s instead: about %d%% less crowded, saving ~%d min of queueing."
                % (p.poi_name, fmt_time(p.new_start or 0), round(100 * max(0.0, p.crowd_relief)), max(0, round(p.wait_saved))))
    if p.action == "substitute":
        return ("Try %s instead of %s: a %d%% interest match, much quieter (crowd -%d%%), %s."
                % (p.alt_name, p.poi_name, round(100 * min(1.0, p.interest_match)), round(100 * max(0.0, p.crowd_relief)),
                   _travel_phrase(p.extra_minutes)))
    if p.action == "switch_mode":
        return ("Reach %s by public transport / on foot: %s and %.2f kg less CO2."
                % (p.poi_name, _travel_phrase(p.extra_minutes), max(0.0, p.co2_saved)))
    if p.action == "add_local":
        return "Add %s, a local business near %s (+%d min)." % (p.alt_name, p.poi_name, round(p.extra_minutes))
    return p.action


def _travel_phrase(extra: float) -> str:
    if extra <= -2:
        return "%d min less travel" % round(-extra)
    if extra < 2:
        return "about the same travel time"
    return "%d min more travel" % round(extra)


class TemplateExplainer:
    def __init__(self, city: City):
        self.city = city

    def explain(self, plan: Dict[str, Any], reasons: List[ReasonRecord], risk: Optional[Dict[str, Any]] = None,
                backup: Optional[Dict[str, Any]] = None) -> str:
        city = self.city
        lines: List[str] = []
        tot = plan.get("totals", {})
        visits = [a for a in plan.get("actions", []) if a["type"] in ("visit", "eat", "local_experience")]
        modes = set()
        for a in plan.get("actions", []):
            for s in a.get("segments", []):
                modes.add(s["mode"])
        lines.append("Your day: %d stops, about %d min of travel (%s), %d min of expected queueing, %.2f kg CO2, cost %d."
                     % (len(visits), round(tot.get("travel_min", 0)), ", ".join(sorted(modes)) or "on foot",
                        round(tot.get("wait_min", 0)), tot.get("co2_kg", 0.0), round(tot.get("money", 0))))
        suggestions = [r for r in reasons if r.module == "M4" and r.change not in ("ask", "review")]
        if suggestions:
            lines.append("")
            lines.append("Options we suggested (you stay in control):")
        for r in reasons:
            text = self._reason(r)
            if text:
                lines.append("- " + text)
        quiet = [a for a in plan.get("actions", []) if a["type"] == "shift_visit"]
        for a in quiet:
            lines.append("- The plan enters %s at %s rather than %s: %s."
                         % (city.name_of(a["poi"]), fmt_time(a["end"]), fmt_time(a["start"]),
                            crowd_change(a.get("rho_before", 0), a.get("rho_after", 0))))
        if risk is not None:
            lines.append("")
            lines.append("Uncertainty: %d%% chance that something in this plan goes wrong (crowds, delays); "
                         "emissions %.2f kg CO2 (90%% range %.2f-%.2f)."
                         % (round(100 * risk["p_fail"]), risk["co2_kg"]["mean"], risk["co2_kg"]["p5"], risk["co2_kg"]["p95"]))
        if backup is not None:
            names = [a.get("name", city.name_of(a["poi"])) for a in backup.get("actions", []) if a["type"] in ("visit", "local_experience")]
            lines.append("Plan B is ready if %s gets too busy: %s." % (backup.get("instead_of", "a stop"), ", ".join(names)))
        return "\n".join(lines)

    def _reason(self, r: ReasonRecord) -> str:
        d, dl = r.detail, r.deltas
        if r.module == "M4":
            if r.change == "review":
                if d.get("suggested"):
                    return ""
                return ("We checked all %d stops for a quieter time, a similar less-crowded place, a greener way to get "
                        "there and nearby local businesses. Your plan already avoids the crowds (at most %d%% of "
                        "capacity expected), so nothing needed changing." % (d["stops"], round(100 * d["max_rho"])))
            if r.change == "ask":
                return "You told us you are %s to quieter or greener alternatives." % (
                    {"low": "not very open", "medium": "somewhat open", "high": "open"}.get(d.get("answer", ""), "open"))
            prefix = "" if r.accepted else "(you kept the original) "
            if r.change == "shift":
                saved = max(0, round(-dl.get("wait_min", 0)))
                return prefix + ("%s at %s rather than %s: %s%s."
                                 % (r.poi_name, fmt_time(d["new_start"]), fmt_time(d["old_start"]),
                                    crowd_change(d["rho_old"], d["rho_new"]),
                                    ", saving about %d min of queueing" % saved if saved >= 1 else ""))
            if r.change == "substitute":
                extra = "; it supports a less-visited neighbourhood (%s)" % d["zone"] if d.get("less_visited") else ""
                return prefix + ("%s instead of %s: it matches your interests (%d%% as well), is expected to be %s rather "
                                 "than %s, and needs %s%s."
                                 % (d["alt_name"], r.poi_name, round(100 * min(1.0, d["match"])), crowd_word(d["rho_new"]),
                                    crowd_word(d["rho_old"]), _travel_phrase(dl.get("travel_min", 0)), extra))
            if r.change == "switch_mode":
                return prefix + ("Getting to %s by public transport or on foot: %s and %.2f kg less CO2 than a taxi."
                                 % (r.poi_name, _travel_phrase(dl.get("travel_min", 0)), max(0.0, -dl.get("co2_kg", 0))))
            if r.change == "add_local":
                return prefix + ("%s, an independent %s near %s, fits your interests and keeps your spending in the "
                                 "local community (+%d min)." % (d["alt_name"], d["kind"], r.poi_name, round(dl.get("travel_min", 0))))
        if r.module == "M1" and r.change == "unsolvable":
            return "Not everything you asked for fits together: %s." % "; ".join(c.get("detail", "") + (" (%s)" % c["name"] if c.get("name") else "") for c in d.get("conflicts", []))
        if r.module == "M1" and r.change == "relax":
            return "To make the day work we relaxed: %s." % ", ".join(d.get("relaxed", []))
        if r.module == "M1" and r.change == "soft_violation":
            return "%s is likely to be %s at your visit time; it is kept because it is one of your top picks." % (
                r.poi_name, crowd_word(d.get("rho", 1.0)))
        if r.module == "M2" and r.change == "replan":
            return "Plan updated at %s because %s (%s)." % (fmt_time(d.get("time", 0)), d.get("trigger", "conditions changed"),
                                                             d.get("strategy", "replanned"))
        if r.module == "M3" and r.change == "contingency":
            return "%s has a %d%% risk of being over capacity or too crowded, so a plan B is kept ready." % (
                r.poi_name, round(100 * d.get("risk", 0)))
        if r.module == "M1" and r.change == "reverted":
            return "We could not fit %s together with the rest of your day, so it stays as before." % r.poi_name
        return ""
