"""Builds outputs/report/report.html: a self-contained page (inline SVG charts,
no external libraries) with the evaluation, RL, forecasting and search results."""
from __future__ import annotations

import html
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tourism.core.config import ensure_dir, load_config, path
from tourism.eval.experiments import load_rows, summarize
from tourism.sim.scenarios import SCENARIOS

PALETTE = ["#2f6db3", "#d1495b", "#3c9d5d", "#e09f3e", "#7a5195", "#5a5a5a", "#1b998b"]
SYSTEM_ORDER = ["conventional", "min_distance", "env_only", "sustainable", "sustainable_no_m3", "sustainable_no_rl",
                "sustainable_no_replan"]
SYSTEM_LABEL = {
    "conventional": "Conventional",
    "min_distance": "Min. distance",
    "env_only": "Env. only",
    "sustainable": "Sustainable (full)",
    "sustainable_no_m3": "No Bayesian forecast",
    "sustainable_no_rl": "Rule instead of RL",
    "sustainable_no_replan": "No replanning",
}


def series_color(name: str, index: int) -> str:
    """Each system keeps the same colour in every chart."""
    for k, sy in enumerate(SYSTEM_ORDER):
        if SYSTEM_LABEL.get(sy) == name:
            return PALETTE[k % len(PALETTE)]
    return PALETTE[index % len(PALETTE)]


def esc(s: Any) -> str:
    return html.escape(str(s))


def fmt(v: float, digits: int = 2) -> str:
    if abs(v) >= 100:
        return "%.0f" % v
    return ("%%.%df" % digits) % v


# ------------------------------------------------------------------ SVG charts
def nice_ticks(lo: float, hi: float, n: int = 4) -> List[float]:
    """Round tick values (1, 2, 2.5, 5 x 10^k) covering [lo, hi]."""
    import math
    span = max(hi - lo, 1e-9)
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v < hi + step * 0.999:
        ticks.append(round(v, 10))
        v += step
    return ticks


def tick_label(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return "%d" % round(v)
    return ("%.2f" % v).rstrip("0").rstrip(".")

def bar_chart(title: str, categories: List[str], series: Dict[str, List[float]], unit: str = "",
              width: int = 900, height: int = 300) -> str:
    left, right, top, bottom = 56, 16, 40, 64
    w, h = width - left - right, height - top - bottom
    ticks = nice_ticks(0.0, max([v for vals in series.values() for v in vals] + [1e-9]))
    vmax = ticks[-1]
    n_cat, n_ser = len(categories), len(series)
    group = w / max(1, n_cat)
    bw = min(26.0, group * 0.8 / max(1, n_ser))
    out = ['<svg viewBox="0 0 %d %d" class="chart" role="img" aria-label="%s">' % (width, height, esc(title)),
           '<text x="%d" y="22" class="ct">%s</text>' % (left, esc(title))]
    for tv in ticks:
        y = top + h - h * tv / vmax
        out.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" class="grid"/>' % (left, left + w, y, y))
        out.append('<text x="%d" y="%.1f" class="tick" text-anchor="end">%s</text>' % (left - 6, y + 4, tick_label(tv)))
    for ci, cat in enumerate(categories):
        gx = left + group * ci + (group - bw * n_ser) / 2
        for si, (name, vals) in enumerate(series.items()):
            v = vals[ci]
            bh = h * v / vmax
            x = gx + si * bw
            out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s"><title>%s - %s: %s%s</title></rect>'
                       % (x, top + h - bh, bw - 2, bh, series_color(name, si), esc(name), esc(cat), fmt(v), esc(unit)))
        out.append('<text x="%.1f" y="%d" class="tick" text-anchor="middle">%s</text>'
                   % (left + group * ci + group / 2, top + h + 16, esc(cat)))
    lx = left
    for si, name in enumerate(series):
        out.append('<rect x="%d" y="%d" width="10" height="10" fill="%s"/>' % (lx, height - 22, series_color(name, si)))
        out.append('<text x="%d" y="%d" class="tick">%s</text>' % (lx + 14, height - 13, esc(name)))
        lx += 22 + 7 * len(name)
    out.append("</svg>")
    return "\n".join(out)


def line_chart(title: str, series: Dict[str, List[float]], xlabel: str, width: int = 900, height: int = 280) -> str:
    left, right, top, bottom = 56, 16, 40, 56
    w, h = width - left - right, height - top - bottom
    vals = [v for s in series.values() for v in s]
    ticks = nice_ticks(min(vals + [0.0]), max(vals + [1e-9]))
    vmin, vmax = ticks[0], ticks[-1]
    n = max(len(s) for s in series.values())
    out = ['<svg viewBox="0 0 %d %d" class="chart" role="img" aria-label="%s">' % (width, height, esc(title)),
           '<text x="%d" y="22" class="ct">%s</text>' % (left, esc(title))]
    for v in ticks:
        y = top + h - h * (v - vmin) / (vmax - vmin)
        out.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" class="%s"/>' % (left, left + w, y, y, "axis0" if abs(v) < 1e-12 else "grid"))
        out.append('<text x="%d" y="%.1f" class="tick" text-anchor="end">%s</text>' % (left - 6, y + 4, tick_label(v)))
    for si, (name, s) in enumerate(series.items()):
        pts = " ".join("%.1f,%.1f" % (left + w * i / max(1, n - 1), top + h - h * (v - vmin) / (vmax - vmin))
                       for i, v in enumerate(s))
        out.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2"/>' % (pts, PALETTE[si % len(PALETTE)]))
    out.append('<text x="%d" y="%d" class="tick" text-anchor="middle">%s</text>' % (left + w / 2, top + h + 18, esc(xlabel)))
    lx = left
    for si, name in enumerate(series):
        out.append('<rect x="%d" y="%d" width="10" height="10" fill="%s"/>' % (lx, height - 20, PALETTE[si % len(PALETTE)]))
        out.append('<text x="%d" y="%d" class="tick">%s</text>' % (lx + 14, height - 11, esc(name)))
        lx += 22 + 7 * len(name)
    out.append("</svg>")
    return "\n".join(out)


def scatter_chart(title: str, points: List[Tuple[float, float, str, int]], xlabel: str, ylabel: str,
                  legend: List[Tuple[int, str]], width: int = 900, height: int = 340) -> str:
    left, right, top, bottom = 64, 24, 40, 70
    w, h = width - left - right, height - top - bottom
    xs = [p[0] for p in points] + [0.0]
    ys = [p[1] for p in points] + [0.0]
    xt = nice_ticks(min(xs), max(xs), 5)
    yt = nice_ticks(min(ys), max(ys), 4)
    xmin, xmax, ymin, ymax = xt[0], xt[-1], yt[0], yt[-1]

    def X(v):
        return left + w * (v - xmin) / (xmax - xmin)

    def Y(v):
        return top + h - h * (v - ymin) / (ymax - ymin)

    out = ['<svg viewBox="0 0 %d %d" class="chart" role="img" aria-label="%s">' % (width, height, esc(title)),
           '<text x="%d" y="22" class="ct">%s</text>' % (left, esc(title)),
           '<line x1="%.1f" x2="%.1f" y1="%d" y2="%d" class="axis0"/>' % (X(0), X(0), top, top + h),
           '<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" class="axis0"/>' % (left, left + w, Y(0), Y(0))]
    for xv in xt:
        out.append('<line x1="%.1f" x2="%.1f" y1="%d" y2="%d" class="grid"/>' % (X(xv), X(xv), top, top + h))
        out.append('<text x="%.1f" y="%d" class="tick" text-anchor="middle">%s</text>' % (X(xv), top + h + 16, tick_label(xv)))
    for yv in yt:
        out.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" class="grid"/>' % (left, left + w, Y(yv), Y(yv)))
        out.append('<text x="%d" y="%.1f" class="tick" text-anchor="end">%s</text>' % (left - 6, Y(yv) + 4, tick_label(yv)))
    for x, y, label, si in points:
        out.append('<circle cx="%.1f" cy="%.1f" r="5" fill="%s"><title>%s: %s, %s</title></circle>'
                   % (X(x), Y(y), PALETTE[si % len(PALETTE)], esc(label), fmt(x, 1), fmt(y, 1)))
        out.append('<text x="%.1f" y="%.1f" class="tiny">%s</text>' % (X(x) + 7, Y(y) - 6, esc(label.split(" ")[0])))
    out.append('<text x="%.1f" y="%d" class="tick" text-anchor="middle">%s</text>' % (left + w / 2, top + h + 34, esc(xlabel)))
    out.append('<text x="14" y="%.1f" class="tick" transform="rotate(-90 14 %.1f)" text-anchor="middle">%s</text>'
               % (top + h / 2, top + h / 2, esc(ylabel)))
    lx = left
    for si, name in legend:
        out.append('<circle cx="%d" cy="%d" r="5" fill="%s"/>' % (lx + 5, height - 14, PALETTE[si % len(PALETTE)]))
        out.append('<text x="%d" y="%d" class="tick">%s</text>' % (lx + 14, height - 10, esc(name)))
        lx += 26 + 7 * len(name)
    out.append("</svg>")
    return "\n".join(out)


def table(headers: Sequence[str], rows: List[Sequence[Any]]) -> str:
    out = ["<table><thead><tr>" + "".join("<th>%s</th>" % esc(h) for h in headers) + "</tr></thead><tbody>"]
    for r in rows:
        out.append("<tr>" + "".join("<td>%s</td>" % (c if isinstance(c, str) and c.startswith("<") else esc(c)) for c in r) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


# ------------------------------------------------------------------ report
def pct_change(new: float, old: float) -> float:
    return 100.0 * (new - old) / old if abs(old) > 1e-9 else 0.0


def build_report(mode: str = "quick") -> str:
    cfg = load_config()
    res_dir = path(cfg, "outputs_dir", "results")
    rep_dir = path(cfg, "outputs_dir", "reports")
    out_dir = ensure_dir(path(cfg, "outputs_dir", "report"))
    csv_path = os.path.join(res_dir, "results_%s.csv" % mode)
    rows = load_rows(csv_path) if os.path.exists(csv_path) else []
    groups = summarize(rows)["groups"] if rows else []
    ev = cfg["evaluation"][mode]
    G = {(g["scenario"], g["system"], g["willingness"]): g for g in groups}
    will = ev["willingness"]
    scen = [s for s in ev["scenarios"] if any((s, sy, will) in G for sy in ev["systems"])]
    systems = [s for s in ev["systems"] if any((sc, s, will) in G for sc in scen)]

    parts: List[str] = []
    parts.append("<h1>Sustainable and Crowd-Aware Tourism Planner - results</h1>")
    parts.append("<p class='lead'>Each simulated day plans %d travellers (each representing a party of %d visitors) "
                 "with every system, on the same background crowds, events and sensor noise, then simulates the day "
                 "with queues, delays and replanning. Seeds: %s. Mode: %s.</p>"
                 % (ev["travellers"], cfg["simulation"]["traveller_weight"], ", ".join(map(str, ev["seeds"])), esc(mode)))

    if groups and "conventional" in systems and "sustainable" in systems:
        head = []
        for s in scen:
            c, f = G.get((s, "conventional", will)), G.get((s, "sustainable", will))
            if not c or not f:
                continue
            head.append([SCENARIOS[s]["label"], "%+.1f%%" % pct_change(f["satisfaction"], c["satisfaction"]),
                         "%s -> %s" % (fmt(c["avg_wait_min"], 1), fmt(f["avg_wait_min"], 1)),
                         "%s -> %s" % (fmt(100 * c["overloaded_visit_share"], 1), fmt(100 * f["overloaded_visit_share"], 1)),
                         "%s -> %s" % (fmt(c["peak_load"]), fmt(f["peak_load"])),
                         "%+.0f%%" % pct_change(f["co2_kg"], c["co2_kg"]),
                         "%s -> %s" % (fmt(c["local_visits"]), fmt(f["local_visits"])),
                         "%s -> %s" % (fmt(100 * c["feasibility"], 0), fmt(100 * f["feasibility"], 0)),
                         fmt(100 * f["acceptance_rate"], 0) + "%"])
        parts.append("<h2>Headline: conventional vs. sustainable planner</h2>")
        parts.append(table(["Scenario", "Satisfaction", "Queue min / traveller", "Overcrowded visits %", "Peak load",
                            "CO2", "Local businesses", "Feasible %", "Suggestions accepted"], head))
        sat_points = []
        for si, sy in enumerate([s for s in systems if s != "conventional"]):
            for s in scen:
                c, f = G.get((s, "conventional", will)), G.get((s, sy, will))
                if c and f:
                    sat_points.append((pct_change(f["satisfaction"], c["satisfaction"]),
                                       100 * (c["overloaded_visit_share"] - f["overloaded_visit_share"]),
                                       "%s %s" % (s.split("_")[0], SYSTEM_LABEL.get(sy, sy)), si + 1))
        others = [s for s in systems if s != "conventional"]
        parts.append(scatter_chart("Trade-off vs. the conventional planner: satisfaction change and fewer overcrowded visits",
                                   sat_points, "satisfaction change vs conventional (%)",
                                   "fewer overcrowded visits (percentage points)",
                                   [(i + 1, SYSTEM_LABEL.get(s, s)) for i, s in enumerate(others)]))
        parts.append("<p class='note'>Each point is one scenario. Top right means more satisfied travellers and "
                     "fewer overcrowded visits than the conventional planner.</p>")
        cats = [s.split("_")[0] for s in scen]

        def ser(metric: str, scale: float = 1.0) -> Dict[str, List[float]]:
            return {SYSTEM_LABEL.get(sy, sy): [scale * G[(s, sy, will)][metric] if (s, sy, will) in G else 0.0 for s in scen]
                    for sy in systems}
        parts.append(bar_chart("Average queueing time per traveller (min)", cats, ser("avg_wait_min"), " min"))
        parts.append(bar_chart("Visits made while the place was over capacity (%)", cats, ser("overloaded_visit_share", 100), "%"))
        parts.append(bar_chart("CO2 per traveller (kg)", cats, ser("co2_kg"), " kg"))
        parts.append(bar_chart("Realized satisfaction per traveller", cats, ser("satisfaction")))
        parts.append("<p class='note'>Scenario codes: " + "; ".join("%s = %s" % (s.split("_")[0], SCENARIOS[s]["label"]) for s in scen) + ".</p>")

        parts.append("<h2>All metrics by scenario</h2>")
        cols = [("feasibility", "Feasible", 100), ("satisfaction", "Satisfaction", 1), ("preferred_activities", "Preferred done", 1),
                ("activities", "Activities", 1), ("avg_wait_min", "Wait min", 1), ("crowd_exposure", "Crowd exposure", 1),
                ("overloaded_visit_share", "Overcrowded %", 100), ("peak_load", "Peak load", 1), ("gini_load", "Load Gini", 1),
                ("top5_share", "Top-5 share", 100), ("co2_kg", "CO2 kg", 1), ("green_share", "Walk+transit %", 100),
                ("travel_min", "Travel min", 1), ("local_visits", "Local visits", 1), ("acceptance_rate", "Accepted %", 100),
                ("replans", "Replans", 1), ("failures", "Failed stops", 1)]
        for s in scen:
            rows_t = []
            for sy in systems:
                g = G.get((s, sy, will))
                if g:
                    rows_t.append([SYSTEM_LABEL.get(sy, sy)] + [fmt(g[m] * k, 2) for m, _, k in cols])
            parts.append("<details><summary>%s</summary>%s</details>" % (esc(SCENARIOS[s]["label"]),
                                                                       table(["System"] + [c[1] for c in cols], rows_t)))

    sw = ev["willingness_sweep"]
    sweep = [(lvl, sy, G.get((sw["scenario"], sy, lvl))) for lvl in sw["levels"] for sy in sw["systems"]]
    if any(g for _, _, g in sweep):
        parts.append("<h2>Travellers with different willingness to accept alternatives (%s)</h2>" % esc(SCENARIOS[sw["scenario"]]["label"]))
        rows_t = [[lvl, SYSTEM_LABEL.get(sy, sy), fmt(g["satisfaction"]), fmt(g["avg_wait_min"], 1),
                   fmt(100 * g["overloaded_visit_share"], 1), fmt(g["co2_kg"]), fmt(100 * g["acceptance_rate"], 0),
                   fmt(g["accepted_alternatives"])] for lvl, sy, g in sweep if g]
        parts.append(table(["Willingness", "System", "Satisfaction", "Wait min", "Overcrowded %", "CO2 kg", "Accepted %",
                            "Accepted per traveller"], rows_t))
        if all(G.get((sw["scenario"], sy, l)) for l in sw["levels"] for sy in sw["systems"]):
            parts.append(bar_chart("Overcrowded visits (%) by willingness to accept alternatives", sw["levels"],
                                   {SYSTEM_LABEL.get(sy, sy): [100 * G[(sw["scenario"], sy, l)]["overloaded_visit_share"]
                                                               for l in sw["levels"]] for sy in sw["systems"]}, "%"))
            parts.append(bar_chart("Suggestions accepted (%) by the sustainable planner's travellers", sw["levels"],
                                   {"Sustainable (full)": [100 * G[(sw["scenario"], "sustainable", l)]["acceptance_rate"]
                                                           for l in sw["levels"]]}, "%", height=240))

    ab = ev["ablation"]
    ab_rows = []
    for sc in ab.get("scenarios", [ab.get("scenario")]):
        for sy in ["sustainable"] + ab["systems"]:
            g = G.get((sc, sy, will))
            if g:
                ab_rows.append([sc.split("_")[0], SYSTEM_LABEL.get(sy, sy), fmt(g["satisfaction"]), fmt(g["avg_wait_min"], 1),
                                fmt(100 * g["overloaded_visit_share"], 1), fmt(g["peak_load"]), fmt(g["co2_kg"]),
                                fmt(100 * g["feasibility"], 0), fmt(g["failures"]), fmt(g["replans"])])
    if ab_rows:
        parts.append("<h2>Ablations: what each module contributes</h2>")
        parts.append("<p>Each row removes one part of the full system: the Bayesian forecast (replaced by historical "
                     "averages), the learned RL policy (replaced by a simple crowd rule) or replanning.</p>")
        parts.append(table(["Scenario", "Variant", "Satisfaction", "Wait min", "Overcrowded %", "Peak load", "CO2 kg",
                            "Feasible %", "Failed stops", "Replans / traveller"], ab_rows))

    rl_path = os.path.join(rep_dir, "rl_results.json")
    if os.path.exists(rl_path):
        with open(rl_path, "r", encoding="utf-8") as fh:
            rl = json.load(fh)
        parts.append("<h2>Module 4: reinforcement learning</h2>")
        parts.append("<p>Reward per decision is the improvement over keeping the stop as planned (0 = keep everything). "
                     "Trained for %d simulated days; evaluated greedily on %d unseen days.</p>" % (rl["episodes"], rl["eval_episodes"]))
        pretty = {"q_learning": "Q-learning", "sarsa": "SARSA", "linucb": "LinUCB bandit"}
        parts.append(line_chart("Learning curves (reward per decision, moving average of %d days)" % rl["curve_window"],
                                {pretty.get(k, k): v for k, v in rl["curves"].items()}, "training days"))
        parts.append("<p class='note'>The full system uses the <b>%s</b> policy (config: strategies.sustainable.policy); "
                     "any of the three can be swapped in.</p>" % esc(pretty.get(cfg["strategies"]["sustainable"]["policy"],
                                                                                cfg["strategies"]["sustainable"]["policy"])))
        names = {"q_learning": "Q-learning", "sarsa": "SARSA", "linucb": "LinUCB bandit", "keep": "Satisfaction only (keep)",
                 "random": "Random", "rule": "Simple crowd rule", "green": "Environment only", "min_distance": "Minimum distance only"}
        rows_t = [[names.get(k, k), fmt(v["reward_per_decision"], 3), fmt(v["utility_per_stop"]), fmt(v["wait_per_stop_min"]),
                   fmt(v["crowd_exposure"]), fmt(100 * v["overloaded_visits"], 1), fmt(v["co2_kg_per_traveller"]),
                   fmt(v["local_per_traveller"]), fmt(100 * v["acceptance_rate"], 0), fmt(v["asks_per_traveller"])]
                  for k, v in sorted(rl["evaluation"].items(), key=lambda kv: -kv[1]["reward_per_decision"])]
        parts.append(table(["Policy", "Reward / decision", "Utility / stop", "Wait / stop", "Crowd exposure",
                            "Overcrowded %", "CO2 / traveller", "Local / traveller", "Accepted %", "Asks / traveller"], rows_t))

    fpath = os.path.join(rep_dir, "forecast_eval.json")
    if os.path.exists(fpath):
        with open(fpath, "r", encoding="utf-8") as fh:
            fe = json.load(fh)
        parts.append("<h2>Module 3: forecasting crowd levels on held-out days</h2>")
        rows_t = [[k, fmt(v["mae_load"], 4), fmt(v["brier"], 4), fmt(100 * v["accuracy"], 1), int(v["samples"])]
                  for k, v in fe["models"].items()]
        parts.append(table(["Model", "MAE of load ratio", "Brier score (crowd level)", "Accuracy %", "Samples"], rows_t))
        if "online" in fe:
            o = fe["online"]
            parts.append("<p>Online Gamma-Poisson update (with forgetting) on a day with an unannounced crowd at %s "
                         "starting %s: detected at %s; demand multiplier estimated at %.2f on detection and up to %.2f "
                         "during the surge (true factor %.1f).</p>"
                         % (esc(o.get("poi", "")), esc(o.get("surge_starts", "")), esc(o.get("detected_at", "-")),
                            o.get("multiplier", 0), o.get("multiplier_during_surge", 0), o.get("true_factor", 0)))

    bpath = os.path.join(rep_dir, "search_bench.json")
    if os.path.exists(bpath):
        with open(bpath, "r", encoding="utf-8") as fh:
            sb = json.load(fh)
        parts.append("<h2>Module 1: search and CSP</h2>")
        rows_t = [[k.upper(), fmt(v["avg_expanded"], 1), fmt(v["avg_runtime_us"], 1), fmt(v["avg_cost_ratio_vs_ucs"], 3),
                   fmt(100 * v["optimal_share"], 0)] for k, v in sb["routing"]["algorithms"].items()]
        parts.append(table(["Algorithm", "Nodes expanded", "Runtime (us)", "Cost vs UCS", "Optimal %"], rows_t))
        if "csp" in sb:
            c = sb["csp"]
            parts.append(table(["CSP solves", "Solved %", "Avg nodes", "Avg backtracks", "Avg pruned (B&B)",
                                "Avg forward-check removals", "Avg runtime ms", "Complete search %", "Greedy = final %"],
                               [[c["solves"], fmt(100 * c["solved"], 0), fmt(c["nodes"], 0), fmt(c["backtracks"], 0),
                                 fmt(c["pruned"], 0), fmt(c["fc_removed"], 0), fmt(c["runtime_ms"], 1),
                                 fmt(100 * c["complete"], 0), fmt(100 * c["greedy_equal"], 0)]]))

    parts.append("<h2>How to read the metrics</h2><ul>"
                 "<li><b>Satisfaction</b>: utility of each completed visit (interest match x quality), reduced by up to half when the "
                 "place is over capacity, minus 0.06 per minute queueing and 0.01 per minute travelling.</li>"
                 "<li><b>Overcrowded visits</b>: visits made while arrivals exceeded the place's capacity in that 15-minute slot.</li>"
                 "<li><b>Peak load</b>: highest arrivals / capacity at any attraction in any slot. <b>Load Gini</b>: inequality of "
                 "attractions' average load 10:00-17:00 (lower is more even).</li>"
                 "<li><b>Feasible</b>: no failed stop (closed, sold out, too late), back before the end of the day, within budget and walking limit.</li>"
                 "<li>CO2 counts taxi and transit legs per passenger; walking is zero.</li></ul>")
    page = PAGE.replace("{{BODY}}", "\n".join(parts))
    out = os.path.join(out_dir, "report.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)
    return out


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tourism Planner Results</title>
<style>
:root { --bg:#ffffff; --fg:#1d2433; --muted:#5b6475; --line:#e3e7ee; --head:#f4f6fa; }
@media (prefers-color-scheme: dark) { :root { --bg:#14171c; --fg:#e7eaf0; --muted:#9aa3b2; --line:#2b313b; --head:#1c2129; } }
body { background:var(--bg); color:var(--fg); font:15px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       max-width:1000px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:26px; margin:8px 0 4px; } h2 { font-size:19px; margin-top:36px; border-bottom:1px solid var(--line); padding-bottom:4px; }
.lead, .note { color:var(--muted); }
table { border-collapse:collapse; width:100%; margin:12px 0; font-size:13px; display:block; overflow-x:auto; }
th, td { border-bottom:1px solid var(--line); padding:6px 8px; text-align:right; white-space:nowrap; }
th { white-space:normal; vertical-align:bottom; min-width:56px; }
td:first-child { white-space:normal; min-width:150px; }
th:first-child, td:first-child { text-align:left; } th { background:var(--head); }
details { margin:6px 0; } summary { cursor:pointer; font-weight:600; }
svg.chart { width:100%; height:auto; margin:14px 0; } .ct { font-size:14px; font-weight:600; fill:var(--fg); }
.tick { font-size:11px; fill:var(--muted); } .tiny { font-size:10px; fill:var(--muted); }
.grid { stroke:var(--line); } .axis0 { stroke:var(--muted); stroke-dasharray:3 3; }
</style></head><body>
{{BODY}}
</body></html>
"""
