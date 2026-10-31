"""
MAIN - connects the five modules into one tourism planner.

    traveller request (text)  --module 5-->  profile
    profile + day context     --module 3-->  crowd forecast (Bayesian network)
    forecast                  --module 1-->  itinerary (A* routes inside a CSP)
    itinerary                 --module 4-->  suggestions: keep / shift / substitute (Q-learning)
    final places              --module 2-->  timed action plan (A* planner), replanning on events
    plan                      --module 3-->  risk (Monte Carlo)
    everything                --module 5-->  explanation

Usage:
    python3 main.py                     full demo, every module step by step
    python3 main.py interactive         choose your day from options and get your own plan
    python3 main.py plan family         plan a sample traveller (culture, family, foodie, impossible)
    python3 main.py ask "text"          describe your trip in plain words
    python3 main.py search              UCS vs Greedy vs A*
    python3 main.py forecast            Bayesian network accuracy and examples
    python3 main.py rl                  train and compare the RL policies
    python3 main.py compare             conventional vs sustainable for 30 travellers from the dataset
    python3 main.py test                run all tests (same as: python3 test.py all)

The data in data/ is made by generate_data.py (it runs automatically the first time).

Options: --interactive (you answer the suggestions yourself), --llm (use Claude in module 5),
         --day weekday|weekend, --season off|peak, --rain 0.3
"""
import argparse
import json
import random

from generate_data import HISTORY_FILE, TRANSIT_FILE, TRAVELLERS_FILE, ensure_data
from module1_search_csp import SYSTEMS, compare_search, hhmm, load_city, solve_itinerary, to_min
from module2_planning import make_problem, plan_summary, plan_with_relaxation, print_plan, replan, validate_plan
from module3_uncertainty import (crowd_distribution, evaluate_forecast, learn_network, make_forecast, plan_risk,
                                 read_history, sensor_update)
from module4_rl import (ACTIONS, BLOCK_START, CrowdEnv, block_index, compare_policies, find_alternative, suggest,
                        train)
from module5_genai import ask_profile, choose_one, explain_plan, parse_request

HISTORY = HISTORY_FILE

SAMPLE_PROFILES = {
    "culture": {"name": "Asha (culture lover)", "hotel": "Central Hotel", "start": "09:30", "end": "18:00",
                "budget": 3000, "interests": {"history": 1.0, "architecture": 0.8, "art": 0.6, "spiritual": 0.4},
                "max_places": 4, "must_visit": [], "accessible_only": False, "lunch": True,
                "modes": ["walk", "bus", "taxi"], "flexibility": 0.7},
    "family": {"name": "The Rao family (wheelchair user)", "hotel": "Lakeside Resort", "start": "10:00",
               "end": "18:00", "budget": 5000, "interests": {"family": 1.0, "nature": 0.8, "food": 0.5, "art": 0.3},
               "max_places": 3, "must_visit": [], "accessible_only": True, "lunch": True,
               "modes": ["walk", "bus"], "flexibility": 0.5},
    "foodie": {"name": "Ravi (foodie and shopper)", "hotel": "Riverside Inn", "start": "10:00", "end": "20:00",
               "budget": 2500, "interests": {"food": 1.0, "shopping": 0.7, "crafts": 0.6, "art": 0.3},
               "max_places": 4, "must_visit": [], "accessible_only": False, "lunch": True,
               "modes": ["walk", "bus", "taxi"], "flexibility": 0.8},
    "impossible": {"name": "Meera (impossible request)", "hotel": "Riverside Inn", "start": "09:00", "end": "14:00",
                   "budget": 1500, "interests": {"history": 1.0, "nature": 0.5}, "max_places": 3,
                   "must_visit": ["Hill Monastery", "Royal Fort"], "accessible_only": True, "lunch": True,
                   "modes": ["walk", "bus", "taxi"], "flexibility": 0.5},
}


def header(title):
    print("\n" + "#" * 72 + "\n#  " + title + "\n" + "#" * 72)


def setup():
    """Make the datasets if needed, load the city and learn the Bayesian network."""
    ensure_data()
    city = load_city()
    net = learn_network(read_history(HISTORY_FILE), read_history(TRANSIT_FILE))
    return city, net


def load_travellers():
    with open(TRAVELLERS_FILE, encoding="utf-8") as f:
        return json.load(f)


def gini(values):
    """Inequality of visits across places: 0 = perfectly even, close to 1 = everyone at one place."""
    v, n = sorted(values), len(values)
    total = sum(v)
    if n == 0 or total == 0:
        return 0.0
    return sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(v)) / (n * total)


def goals_of(result):
    sights = [s["name"] for s in result["stops"] if s["type"] != "restaurant"]
    meals = [s["name"] for s in result["stops"] if s["type"] == "restaurant"]
    return sights, (meals[0] if meals else None)


def ask_traveller(text, profile, interactive, rng):
    if interactive:
        return input("  Suggestion: %s. Accept? [y/n] " % text).strip().lower().startswith("y")
    return rng.random() < profile.get("flexibility", 0.6)       # a simulated traveller


# ------------------------------------------------------------------ pipeline
def plan_day(city, net, agent, profile, context, interactive=False, use_llm=False, show=True, seed=0):
    rng = random.Random(seed)
    forecast = make_forecast(net, city, context)
    profile = dict(profile, must_visit=list(profile.get("must_visit", [])))
    end = to_min(profile["end"])

    # Module 1: the conventional baseline and our crowd-aware itinerary
    conventional = solve_itinerary(city, profile, "conventional", forecast)
    earliest, banned = {}, set()
    best = solve_itinerary(city, profile, "sustainable", forecast)
    if not best["feasible"]:
        if show:
            print("\nNo feasible day exists for %s, even after relaxing must-visits." % profile["name"])
        return {"feasible": False}
    relaxed = best["relaxed"]
    if relaxed:                                                   # keep the relaxed version from now on
        profile["must_visit"] = [m for m in profile["must_visit"] if m not in relaxed]

    # Module 4: the learned policy proposes changes, one stop at a time
    suggestions = []
    checked = len([s for s in best["stops"] if s["type"] != "restaurant"]) if agent else 0
    for name in [s["name"] for s in best["stops"] if s["type"] != "restaurant"]:
        stop = next((s for s in best["stops"] if s["name"] == name), None)
        if stop is None or agent is None:
            continue
        alt = find_alternative(city, name, profile["accessible_only"])
        if alt in [s["name"] for s in best["stops"]] or name in profile["must_visit"]:
            alt = None                                            # must-visits may move in time, never be replaced
        dist = crowd_distribution(net, city, context, name, stop["start"])
        action = suggest(agent, dist, stop["start"], alt is not None)
        if action == "keep":
            continue
        new_earliest, new_banned, new_profile = dict(earliest), set(banned), dict(profile)
        if action == "shift":
            later = BLOCK_START[block_index(stop["start"]) + 1]
            new_earliest[name] = later
            text = "visit %s after %s, when it is quieter" % (name, hhmm(later))
        else:
            new_banned.add(name)
            new_profile["must_visit"] = profile["must_visit"] + [alt]
            text = "visit %s instead of %s (similar, but less crowded)" % (alt, name)
        accepted = ask_traveller(text, profile, interactive, rng)
        applied = False
        if accepted:                                             # re-solve the CSP; revert if it breaks
            trial = solve_itinerary(city, new_profile, "sustainable", forecast, new_earliest, new_banned)
            if trial["feasible"] and not trial["relaxed"]:
                best, earliest, banned, profile, applied = trial, new_earliest, new_banned, new_profile, True
        suggestions.append({"text": text, "action": action, "accepted": accepted, "applied": applied})

    # Module 2: turn both itineraries into timed action plans
    def to_plan(result, system):
        if not result["feasible"]:
            return None, None, []
        sights, lunch = goals_of(result)
        problem = make_problem(city, profile, sights, lunch, system, forecast,
                               earliest if system == "sustainable" else None)
        steps, dropped = plan_with_relaxation(problem)
        return problem, steps, dropped

    _, conv_steps, _ = to_plan(conventional, "conventional")
    problem, steps, dropped = to_plan(best, "sustainable")

    # Module 3: Monte Carlo risk, and a plan B when the risk is high
    def measure(result, plan_steps):
        if not plan_steps:
            return None
        risk = plan_risk(net, city, context, plan_steps, end)
        sm = plan_summary(plan_steps)
        return {"stops": len([s for s in plan_steps if s["action"] in ("visit", "eat")]),
                "back_at": hhmm(sm["back_at"]), "co2": sm["co2_g"], "travel": sm["travel_min"],
                "queue": risk["mean_queue_min"], "crowded": risk["mean_crowded_visits"], "p_fail": risk["p_fail"],
                "utility": result["utility"], "spend": result["spend"]}

    conv_m, ours_m = measure(conventional, conv_steps), measure(best, steps)
    plan_b = None
    if ours_m and ours_m["p_fail"] > 0.3:
        safer = solve_itinerary(city, dict(profile, max_places=max(1, profile["max_places"] - 1)), "sustainable",
                                forecast, earliest, banned)
        if safer["feasible"]:
            plan_b = "a shorter day with " + ", ".join(goals_of(safer)[0])

    # Independent check of the final plan (module 2's validator does not trust the planner)
    violations = validate_plan(city, profile, steps, relaxed=relaxed) if steps else ["no plan was found"]

    # Module 5: explanation
    facts = {"conventional": conv_m, "sustainable": ours_m, "suggestions": suggestions, "checked": checked,
             "relaxed": best["reasons"], "dropped": dropped, "plan_b": plan_b}
    explanation = explain_plan(facts, use_llm)

    if show:
        if conv_steps:
            print_plan(conv_steps, "%s - %s plan" % (profile["name"], SYSTEMS["conventional"]["label"]))
        print_plan(steps, "%s - %s plan" % (profile["name"], SYSTEMS["sustainable"]["label"]))
        print("\n  " + explanation)
        print("  Independent check: " + ("; ".join(violations) if violations else "every rule holds."))
    return {"feasible": True, "conventional": conv_m, "sustainable": ours_m, "suggestions": suggestions,
            "steps": steps, "problem": problem, "result": best, "explanation": explanation,
            "violations": violations, "conventional_result": conventional, "conventional_steps": conv_steps}


def get_context(args):
    return {"day": args.day, "season": args.season, "rain_prob": args.rain}


def get_agent(city):
    agent, _ = train(CrowdEnv(city), "q_learning", episodes=400, seed=1)
    return agent


# ------------------------------------------------------------------ commands
def cmd_search(city):
    header("MODULE 1 - Search: UCS vs Greedy best-first vs A* (cost = minutes + CO2/10)")
    pairs = [("Central Hotel", "Hill Monastery"), ("Lakeside Resort", "Riverside Shrine"),
             ("Riverside Inn", "City Zoo"), ("Lakeside Resort", "Grand Bazaar"), ("Central Hotel", "Botanical Garden")]
    rows = compare_search(city, pairs, co2_weight=1.0)
    print("  %-38s %16s %16s %16s" % ("route", "UCS cost/nodes", "Greedy cost/nodes", "A* cost/nodes"))
    for r in rows:
        print("  %-38s %9.1f / %3d %9.1f / %3d %9.1f / %3d" % (
            r["from"] + " -> " + r["to"], *r["ucs"], *r["greedy"], *r["astar"]))
    n = len(rows)
    print("  average nodes expanded: UCS %.0f, Greedy %.0f, A* %.0f"
          % tuple(sum(r[m][1] for r in rows) / n for m in ("ucs", "greedy", "astar")))
    print("  A* always finds the same (optimal) cost as UCS while expanding fewer nodes;")
    print("  Greedy is fastest but its routes cost more.")


def cmd_forecast(city, net, context):
    header("MODULE 3 - Bayesian network forecast")
    ev = evaluate_forecast(read_history(HISTORY))
    print("  %d held-out hours: accuracy %.1f%% (Bayesian network) vs %.1f%% (per-place average);"
          % (ev["test_rows"], 100 * ev["bn_accuracy"], 100 * ev["baseline_accuracy"]))
    print("  Brier score (lower is better) %.3f vs %.3f"
          % (ev["bn_brier"], ev["baseline_brier"]))
    print("  context: %s" % context)
    print("  %-22s %26s %26s %26s" % ("P(low/medium/high)", "10:00", "12:30", "16:00"))
    for name in ["Royal Fort", "National Art Gallery", "Riverside Ruins", "Lakeside Promenade"]:
        cells = []
        for minute in (600, 750, 960):
            d = crowd_distribution(net, city, context, name, minute)
            cells.append("%.2f / %.2f / %.2f" % (d["low"], d["medium"], d["high"]))
        print("  %-22s %26s %26s %26s" % (name, *cells))
    prior = crowd_distribution(net, city, context, "Royal Fort", 600)
    post = sensor_update(prior, "high")
    print("  a live sensor says 'high' at Royal Fort at 10:00: P(high) %.2f -> %.2f "
          "(Bayes' rule, sensor right 80%% of the time)" % (prior["high"], post["high"]))


def cmd_rl(city):
    header("MODULE 4 - Reinforcement learning: which suggestions reduce crowds?")
    agent, results, curves = compare_policies(city)
    print("  %-12s %10s %12s %10s %11s" % ("policy", "reward", "overcrowded", "enjoyment", "acceptance"))
    for name, r in results.items():
        print("  %-12s %10.3f %11.1f%% %10.2f %10.0f%%" % (name, r["reward_per_decision"],
                                                           100 * r["overcrowded_share"], r["enjoyment"],
                                                           100 * r["acceptance"]))
    curve = curves["Q-learning"]
    print("  Q-learning reward per decision, every 100 episodes: "
          + ", ".join("%.3f" % (sum(curve[i:i + 100]) / 100) for i in range(0, len(curve), 100)))
    env = CrowdEnv(city)
    print("  learned policy when a place is very crowded:")
    for block, label in enumerate(["morning", "midday", "afternoon", "evening"]):
        for has_alt in (0, 1):
            s = (2, block, has_alt)
            print("    %-9s %-22s -> %s" % (label, "quieter place nearby" if has_alt else "no similar place",
                                           ACTIONS[agent.act(s, env.valid_actions(s))]))


def review_plan(city, net, agent, result, profile, context):
    """Module 4 on a real itinerary: what would the learned policy suggest for each stop?"""
    print("  %-22s %6s %8s  %-11s %s" % ("stop", "time", "P(high)", "action", "suggestion"))
    for stop in result["stops"]:
        if stop["type"] == "restaurant":
            continue
        name = stop["name"]
        alt = find_alternative(city, name, profile["accessible_only"])
        dist = crowd_distribution(net, city, context, name, stop["start"])
        action = suggest(agent, dist, stop["start"], alt is not None)
        text = {"keep": "-",
                "shift": "go after %s" % hhmm(BLOCK_START[min(3, block_index(stop["start"]) + 1)]),
                "substitute": "try %s instead" % alt}[action]
        print("  %-22s %6s %8.2f  %-11s %s" % (name, hhmm(stop["start"]), dist["high"], action, text))


def cmd_compare(city, net, context, n=30):
    header("Conventional vs sustainable planner, first %d travellers of data/travellers.json (%s)" % (n, context))
    agent = get_agent(city)
    sights = [p["name"] for p in city["places"] if p["type"] in ("attraction", "local")]
    totals = {"conventional": {}, "sustainable": {}}
    visits = {"conventional": dict.fromkeys(sights, 0), "sustainable": dict.fromkeys(sights, 0)}
    violations = 0
    feasible = 0
    for i, profile in enumerate(load_travellers()[:n]):
        out = plan_day(city, net, agent, profile, context, show=False, seed=i)
        if not out["feasible"] or not out["conventional"]:
            continue
        feasible += 1
        violations += len(out["violations"])
        for system, steps in (("conventional", out["conventional_steps"]), ("sustainable", out["steps"])):
            for k, v in out[system].items():
                if isinstance(v, (int, float)):
                    totals[system][k] = totals[system].get(k, 0) + v
            for s in steps:
                if s["action"] == "visit" and s["place"] in visits[system]:
                    visits[system][s["place"]] += 1
    print("  travellers with a plan from both systems: %d of %d" % (feasible, n))
    print("  problems found by the independent validator in our plans: %d" % violations)
    labels = [("utility", "interest score (higher is better)"), ("queue", "queue minutes"),
              ("crowded", "visits at a crowded place"), ("co2", "CO2 grams"), ("travel", "travel minutes"),
              ("p_fail", "chance the plan fails"), ("spend", "money spent")]
    print("  %-36s %14s %14s" % ("average per traveller", "conventional", "sustainable"))
    for key, label in labels:
        c, s = totals["conventional"][key] / feasible, totals["sustainable"][key] / feasible
        print("  %-36s %14.2f %14.2f" % (label, c, s))
    print("  %-36s %14.2f %14.2f" % ("load inequality across places (Gini)", gini(visits["conventional"].values()),
                                     gini(visits["sustainable"].values())))
    print("  %-36s %14d %14d" % ("different places visited", sum(v > 0 for v in visits["conventional"].values()),
                                 sum(v > 0 for v in visits["sustainable"].values())))


# ---------------------------------------------------------- interactive trip planner
def what_if_closed(city, out):
    steps = out["steps"]
    visits = [s for s in steps if s["action"] == "visit"]
    i = choose_one("Which place is closed when you get there?",
                   ["%s (planned at %s)" % (s["place"], hhmm(s["start"])) for s in visits], len(visits))
    target = visits[i]["place"]
    arrival = next(s["end"] for s in steps if s["action"] == "travel" and s["place"] == target)
    problem = dict(out["problem"], closed=set())                   # a fresh copy of the planning problem
    strategy, new_steps, dropped = replan(problem, steps, arrival, {"type": "closed", "place": target})
    print("\n  You arrive at %s at %s and it is closed. Replanning: %s; dropped: %s"
          % (target, hhmm(arrival), strategy, ", ".join(dropped) or "nothing"))
    print_plan(new_steps)
    problems = validate_plan(city, problem["profile"], new_steps, closed={target}, relaxed=dropped)
    print("  Independent check: " + ("; ".join(problems) if problems else "every rule holds."))


def what_if_delay(city, out):
    steps = out["steps"]
    travels = [s for s in steps if s["action"] == "travel"]
    i = choose_one("On which trip does it happen?",
                   ["%s %s -> %s" % (hhmm(s["start"]), s["from"], s["place"]) for s in travels], min(2, len(travels)))
    minutes = [20, 40, 60][choose_one("How long is the delay?", ["20 minutes", "40 minutes", "60 minutes"], 2)]
    problem = dict(out["problem"], closed=set())
    strategy, new_steps, dropped = replan(problem, steps, travels[i]["start"], {"type": "delay", "minutes": minutes})
    print("\n  %d minutes lost at %s. Replanning: %s; dropped: %s"
          % (minutes, hhmm(travels[i]["start"]), strategy, ", ".join(dropped) or "nothing"))
    print_plan(new_steps)
    problems = validate_plan(city, problem["profile"], new_steps, relaxed=dropped)
    print("  Independent check: " + ("; ".join(problems) if problems else "every rule holds."))


def show_alternatives(city, net, out, profile, context):
    print("\n  %-27s %6s %8s   %-27s %8s" % ("your stop", "time", "crowded", "similar quieter place", "crowded"))
    for s in out["result"]["stops"]:
        if s["type"] == "restaurant":
            continue
        here = crowd_distribution(net, city, context, s["name"], s["start"])["high"]
        alt = find_alternative(city, s["name"], profile["accessible_only"])
        there = "%7.0f%%" % (100 * crowd_distribution(net, city, context, alt, s["start"])["high"]) if alt else ""
        print("  %-27s %6s %7.0f%%   %-27s %s" % (s["name"], hhmm(s["start"]), 100 * here, alt or "-", there))


def cmd_interactive(city, net, args):
    """Choose your day from numbered options, get a plan, then try what-if events."""
    header("INTERACTIVE TRIP PLANNER - choose your day, get your plan")
    agent = get_agent(city)

    def crowd_at(name, ctx):
        return crowd_distribution(net, city, ctx, name, 12 * 60 + 30)["high"]

    profile, context = ask_profile(city, crowd_at)
    while True:
        header("YOUR PLAN")
        out = plan_day(city, net, agent, profile, context, interactive=True, use_llm=args.llm)
        if not out["feasible"]:
            if choose_one("What now?", ["Change my answers", "Quit"], 1) == 1:
                return out
            profile, context = ask_profile(city, crowd_at)
            continue
        while True:
            pick = choose_one("What would you like to do next?",
                              ["Keep this plan, I am done", "Change my answers and plan again",
                               "What if one of my places is closed when I get there?",
                               "What if my bus or taxi is delayed?", "Show quieter alternatives for my stops"], 1)
            if pick == 0:
                print("\n  Enjoy your day!")
                return out
            if pick == 1:
                profile, context = ask_profile(city, crowd_at)
                break
            if pick == 2:
                what_if_closed(city, out)
            elif pick == 3:
                what_if_delay(city, out)
            else:
                show_alternatives(city, net, out, profile, context)


def cmd_demo(city, net, context, args):
    cmd_search(city)
    cmd_forecast(city, net, context)
    cmd_rl(city)
    agent = get_agent(city)

    header("ALL MODULES TOGETHER - planning a day (%s)" % context)
    out = plan_day(city, net, agent, SAMPLE_PROFILES["culture"], context, args.interactive, args.llm)

    header("MODULE 4 - The learned policy reviewing the famous-places plan")
    conventional = solve_itinerary(city, SAMPLE_PROFILES["culture"], "conventional")
    review_plan(city, net, agent, conventional, SAMPLE_PROFILES["culture"], context)
    print("  (On the crowd-aware plan above it found nothing to change: that plan already avoids the crowds.)")

    header("MODULE 2 - Replanning during the day")
    steps = out["steps"]
    visits = [s for s in steps if s["action"] == "visit"]
    events = [(visits[1]["start"], {"type": "closed", "place": visits[-1]["place"]},
               "%s suddenly closes for the day" % visits[-1]["place"])]
    later_trips = [s for s in steps if s["action"] == "travel" and s["start"] >= visits[0]["end"]]
    if later_trips:
        trip = later_trips[0]
        events.append((trip["start"], {"type": "delay", "minutes": 40},
                       "on the way to %s you are held up for 40 minutes" % trip["place"]))
    for when, event, story in events:
        problem = dict(out["problem"], closed=set())              # a fresh copy of the planning problem
        print("\n  %s: %s." % (hhmm(when), story))
        strategy, new_steps, dropped = replan(problem, steps, when, event)
        print("  strategy: %s, dropped: %s" % (strategy, ", ".join(dropped) or "nothing"))
        print_plan(new_steps)

    header("MODULE 5 - Plain-language request")
    text = ("We are a family staying at Lakeside Resort with a wheelchair user. We love nature and the zoo, "
            "budget 3000, from 10 to 6, no taxis please.")
    print('  "%s"' % text)
    profile, notes = parse_request(text, city, args.llm)
    print("  understood: " + "; ".join(notes))
    profile["name"] = "Family from the text request"
    plan_day(city, net, agent, profile, context, args.interactive, args.llm)

    header("An impossible request")
    plan_day(city, net, agent, SAMPLE_PROFILES["impossible"], context, args.interactive, args.llm)


def main():
    ap = argparse.ArgumentParser(description="Sustainable and crowd-aware tourism planner")
    ap.add_argument("command", nargs="?", default="demo",
                    choices=["demo", "interactive", "plan", "ask", "search", "forecast", "rl", "compare", "test"])
    ap.add_argument("arg", nargs="?", help="profile name for 'plan', or the request text for 'ask'")
    ap.add_argument("--interactive", action="store_true", help="answer the suggestions yourself")
    ap.add_argument("--llm", action="store_true", help="use Claude for module 5 (needs: pip install anthropic)")
    ap.add_argument("--day", default="weekend", choices=["weekday", "weekend"])
    ap.add_argument("--season", default="peak", choices=["off", "peak"])
    ap.add_argument("--rain", type=float, default=0.3, help="chance of rain, 0 to 1")
    args = ap.parse_args()

    city, net = setup()
    context = get_context(args)
    if args.command == "demo":
        cmd_demo(city, net, context, args)
    elif args.command == "interactive" or (args.command == "plan" and args.interactive and not args.arg):
        cmd_interactive(city, net, args)
    elif args.command == "plan":
        name = args.arg or "culture"
        if name not in SAMPLE_PROFILES:
            ap.error("profile must be one of: " + ", ".join(SAMPLE_PROFILES))
        plan_day(city, net, get_agent(city), SAMPLE_PROFILES[name], context, args.interactive, args.llm)
    elif args.command == "ask":
        text = args.arg or input("Describe your trip: ")
        profile, notes = parse_request(text, city, args.llm)
        print("  understood: " + ("; ".join(notes) or "nothing specific, using defaults"))
        plan_day(city, net, get_agent(city), profile, context, args.interactive, args.llm)
    elif args.command == "search":
        cmd_search(city)
    elif args.command == "forecast":
        cmd_forecast(city, net, context)
    elif args.command == "rl":
        cmd_rl(city)
    elif args.command == "compare":
        cmd_compare(city, net, context)
    elif args.command == "test":
        import test                                                         # the tests live in test.py
        raise SystemExit(0 if test.run_all() else 1)


if __name__ == "__main__":
    main()
