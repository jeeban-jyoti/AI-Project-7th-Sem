"""
MODULE 5 - GENERATIVE AI: talking to the planner in plain language

1. parse_request(text)  turns a sentence like
       "We are a family with a wheelchair user at Lakeside Resort, we love
        nature and kids' places, budget 3000, from 10 to 6, no taxis"
   into the traveller profile the other modules need.
2. explain_plan(...)    turns the planner's numbers into a short friendly
   explanation of the plan and why it was chosen.
3. ask_profile(city)    a guided form for the interactive trip planner: the
   traveller answers by picking numbered options (hotel, kind of day, what
   they enjoy, must-see places, times, budget, transport, pace).

Both work offline with simple rules (keyword matching and templates).
If the `anthropic` package is installed (pip install anthropic) and an API key
is set, pass use_llm=True to let Claude do both steps instead. Anything Claude
returns is checked, and the rules are used whenever Claude is unavailable.
"""
import json
import re

MODEL = "claude-opus-5-5"

INTEREST_WORDS = {
    "history": ["history", "historic", "heritage", "fort", "palace", "museum", "ruins"],
    "art": ["art", "gallery", "painting", "sculpture"],
    "nature": ["nature", "garden", "park", "lake", "outdoor", "view"],
    "spiritual": ["temple", "shrine", "spiritual", "monastery", "meditation"],
    "architecture": ["architecture", "building", "tower", "bridge"],
    "family": ["kid", "child", "family", "zoo", "science"],
    "food": ["food", "eat", "restaurant", "biryani", "thali", "foodie"],
    "shopping": ["shopping", "market", "bazaar", "shop"],
    "crafts": ["craft", "pottery", "handloom", "workshop"],
}

DEFAULT_PROFILE = {
    "name": "You", "hotel": "Central Hotel", "start": "09:30", "end": "18:00", "budget": 3000,
    "interests": {"history": 0.6, "art": 0.4, "nature": 0.4, "food": 0.4},
    "max_places": 4, "must_visit": [], "accessible_only": False, "lunch": True,
    "modes": ["walk", "bus", "taxi"], "flexibility": 0.6,
}


# ------------------------------------------------------------ rule-based
def _clock(hour, minute, ampm):
    h, m = int(hour), int(minute or 0)
    if ampm == "pm" and h < 12:
        h += 12
    elif not ampm and h < 8:          # "from 10 to 6" means 6 pm
        h += 12
    return "%02d:%02d" % (h, m)


def parse_request(text, city, use_llm=False):
    """Plain text -> (profile, list of things we understood)."""
    if use_llm:
        profile = _parse_with_claude(text, city)
        if profile:
            return profile, ["understood by Claude"]
    t = text.lower()
    profile = json.loads(json.dumps(DEFAULT_PROFILE))       # deep copy
    notes = []

    found = [cat for cat, words in INTEREST_WORDS.items() if any(w in t for w in words)]
    if found:
        profile["interests"] = {cat: 1.0 if i == 0 else 0.8 for i, cat in enumerate(found)}
        notes.append("interests: " + ", ".join(found))
    if "wheelchair" in t:
        profile["accessible_only"] = True
        profile["modes"] = ["walk", "bus"]
        notes.append("wheelchair accessible places only")
    if any(w in t for w in ["no taxi", "no cab", "eco", "green", "sustainab"]):
        profile["modes"] = ["walk", "bus"]
        notes.append("no taxis")
    if "no lunch" in t:
        profile["lunch"] = False
        notes.append("no lunch break")
    m = re.search(r"(?:budget|rs\.?|inr|₹)\s*(\d+)", t) or re.search(r"(\d+)\s*(?:rupees|rs\b)", t)
    if m:
        profile["budget"] = int(m.group(1))
        notes.append("budget %s" % m.group(1))
    m = re.search(r"from (\d{1,2})(?::(\d\d))?\s*(am|pm)?\s*(?:to|till|until) (\d{1,2})(?::(\d\d))?\s*(am|pm)?", t)
    if m:
        profile["start"] = _clock(m.group(1), m.group(2), m.group(3))
        profile["end"] = _clock(m.group(4), m.group(5), m.group(6))
        notes.append("day %s-%s" % (profile["start"], profile["end"]))
    m = re.search(r"(\d+) (?:places|stops|sights)", t)
    if m:
        profile["max_places"] = max(1, min(6, int(m.group(1))))
        notes.append("up to %d places" % profile["max_places"])
    for p in city["places"]:
        if p["name"].lower() in t:
            if p["type"] == "hotel":
                profile["hotel"] = p["name"]
                notes.append("staying at " + p["name"])
            elif p["type"] in ("attraction", "local"):
                profile["must_visit"].append(p["name"])
                notes.append("must visit " + p["name"])
    return profile, notes


def explain_plan(facts, use_llm=False):
    """facts: dict built by main.py with numbers from all modules."""
    lines = []
    c, s = facts.get("conventional"), facts["sustainable"]
    lines.append("Your day has %d stops and you are back at %s." % (s["stops"], s["back_at"]))
    if c:
        if c["crowded"] - s["crowded"] >= 0.1:
            lines.append("Compared with a plan that only chases famous places, you meet crowds %.1f times "
                         "instead of %.1f." % (s["crowded"], c["crowded"]))
        if c["queue"] - s["queue"] >= 1:
            lines.append("You queue about %.0f min instead of %.0f min." % (s["queue"], c["queue"]))
        if c["co2"] - s["co2"] >= 1:
            lines.append("Walking and buses cut CO2 from %.0f g to %.0f g." % (c["co2"], s["co2"]))
    if facts.get("checked") and not facts.get("suggestions"):
        lines.append("Our learned policy checked each stop for a quieter time or a similar quieter place "
                     "and found nothing worth changing.")
    for sug in facts.get("suggestions", []):
        lines.append("We suggested: %s (%s)." % (sug["text"], "you accepted" if sug["accepted"] else "you said no"))
    for r in facts.get("relaxed", []):
        lines.append("Not everything fit: %s, so it was left out." % r)
    if facts.get("dropped"):
        lines.append("The detailed timing did not allow " + ", ".join(facts["dropped"]) + ".")
    lines.append("There is a %.0f%% chance that queues or delays break the plan." % (100 * s["p_fail"]))
    if facts.get("plan_b"):
        lines.append("Plan B if things go wrong: " + facts["plan_b"] + ".")
    text = " ".join(lines)
    if use_llm:
        better = _ask_claude(
            "Rewrite this travel plan summary for the traveller in a warm, clear tone, under 120 words. "
            "Keep every number exactly as given and add no new facts.\n\n" + text,
            "You explain day trip itineraries to tourists.")
        if better:
            return better
    return text


# ------------------------------------------------------------ guided form
CATEGORY_LABELS = {
    "history": "History (forts, palaces, museums)", "art": "Art (galleries, street art)",
    "nature": "Nature (gardens, lakes, viewpoints)", "spiritual": "Temples, shrines and monasteries",
    "architecture": "Architecture (towers, bridges, old buildings)", "family": "Family and kids (zoo, science centre)",
    "food": "Food (markets, food streets)", "shopping": "Shopping (bazaars, local shops)",
    "crafts": "Crafts (pottery, weaving, jewellery)",
}


def ask(prompt, default=""):
    """input() that returns the default on Enter, or when nothing can be typed."""
    try:
        answer = input(prompt).strip()
    except EOFError:
        print()
        return default
    return answer or default


def choose_one(title, options, default=1):
    """Show numbered options and return the index (0-based) of the one chosen."""
    print("\n  " + title)
    for i, label in enumerate(options, 1):
        print("    %d. %s" % (i, label))
    answer = ask("  Your choice [%d]: " % default, str(default))
    if answer.isdigit() and 1 <= int(answer) <= len(options):
        return int(answer) - 1
    print("  (not one of the options, using %d)" % default)
    return default - 1


def choose_many(title, options, default=""):
    """Show numbered options and return the indexes chosen, in the order typed ("2,5" or "2 5")."""
    print("\n  " + title)
    for i, label in enumerate(options, 1):
        print("    %d. %s" % (i, label))
    answer = ask("  Your choices, e.g. 1,3 [%s]: " % (default or "none"), default)
    picks = []
    for part in re.split(r"[,\s]+", answer):
        if part.isdigit() and 1 <= int(part) <= len(options) and int(part) - 1 not in picks:
            picks.append(int(part) - 1)
    return picks


def choose_time(title, options, default):
    """Pick a time from a list, or type another one."""
    i = choose_one(title, options + ["Another time (type it)"], options.index(default) + 1)
    if i < len(options):
        return options[i]
    typed = ask("  Time as HH:MM [%s]: " % default, default)
    if re.fullmatch(r"([01]?\d|2[0-3]):[0-5]\d", typed):
        return "%02d:%s" % (int(typed.split(":")[0]), typed.split(":")[1])
    print("  (not a time, using %s)" % default)
    return default


def ask_profile(city, crowd_at=None):
    """The guided form of the interactive trip planner. Returns (profile, day context).
    crowd_at(place name, context) -> chance it is crowded at midday (shown next to places)."""
    print("\n  Answer with the number of an option. Enter takes the suggestion in [brackets].")
    hotels = [p for p in city["places"] if p["type"] == "hotel"]
    hotel = hotels[choose_one("1/12  Where are you staying?", ["%s (%s)" % (p["name"], p["zone"]) for p in hotels])]
    day = ["weekday", "weekend"][choose_one("2/12  Which day are you going out?",
                                            ["A weekday (Monday to Friday)", "A weekend day"], 2)]
    season = ["off", "peak"][choose_one("3/12  Which season is it?",
                                        ["Off season (fewer tourists)", "Peak season (holidays, festivals)"], 2)]
    rain = [0.1, 0.4, 0.8][choose_one("4/12  What does the weather forecast say?",
                                      ["Sunny (10% chance of rain)", "Cloudy (40%)", "Rainy (80%)"], 1)]
    context = {"day": day, "season": season, "rain_prob": rain}

    cats = list(CATEGORY_LABELS)
    picks = choose_many("5/12  What do you enjoy? Pick one or more, your favourite first.",
                        [CATEGORY_LABELS[c] for c in cats], "1") or [0]
    interests = {cats[i]: 1.0 if k == 0 else 0.8 for k, i in enumerate(picks)}

    def match(p):
        return min(1.0, sum(interests.get(c, 0) for c in p["categories"])) * p["rating"]
    top = sorted([p for p in city["places"] if p["type"] in ("attraction", "local") and match(p) > 0],
                 key=match, reverse=True)[:10]
    lines = []
    for p in top:
        line = "%-27s %-15s rating %.1f, %s, open %s-%s" % (p["name"], p["zone"], p["rating"],
                                                         "free" if not p["price"] else "costs %d" % p["price"],
                                                         p["open"], p["close"])
        if crowd_at:
            line += ", crowded at midday: %d%%" % round(100 * crowd_at(p["name"], context))
        if not p.get("accessible", True):
            line += ", NOT wheelchair accessible"
        lines.append(line)
    must = [top[i]["name"] for i in choose_many("6/12  Any place you must see? (Enter = none, let the planner choose)",
                                                lines)]

    start = choose_time("7/12  When do you leave the hotel?", ["08:00", "09:00", "09:30", "10:00", "11:00"], "09:30")
    end = choose_time("8/12  When do you want to be back?", ["15:00", "17:00", "18:00", "19:00", "20:00"], "18:00")
    if end <= start:
        end = "%02d:%s" % (min(21, int(start[:2]) + 8), start[3:])
        print("  (the end must be after the start, using %s)" % end)
    budgets = [1000, 2000, 3000, 5000]
    i = choose_one("9/12  What is your budget for the day (tickets, food, transport)?",
                   ["%d" % b for b in budgets] + ["Another amount (type it)"], 3)
    if i < len(budgets):
        budget = budgets[i]
    else:
        typed = ask("  Amount [3000]: ", "3000")
        budget = int(typed) if typed.isdigit() and int(typed) > 0 else 3000
    taxi = choose_one("10/12 How do you want to get around?",
                      ["Walking and buses only (greenest)", "Walking, buses and taxis"], 2) == 1
    wheelchair = choose_one("11/12 Does anyone in your group need wheelchair access?", ["No", "Yes"], 1) == 1
    pace = choose_one("12/12 What pace do you like?",
                      ["Relaxed: 3 places and lunch", "Normal: 4 places and lunch", "Packed: 5 places and lunch",
                       "Normal, but no lunch break"], 2)
    profile = {"name": "Your day", "hotel": hotel["name"], "start": start, "end": end, "budget": budget,
               "interests": interests, "max_places": [3, 4, 5, 4][pace], "must_visit": must,
               "accessible_only": wheelchair, "lunch": pace != 3,
               "modes": ["walk", "bus", "taxi"] if taxi and not wheelchair else ["walk", "bus"], "flexibility": 0.6}
    print("\n  Your choices: %s, %s %s season, %d%% rain; %s; %s-%s; budget %d; %s; up to %d places%s%s."
          % (profile["hotel"], day, season, 100 * rain, ", ".join(interests), start, end, budget,
             "taxis allowed" if "taxi" in profile["modes"] else "no taxis", profile["max_places"],
             "; wheelchair access" if wheelchair else "", "; must see " + ", ".join(must) if must else ""))
    return profile, context


# --------------------------------------------------------- Claude (optional)
def _ask_claude(prompt, system):
    """One Claude call. Returns the text, or None if Claude is not available."""
    try:
        import anthropic
    except ImportError:
        print("  (the anthropic package is not installed, using the rules instead)")
        return None
    try:
        client = anthropic.Anthropic()            # reads ANTHROPIC_API_KEY from the environment
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=2000,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",                  # if this model declines, the API retries on a fallback model
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIStatusError as e:
        print("  (Claude error %s, using the rules instead)" % e.status_code)
        return None
    except anthropic.APIConnectionError:
        print("  (no connection to Claude, using the rules instead)")
        return None
    except anthropic.AnthropicError as e:         # e.g. no API key configured
        print("  (Claude not available: %s; using the rules instead)" % type(e).__name__)
        return None
    if response.stop_reason == "refusal":
        return None
    return "".join(b.text for b in response.content if b.type == "text").strip() or None


def _parse_with_claude(text, city):
    names = [p["name"] for p in city["places"] if p["type"] in ("attraction", "local")]
    hotels = [p["name"] for p in city["places"] if p["type"] == "hotel"]
    prompt = (
        "Turn this travel request into JSON with exactly these keys: hotel (one of %s), start and end (HH:MM), "
        "budget (number), interests (object mapping some of %s to weights 0-1), max_places (1-6), "
        "must_visit (list, only names from %s), accessible_only (bool), lunch (bool), "
        "modes (list from walk, bus, taxi). Reply with the JSON only.\n\nRequest: %s"
        % (hotels, list(INTEREST_WORDS), names, text))
    reply = _ask_claude(prompt, "You convert travel requests into structured data.")
    if not reply or "{" not in reply:
        return None
    try:
        data = json.loads(reply[reply.index("{"):reply.rindex("}") + 1])
    except ValueError:
        return None
    # Never trust model output blindly: keep only valid keys and values.
    profile = json.loads(json.dumps(DEFAULT_PROFILE))
    if data.get("hotel") in hotels:
        profile["hotel"] = data["hotel"]
    for key in ("start", "end"):
        if isinstance(data.get(key), str) and re.fullmatch(r"\d\d:\d\d", data[key]):
            profile[key] = data[key]
    if isinstance(data.get("budget"), (int, float)) and data["budget"] > 0:
        profile["budget"] = data["budget"]
    if isinstance(data.get("interests"), dict):
        interests = {k: float(v) for k, v in data["interests"].items()
                     if k in INTEREST_WORDS and isinstance(v, (int, float))}
        profile["interests"] = interests or profile["interests"]
    if isinstance(data.get("max_places"), int):
        profile["max_places"] = max(1, min(6, data["max_places"]))
    profile["must_visit"] = [n for n in data.get("must_visit", []) if n in names]
    for key in ("accessible_only", "lunch"):
        if isinstance(data.get(key), bool):
            profile[key] = data[key]
    modes = [m for m in data.get("modes", []) if m in ("walk", "bus", "taxi")]
    profile["modes"] = modes if "walk" in modes else ["walk"] + modes
    return profile
