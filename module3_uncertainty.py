r"""
MODULE 3 - UNCERTAINTY: Bayesian network, sensor updates, Monte Carlo risk

Nobody knows in advance how crowded a place will be. We model it with a
Bayesian network and learn its probability tables from past data.

    Day ----\                                   (weekday / weekend)
    Season --\                                  (off / peak)
    Weather ---> Crowd ---> Wait                (low/medium/high -> short/medium/long)
    Block ---/       (time of day: morning / midday / afternoon / evening)
    Fame ---/        (quiet / normal / famous place)
    Setting /        (indoor / outdoor)
    Weather ---> BusDelay                       (on_time / late)

1. Data: data/history.csv (hourly visitor logs) and data/transit_history.csv
   (bus delays), made by generate_data.py from a hidden "true" model that
   this module never sees, like a real system that only has logs.
2. Learning: count how often each value happens for every combination of
   parent values, with Laplace (+1) smoothing.
3. Inference: P(Crowd | evidence) by enumeration (sum over unknown variables).
   A weather forecast like "60% rain" is used as soft evidence.
4. Sensor update: Bayes' rule with a noisy live crowd sensor.
5. Monte Carlo risk: sample crowds, queues and bus delays many times to
   estimate the chance that a plan fails.
"""
import csv
import random

DAY = ["weekday", "weekend"]
SEASON = ["off", "peak"]
WEATHER = ["clear", "rain"]
BLOCK = ["morning", "midday", "afternoon", "evening"]
FAME = ["quiet", "normal", "famous"]
SETTING = ["indoor", "outdoor"]
CROWD = ["low", "medium", "high"]
WAIT = ["short", "medium", "long"]
DELAY = ["on_time", "late"]
WAIT_MINUTES = {"short": (0, 5), "medium": (5, 20), "long": (20, 45)}


def block_of(minute):
    if minute < 11 * 60:
        return "morning"
    if minute < 14 * 60:
        return "midday"
    if minute < 17 * 60:
        return "afternoon"
    return "evening"


def fame_of(place):
    p = place.get("popularity", 0.5)
    return "quiet" if p < 0.45 else ("normal" if p < 0.75 else "famous")


def setting_of(place):
    return "indoor" if place.get("indoor") else "outdoor"


def read_history(path="data/history.csv"):
    """Rows of a CSV file as dicts (works for history.csv and transit_history.csv)."""
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------- Bayesian network
class BayesNet:
    def __init__(self):
        self.nodes = {}          # name -> {"values", "parents", "cpt"}; cpt[parent values] = {value: prob}
        self.order = []          # topological order (parents are always added before children)

    def add(self, name, values, parents, cpt):
        self.nodes[name] = {"values": values, "parents": parents, "cpt": cpt}
        self.order.append(name)

    def p(self, name, value, assignment, soft=None):
        """P(name = value | its parents), using soft evidence for root nodes if given."""
        if soft and name in soft:
            return soft[name][value]
        node = self.nodes[name]
        key = tuple(assignment[q] for q in node["parents"])
        return node["cpt"][key][value]

    def query(self, var, evidence, soft=None):
        """P(var | evidence) by enumeration. soft = {"Weather": {"clear": .4, "rain": .6}}."""
        dist = {}
        for v in self.nodes[var]["values"]:
            dist[v] = self._enumerate(self.order, dict(evidence, **{var: v}), soft)
        total = sum(dist.values())
        return {v: x / total for v, x in dist.items()}

    def _enumerate(self, names, assignment, soft):
        if not names:
            return 1.0
        first, rest = names[0], names[1:]
        if first in assignment:
            return self.p(first, assignment[first], assignment, soft) * self._enumerate(rest, assignment, soft)
        return sum(self.p(first, v, assignment, soft) * self._enumerate(rest, dict(assignment, **{first: v}), soft)
                   for v in self.nodes[first]["values"])


def _cpt(rows, child, child_values, parents, parent_values):
    """Count child values for every parent combination, with Laplace (+1) smoothing."""
    counts = {}
    for r in rows:
        key = tuple(r[q] for q in parents)
        counts.setdefault(key, {v: 0 for v in child_values})
        counts[key][r[child]] += 1
    cpt = {}

    def all_keys(i):
        if i == len(parents):
            yield ()
            return
        for v in parent_values[i]:
            for rest in all_keys(i + 1):
                yield (v,) + rest

    for key in all_keys(0):
        c = counts.get(key, {v: 0 for v in child_values})
        total = sum(c.values()) + len(child_values)
        cpt[key] = {v: (c[v] + 1) / total for v in child_values}
    return cpt


def learn_network(rows, transit_rows=()):
    """Learn every probability table from the visitor history and the bus delay history."""
    days = list({r["day"]: r for r in rows}.values())                    # one row per day
    uniform = lambda values: {(): {v: 1 / len(values) for v in values}}
    net = BayesNet()
    net.add("Day", DAY, [], _cpt(days, "Day", DAY, [], []))
    net.add("Season", SEASON, [], _cpt(days, "Season", SEASON, [], []))
    net.add("Weather", WEATHER, [], _cpt(days, "Weather", WEATHER, [], []))
    net.add("Block", BLOCK, [], uniform(BLOCK))          # always observed, so the prior does not matter
    net.add("Fame", FAME, [], uniform(FAME))
    net.add("Setting", SETTING, [], uniform(SETTING))
    parents = ["Day", "Season", "Weather", "Block", "Fame", "Setting"]
    net.add("Crowd", CROWD, parents, _cpt(rows, "Crowd", CROWD, parents, [DAY, SEASON, WEATHER, BLOCK, FAME, SETTING]))
    net.add("Wait", WAIT, ["Crowd"], _cpt(rows, "Wait", WAIT, ["Crowd"], [CROWD]))
    net.add("BusDelay", DELAY, ["Weather"], _cpt(transit_rows, "BusDelay", DELAY, ["Weather"], [WEATHER]))
    return net


# ------------------------------------------------------------- forecasting
def evidence_for(place, minute, context):
    ev = {"Day": context["day"], "Season": context["season"], "Block": block_of(minute),
          "Fame": fame_of(place), "Setting": setting_of(place)}
    soft = None
    if "weather" in context:                         # weather known for sure
        ev["Weather"] = context["weather"]
    else:                                            # only a forecast: soft evidence
        r = context.get("rain_prob", 0.2)
        soft = {"Weather": {"clear": 1 - r, "rain": r}}
    return ev, soft


def crowd_distribution(net, city, context, name, minute):
    ev, soft = evidence_for(city["by_name"][name], minute, context)
    return net.query("Crowd", ev, soft)


def make_forecast(net, city, context):
    """Returns forecast(place name, minute) -> (P(crowd is high), expected queue in minutes).
    Modules 1 and 2 call this while planning."""
    cache = {}

    def forecast(name, minute):
        minute = 12 * 60 if minute is None else minute
        key = (name, block_of(minute))
        if key not in cache:
            ev, soft = evidence_for(city["by_name"][name], minute, context)
            crowd = net.query("Crowd", ev, soft)
            wait = net.query("Wait", ev, soft)
            expected = sum(wait[w] * sum(WAIT_MINUTES[w]) / 2 for w in WAIT)
            cache[key] = (crowd["high"], expected)
        return cache[key]
    return forecast


def sensor_update(prior, reading, accuracy=0.8):
    """Bayes' rule: P(crowd | sensor says `reading`). The sensor is right with
    probability `accuracy`, otherwise it reports one of the other two levels."""
    post = {c: prior[c] * (accuracy if c == reading else (1 - accuracy) / 2) for c in CROWD}
    total = sum(post.values())
    return {c: v / total for c, v in post.items()}


def evaluate_forecast(rows, test_days=30):
    """Train on the first days, test on the last ones. Compares the Bayesian
    network with a simple baseline that only knows each place's usual crowd levels.
    accuracy: how often the most likely level is right (higher is better)
    brier:    mean squared error of the probabilities (lower is better)"""
    last = max(int(r["day"]) for r in rows)
    train = [r for r in rows if int(r["day"]) <= last - test_days]
    test = [r for r in rows if int(r["day"]) > last - test_days]
    net = learn_network(train)
    usual = {}
    for r in train:
        usual.setdefault(r["place"], {c: 0 for c in CROWD})[r["Crowd"]] += 1
    bn_ok = base_ok = bn_brier = base_brier = 0.0
    cache = {}
    for r in test:
        ev = {k: r[k] for k in ("Day", "Season", "Weather", "Block", "Fame", "Setting")}
        key = tuple(ev.values())
        if key not in cache:
            cache[key] = net.query("Crowd", ev)
        bn = cache[key]
        counts = usual[r["place"]]
        base = {c: counts[c] / sum(counts.values()) for c in CROWD}
        bn_ok += max(bn, key=bn.get) == r["Crowd"]
        base_ok += max(base, key=base.get) == r["Crowd"]
        bn_brier += sum((bn[c] - (c == r["Crowd"])) ** 2 for c in CROWD)
        base_brier += sum((base[c] - (c == r["Crowd"])) ** 2 for c in CROWD)
    n = len(test)
    return {"test_rows": n, "bn_accuracy": bn_ok / n, "baseline_accuracy": base_ok / n,
            "bn_brier": bn_brier / n, "baseline_brier": base_brier / n}


# ---------------------------------------------------------- Monte Carlo risk
def plan_risk(net, city, context, steps, end_minute, n=500, seed=1):
    """Simulate the plan n times with random crowds, queues and bus delays.
    A run fails if a visit cannot finish before closing or the traveller is
    back after the end of the day."""
    rng = random.Random(seed)
    crowd_cache = {}
    fails, waits, highs, extras = 0, 0.0, 0, []
    for _ in range(n):
        if "weather" in context:
            weather = context["weather"]
        else:
            weather = "rain" if rng.random() < context.get("rain_prob", 0.2) else "clear"
        ctx = dict(context, weather=weather)
        extra, failed = 0.0, False
        for s in steps:
            if s["action"] == "visit":
                key = (s["place"], block_of(s["start"]), weather)
                if key not in crowd_cache:
                    crowd_cache[key] = crowd_distribution(net, city, ctx, s["place"], s["start"])
                dist = crowd_cache[key]
                crowd = rng.choices(CROWD, [dist[c] for c in CROWD])[0]
                wait_probs = net.nodes["Wait"]["cpt"][(crowd,)]
                lo, hi = WAIT_MINUTES[rng.choices(WAIT, [wait_probs[w] for w in WAIT])[0]]
                minutes = rng.uniform(lo, hi)
                waits += minutes
                highs += crowd == "high"
                extra += max(0.0, minutes - s.get("queue", 0.0))     # the plan already allowed for `queue`
                if s["end"] + extra > city["by_name"][s["place"]]["close_min"]:
                    failed = True
            elif s["action"] == "travel" and any(l["mode"] == "bus" for l in s["route"]["legs"]):
                if rng.random() < net.nodes["BusDelay"]["cpt"][(weather,)]["late"]:
                    extra += rng.uniform(5, 15)
        if steps and steps[-1]["end"] + extra > end_minute:
            failed = True
        fails += failed
        extras.append(extra)
    extras.sort()
    return {"p_fail": fails / n, "mean_queue_min": waits / n, "mean_crowded_visits": highs / n,
            "p90_delay_min": extras[int(0.9 * (n - 1))]}
