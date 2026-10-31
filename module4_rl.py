"""
MODULE 4 - REINFORCEMENT LEARNING

Problem: many travellers want the same famous places at the same time.
For each stop in a traveller's plan, an agent decides whether to suggest
a change. It has to learn when a suggestion is worth it: suggestions
reduce crowds, but travellers may reject them and lose some enjoyment.

MDP (one episode = one simulated day)
    State  : (crowd level at the stop: low/medium/high,
              time block: morning/midday/afternoon/evening,
              is there a similar quieter place nearby: no/yes)  -> 3 x 4 x 2 = 24 states
    Actions: keep        - leave the stop as planned
             shift       - suggest going in the next time block
             substitute  - suggest a similar, less crowded place
    Reward : enjoyment of the final choice
             - penalty if the place is over 80% full (hurts this traveller and others)
             + small bonus for using a less-visited place
             - small penalty if the traveller rejects the suggestion
    Transition: the traveller accepts or rejects (hidden willingness), the
             choice is added to the place's load, so later travellers that day
             face the crowds that earlier decisions created.

Algorithms: Q-learning (off-policy) and SARSA (on-policy), both tabular,
compared with three fixed policies: always keep, random, and a simple rule.
"""
import random

from module1_search_csp import distance_km

ACTIONS = ["keep", "shift", "substitute"]
KEEP, SHIFT, SUBSTITUTE = 0, 1, 2
BLOCK_START = [8 * 60, 11 * 60, 14 * 60, 17 * 60]      # morning, midday, afternoon, evening
BLOCK_SHAPE = [0.7, 1.15, 1.0, 0.6]                     # how busy each block usually is
GROUP_LOAD = 0.05                                       # one traveller group fills 5% of a place


def block_index(minute):
    return max(i for i, start in enumerate(BLOCK_START) if minute >= start) if minute >= BLOCK_START[0] else 0


def crowd_level(ratio):
    return 0 if ratio < 0.6 else (1 if ratio < 0.9 else 2)


def find_alternative(city, name, accessible_only=False):
    """A similar place (shares the main category) that is clearly less popular,
    nearly as good, and within 2.5 km. None if there is no such place."""
    p = city["by_name"][name]
    if p["type"] not in ("attraction", "local"):
        return None
    best = None
    for q in city["places"]:
        if q["type"] not in ("attraction", "local") or q["name"] == name:
            continue
        if (p["categories"][0] in q["categories"] and q["popularity"] <= p["popularity"] - 0.2
                and q["rating"] >= p["rating"] - 0.3 and distance_km(p, q) <= 2.5
                and (q.get("accessible", True) or not accessible_only)):
            if best is None or q["rating"] > best["rating"]:
                best = q
    return best["name"] if best else None


# --------------------------------------------------------------- environment
class CrowdEnv:
    def __init__(self, city, travellers=40, stops_each=3):
        self.city = city
        self.sights = [p for p in city["places"] if p["type"] == "attraction"]            # what people ask for
        self.all_places = [p for p in city["places"] if p["type"] in ("attraction", "local")]
        self.n_travellers = travellers
        self.stops_each = stops_each
        self.alt = {p["name"]: find_alternative(city, p["name"]) for p in self.sights}

    def reset(self, seed):
        self.rng = random.Random(seed)
        # background visitors that are not using our app
        self.load = {p["name"]: [0.25 + 0.55 * p["popularity"] * s for s in BLOCK_SHAPE] for p in self.all_places}
        self.queue = []                    # (willingness, place, block) for every stop of every traveller
        weights = [p["popularity"] ** 3 for p in self.sights]        # most people want the famous places
        for _ in range(self.n_travellers):
            willingness = self.rng.random()                          # hidden from the agent
            places = []
            while len(places) < self.stops_each:
                name = self.rng.choices(self.sights, weights)[0]["name"]
                if name not in places:
                    places.append(name)
            for name in places:
                block = self.rng.choices(range(4), [0.2, 0.4, 0.3, 0.1])[0]
                self.queue.append((willingness, name, block))
        self.i = 0
        self.stats = {"reward": 0.0, "decisions": 0, "suggested": 0, "accepted": 0, "overcrowded": 0,
                      "enjoyment": 0.0}
        return self.state()

    def done(self):
        return self.i >= len(self.queue)

    def state(self):
        _, name, block = self.queue[self.i]
        return (crowd_level(self.load[name][block]), block, int(self.alt[name] is not None))

    def valid_actions(self, state):
        acts = [KEEP]
        if state[1] < 3:
            acts.append(SHIFT)
        if state[2] == 1:
            acts.append(SUBSTITUTE)
        return acts

    def step(self, action):
        willingness, name, block = self.queue[self.i]
        place, new_block, enjoy = name, block, 5.0
        accepted = True
        if action != KEEP:
            self.stats["suggested"] += 1
            accepted = self.rng.random() < willingness * (0.9 if action == SHIFT else 0.7)
            self.stats["accepted"] += accepted
        if accepted and action == SHIFT:
            new_block, enjoy = block + 1, 4.6
        elif accepted and action == SUBSTITUTE:
            place = self.alt[name]
            orig, alt = self.city["by_name"][name], self.city["by_name"][place]
            enjoy = 5.0 * 0.85 * alt["rating"] / orig["rating"]
        self.load[place][new_block] += GROUP_LOAD
        ratio = self.load[place][new_block]
        reward = enjoy - 8.0 * max(0.0, ratio - 0.8)
        if self.city["by_name"][place]["popularity"] < 0.5:
            reward += 0.5
        if not accepted:
            reward -= 0.3
        self.stats["reward"] += reward
        self.stats["decisions"] += 1
        self.stats["overcrowded"] += ratio > 1.0
        self.stats["enjoyment"] += enjoy
        self.i += 1
        if self.done():
            return None, reward, True
        return self.state(), reward, False


# ------------------------------------------------------------------- agents
class TabularAgent:
    """Q-table agent. method = 'q_learning' or 'sarsa'."""

    def __init__(self, method="q_learning", gamma=0.5, min_alpha=0.01, seed=0):
        self.method, self.gamma, self.min_alpha = method, gamma, min_alpha
        self.q = {}
        self.visits = {}
        self.rng = random.Random(seed)

    def Q(self, s, a):
        return self.q.get((s, a), 0.0)

    def act(self, s, valid, eps=0.0):
        """Epsilon-greedy: explore with probability eps, otherwise take the best known action."""
        if self.rng.random() < eps:
            return self.rng.choice(valid)
        return max(valid, key=lambda a: self.Q(s, a))

    def update(self, s, a, r, s2, a2, valid2):
        if s2 is None:                                     # end of the episode
            target = r
        elif self.method == "q_learning":                  # best next action
            target = r + self.gamma * max(self.Q(s2, b) for b in valid2)
        else:                                              # SARSA: the next action actually taken
            target = r + self.gamma * self.Q(s2, a2)
        self.visits[(s, a)] = self.visits.get((s, a), 0) + 1
        alpha = max(self.min_alpha, 1.0 / self.visits[(s, a)] ** 0.6)   # learning rate: big at first, then smaller
        self.q[(s, a)] = self.Q(s, a) + alpha * (target - self.Q(s, a))


def train(env, method, episodes=400, seed=0):
    agent = TabularAgent(method, seed=seed)
    curve = []
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.7 * episodes))       # explore a lot first, then exploit
        s = env.reset(seed=1000 + ep)
        a = agent.act(s, env.valid_actions(s), eps)
        while True:
            s2, r, done = env.step(a)
            valid2 = env.valid_actions(s2) if not done else []
            a2 = agent.act(s2, valid2, eps) if not done else None
            agent.update(s, a, r, s2, a2, valid2)
            if done:
                break
            s, a = s2, a2
        curve.append(env.stats["reward"] / env.stats["decisions"])
    return agent, curve


# fixed comparison policies: (state, valid actions, rng) -> action
def keep_policy(s, valid, rng):
    return KEEP


def random_policy(s, valid, rng):
    return rng.choice(valid)


def rule_policy(s, valid, rng):
    """Hand-written rule: if it is very crowded, substitute (or shift); otherwise keep."""
    if s[0] == 2:
        return SUBSTITUTE if SUBSTITUTE in valid else (SHIFT if SHIFT in valid else KEEP)
    return KEEP


def evaluate(env, policy, episodes=60):
    rng = random.Random(7)
    total = {"reward": 0.0, "decisions": 0, "suggested": 0, "accepted": 0, "overcrowded": 0, "enjoyment": 0.0}
    for ep in range(episodes):
        s = env.reset(seed=50000 + ep)
        while not env.done():
            s, _, _ = env.step(policy(s, env.valid_actions(s), rng))
        for k in total:
            total[k] += env.stats[k]
    n = total["decisions"]
    return {"reward_per_decision": total["reward"] / n, "overcrowded_share": total["overcrowded"] / n,
            "enjoyment": total["enjoyment"] / n,
            "acceptance": total["accepted"] / total["suggested"] if total["suggested"] else 0.0}


def compare_policies(city, episodes=400):
    env = CrowdEnv(city)
    q_agent, q_curve = train(env, "q_learning", episodes, seed=1)
    s_agent, s_curve = train(env, "sarsa", episodes, seed=2)
    policies = {
        "always keep": keep_policy,
        "random": random_policy,
        "simple rule": rule_policy,
        "Q-learning": lambda s, v, rng: q_agent.act(s, v),
        "SARSA": lambda s, v, rng: s_agent.act(s, v),
    }
    results = {name: evaluate(env, pol) for name, pol in policies.items()}
    return q_agent, results, {"Q-learning": q_curve, "SARSA": s_curve}


def suggest(agent, crowd_dist, minute, has_alt):
    """Used by main.py on a real itinerary: crowd_dist comes from the Bayesian network (module 3)."""
    level = max(range(3), key=lambda i: crowd_dist[["low", "medium", "high"][i]])
    s = (level, block_index(minute), int(has_alt))
    valid = [KEEP] + ([SHIFT] if s[1] < 3 else []) + ([SUBSTITUTE] if has_alt else [])
    return ACTIONS[agent.act(s, valid)]
