"""
GENERATE DATA - builds every dataset the planner needs.

    python3 generate_data.py              write all files into data/
    python3 generate_data.py --seed 8     a different (but equally valid) city and history

main.py and test.py run this automatically when a file is missing.

Files it writes:
    data/city.json            the city: 73 places (35 sights, 10 local businesses, 14 restaurants,
                              6 hotels, 8 bus stations) in 8 districts, walking links, 4 bus lines,
                              and the speed, CO2 and fare of walking, bus and taxi
    data/history.csv          120 days of hourly visitor counts at every place (65,040 rows).
                              Module 3 learns its Bayesian network from this file.
    data/transit_history.csv  120 days of hourly delays on every bus line (6,240 rows)
    data/travellers.json      300 synthetic travellers: what they tell the planner, and how willing
                              they are (secretly) to accept a suggestion

Everything is synthetic, so the project runs anywhere without downloads, but it is shaped like
real open data (places with opening hours and prices, a street network, bus lines, visitor logs).
The visitor numbers come from a hidden "true" model (the true_* functions below). Module 3 never
sees this model, only the data, just like a real system that only has visitor logs.
"""
import argparse
import csv
import json
import math
import os
import random

from module1_search_csp import interest_utility
from module3_uncertainty import block_of, fame_of, setting_of

DATA_DIR = "data"
CITY_FILE = os.path.join(DATA_DIR, "city.json")
HISTORY_FILE = os.path.join(DATA_DIR, "history.csv")
TRANSIT_FILE = os.path.join(DATA_DIR, "transit_history.csv")
TRAVELLERS_FILE = os.path.join(DATA_DIR, "travellers.json")
ALL_FILES = [CITY_FILE, HISTORY_FILE, TRANSIT_FILE, TRAVELLERS_FILE]

DAYS = 120
HOURS = range(8, 21)                    # one row per hour, 08:00 ... 20:00
WALK_LINK_KM = 1.2                      # places closer than this are joined by a footpath

# ------------------------------------------------------------------ the city
# district name, centre x, centre y (km)
ZONES = [("Old Town", 5.0, 4.0), ("Riverside", 6.6, 3.1), ("Museum Quarter", 3.7, 4.9),
         ("Market District", 5.7, 5.3), ("Lakeside", 2.1, 2.4), ("Hill Park", 7.9, 6.3),
         ("Craft Village", 8.7, 1.7), ("University Town", 1.7, 6.5)]
HILLY_ZONE = 5                          # steep paths: not every footpath here suits a wheelchair

# name, zone, themes, popularity, rating, visitors per hour it can take, minutes, price, indoor,
# opens, closes, needs a ticket, wheelchair accessible
ATTRACTIONS = [
    ("Royal Fort", 0, {"history": 1, "architecture": .8}, .98, 4.7, 560, 105, 400, False, "09:00", "18:00", True, True),
    ("Old Palace", 0, {"history": .9, "architecture": .9, "art": .3}, .90, 4.6, 440, 90, 350, True, "09:30", "17:30", True, True),
    ("City History Museum", 2, {"history": 1, "art": .4, "family": .3}, .75, 4.5, 360, 90, 200, True, "10:00", "18:00", True, True),
    ("Riverside Ruins", 1, {"history": .9, "nature": .4, "architecture": .4}, .40, 4.4, 280, 75, 100, False, "09:00", "18:30", True, True),
    ("Hill Fortress", 5, {"history": .85, "nature": .5, "architecture": .5}, .35, 4.5, 240, 90, 120, False, "09:00", "17:30", True, False),
    ("Heritage Walk House", 6, {"history": .8, "crafts": .5}, .20, 4.4, 140, 60, 80, True, "10:00", "18:00", False, True),
    ("National Art Gallery", 2, {"art": 1, "architecture": .3}, .85, 4.6, 400, 90, 300, True, "10:00", "18:00", True, True),
    ("Modern Art Hall", 2, {"art": .95}, .55, 4.4, 280, 75, 200, True, "10:30", "18:30", True, True),
    ("Street Art Lane", 3, {"art": .8, "shopping": .3}, .45, 4.3, 320, 45, 0, False, "08:00", "21:00", False, True),
    ("University Art Collection", 7, {"art": .9, "history": .3}, .20, 4.5, 160, 60, 50, True, "10:00", "17:30", False, True),
    ("Riverside Sculpture Garden", 1, {"art": .7, "nature": .6}, .35, 4.4, 240, 60, 0, False, "09:00", "19:00", False, True),
    ("Botanical Garden", 4, {"nature": 1, "family": .6}, .80, 4.6, 600, 90, 100, False, "08:00", "18:00", True, True),
    ("Lakeside Promenade", 4, {"nature": .8, "family": .5}, .60, 4.5, 800, 60, 0, False, "08:00", "21:00", False, True),
    ("Hill Park Viewpoint", 5, {"nature": .9, "architecture": .2}, .70, 4.7, 480, 60, 0, False, "08:00", "20:00", False, False),
    ("Butterfly Reserve", 6, {"nature": .9, "family": .7}, .30, 4.5, 200, 75, 150, False, "09:00", "17:30", True, True),
    ("University Arboretum", 7, {"nature": .85, "family": .3}, .20, 4.4, 320, 60, 0, False, "08:00", "19:00", False, True),
    ("River Boat Ride", 1, {"nature": .6, "family": .7}, .75, 4.5, 240, 45, 250, False, "10:00", "19:00", True, True),
    ("Grand Temple", 0, {"spiritual": 1, "architecture": .8, "history": .6}, .95, 4.8, 640, 60, 0, False, "08:00", "21:00", False, True),
    ("Riverside Shrine", 1, {"spiritual": .9, "architecture": .5, "history": .4}, .40, 4.6, 280, 45, 0, False, "08:00", "20:00", False, True),
    ("Hill Monastery", 5, {"spiritual": .95, "nature": .4, "history": .5}, .30, 4.6, 200, 60, 0, True, "09:00", "18:00", False, False),
    ("Lotus Meditation Centre", 4, {"spiritual": .8, "nature": .3}, .25, 4.4, 160, 60, 50, True, "09:00", "19:00", False, True),
    ("Clock Tower", 0, {"architecture": .9, "history": .5}, .85, 4.4, 360, 30, 50, False, "09:00", "19:00", True, True),
    ("Old Bridge", 1, {"architecture": .8, "history": .6}, .60, 4.4, 600, 30, 0, False, "08:00", "21:00", False, True),
    ("Colonial Library", 2, {"architecture": .85, "history": .6, "art": .3}, .35, 4.5, 200, 45, 0, True, "10:00", "18:00", False, True),
    ("Merchant Mansions", 3, {"architecture": .8, "history": .5, "shopping": .2}, .30, 4.3, 240, 45, 60, True, "10:00", "18:00", False, True),
    ("Science Centre", 2, {"family": 1, "art": .2}, .70, 4.5, 480, 120, 350, True, "10:00", "18:00", True, True),
    ("City Zoo", 4, {"family": 1, "nature": .7}, .85, 4.4, 720, 120, 300, False, "09:00", "17:30", True, True),
    ("Adventure Park", 5, {"family": .9, "nature": .5}, .45, 4.3, 320, 120, 400, False, "10:00", "18:00", True, False),
    ("Toy and Puppet Museum", 3, {"family": .85, "crafts": .4, "art": .3}, .25, 4.4, 200, 60, 100, True, "10:00", "18:00", False, True),
    ("Grand Bazaar", 3, {"shopping": 1, "food": .5, "crafts": .4}, .90, 4.4, 800, 75, 0, False, "10:00", "21:00", False, True),
    ("Spice Market", 3, {"food": .9, "shopping": .6}, .65, 4.5, 480, 45, 0, False, "09:00", "19:00", False, True),
    ("Night Food Street", 0, {"food": 1, "shopping": .3}, .70, 4.5, 600, 60, 0, False, "17:00", "21:00", False, True),
    ("Riverside Food Market", 1, {"food": .9, "shopping": .3}, .35, 4.5, 360, 45, 0, False, "11:00", "20:00", False, True),
    ("Weavers' Hall", 6, {"crafts": 1, "history": .4, "shopping": .4}, .25, 4.6, 160, 60, 50, True, "10:00", "18:00", False, True),
    ("Pottery Courtyard", 6, {"crafts": .95, "art": .5}, .20, 4.5, 120, 60, 0, False, "10:00", "18:00", False, True),
]

# Independent local businesses: name, zone, indoor shop?, themes, rating, visitors per hour, minutes, price, opens, closes
LOCAL = [
    ("Handloom Co-op Store", 6, True, {"shopping": .8, "crafts": .9}, 4.6, 100, 30, 0, "10:00", "19:00"),
    ("Village Cooking Class", 6, False, {"food": 1, "crafts": .4}, 4.8, 60, 90, 600, "10:00", "17:00"),
    ("Riverside Bookshop Cafe", 1, True, {"art": .4, "food": .5, "shopping": .5}, 4.5, 80, 30, 150, "09:00", "20:00"),
    ("Hill Tea Garden", 5, False, {"nature": .6, "food": .6}, 4.6, 100, 45, 120, "09:00", "18:00"),
    ("Guided Lane Walk", 3, False, {"history": .7, "architecture": .6, "food": .3}, 4.7, 80, 75, 250, "09:00", "18:00"),
    ("Artisan Jewellery Studio", 3, True, {"crafts": .9, "shopping": .7, "art": .4}, 4.5, 60, 30, 0, "10:00", "20:00"),
    ("Lakeside Kayak Rentals", 4, False, {"nature": .8, "family": .6}, 4.5, 80, 60, 300, "09:00", "18:00"),
    ("Student Theatre Matinee", 7, True, {"art": .8, "family": .3}, 4.4, 160, 90, 150, "14:00", "18:30"),
    ("Old Town Pottery Workshop", 0, True, {"crafts": 1, "art": .5}, 4.6, 60, 60, 300, "10:00", "19:00"),
    ("Print Studio", 2, True, {"art": .8, "crafts": .6}, 4.4, 60, 30, 0, "10:00", "19:00"),
]

# name, zone, locally owned, popularity, rating, seats, price of a meal
RESTAURANTS = [
    ("Fort View Diner", 0, False, .80, 4.0, 60, 600), ("Burger Plaza", 0, False, .70, 3.8, 80, 400),
    ("Temple Street Thali", 0, True, .75, 4.6, 50, 350), ("Gallery Cafe", 2, False, .60, 4.2, 40, 450),
    ("Pizza Corner", 2, False, .50, 3.9, 50, 450), ("Riverside Fish House", 1, True, .50, 4.6, 40, 500),
    ("Spice Route Bistro", 1, True, .40, 4.5, 35, 450), ("Market Biryani House", 3, True, .70, 4.7, 50, 300),
    ("Food Court Central", 3, False, .60, 3.9, 100, 350), ("Lakeside Grill", 4, False, .50, 4.1, 60, 550),
    ("Garden Kitchen", 4, True, .30, 4.5, 30, 400), ("Hilltop Dhaba", 5, True, .30, 4.6, 30, 250),
    ("Craft Village Kitchen", 6, True, .20, 4.7, 25, 250), ("Campus Canteen Co-op", 7, True, .20, 4.3, 40, 150),
]

HOTELS = [("Central Hotel", 0), ("Riverside Inn", 1), ("Museum Suites", 2), ("Lakeside Resort", 4),
          ("Hill Lodge", 5), ("Campus Guesthouse", 7)]

# bus line name, districts it connects (in order)
LINES = [("Bus 1 (Blue)", [4, 2, 0, 3, 5]), ("Bus 2 (Red)", [7, 2, 0, 1, 6]),
         ("Bus 3 (Green)", [4, 0, 1, 5]), ("Bus 4 (Yellow)", [6, 1, 3, 7])]

TRANSPORT = {
    "walk": {"speed_kmh": 4.5, "co2_g_per_km": 0, "fare_per_km": 0, "fare_per_ride": 0, "stop_wait_min": 0},
    "bus": {"speed_kmh": 22, "co2_g_per_km": 70, "fare_per_km": 0, "fare_per_ride": 5, "stop_wait_min": 2},
    "taxi": {"speed_kmh": 25, "co2_g_per_km": 160, "fare_per_km": 20, "fare_per_ride": 0, "stop_wait_min": 0},
    "street_detour": 1.3,
    "taxi_link_km": 3.0,
}


def themes_to_categories(themes):
    """{"history": 1, "art": .3} -> ["history", "art"] (strongest first, weak ones dropped)."""
    return [t for t, w in sorted(themes.items(), key=lambda kv: -kv[1]) if w >= 0.3]


def make_city(seed=7):
    rng = random.Random(seed)
    places = []

    def spot(zone, spread):
        _, cx, cy = ZONES[zone]
        return round(cx + rng.gauss(0, spread), 2), round(cy + rng.gauss(0, spread), 2)

    for (name, zone, themes, pop, rating, cap, dur, price, indoor, op, cl, ticket, acc) in ATTRACTIONS:
        x, y = spot(zone, 0.48)
        places.append({"name": name, "type": "attraction", "zone": ZONES[zone][0], "x": x, "y": y,
                       "categories": themes_to_categories(themes), "popularity": pop, "rating": rating,
                       "capacity_per_hour": cap, "duration": dur, "price": price, "open": op, "close": cl,
                       "indoor": indoor, "accessible": acc, "ticketed": ticket})
    for (name, zone, shop, themes, rating, cap, dur, price, op, cl) in LOCAL:
        x, y = spot(zone, 0.42)
        places.append({"name": name, "type": "local", "zone": ZONES[zone][0], "x": x, "y": y,
                       "categories": themes_to_categories(themes), "popularity": round(0.15 + 0.1 * rng.random(), 2),
                       "rating": rating, "capacity_per_hour": cap, "duration": dur, "price": price, "open": op,
                       "close": cl, "indoor": shop, "accessible": zone != HILLY_ZONE, "ticketed": False})
    for (name, zone, local, pop, rating, seats, price) in RESTAURANTS:
        x, y = spot(zone, 0.42)
        places.append({"name": name, "type": "restaurant", "zone": ZONES[zone][0], "x": x, "y": y,
                       "categories": ["food"], "popularity": pop, "rating": rating, "capacity_per_hour": 2 * seats,
                       "duration": 50, "price": price, "open": "11:30", "close": "21:00", "indoor": True,
                       "accessible": True, "ticketed": False, "local_owned": local})
    for (name, zone) in HOTELS:
        x, y = spot(zone, 0.35)
        places.append({"name": name, "type": "hotel", "zone": ZONES[zone][0], "x": x, "y": y})
    for zname, cx, cy in ZONES:
        places.append({"name": zname + " Station", "type": "stop", "zone": zname,
                       "x": round(cx + rng.gauss(0, 0.06), 2), "y": round(cy + rng.gauss(0, 0.06), 2)})

    def km(a, b):
        return math.hypot(a["x"] - b["x"], a["y"] - b["y"])

    # Footpaths: every pair closer than WALK_LINK_KM, and at least the 2 nearest neighbours.
    pairs = set()
    for i, a in enumerate(places):
        nearest = sorted((j for j in range(len(places)) if j != i), key=lambda j: km(a, places[j]))
        for k, j in enumerate(nearest):
            if km(a, places[j]) <= WALK_LINK_KM or k < 2:
                pairs.add((min(i, j), max(i, j)))
    # Join separate pieces with the shortest missing footpath until everything is connected.
    while True:
        groups = list(range(len(places)))

        def root(i):
            while groups[i] != i:
                i = groups[i]
            return i
        for i, j in pairs:
            groups[root(i)] = root(j)
        roots = {root(i) for i in range(len(places))}
        if len(roots) == 1:
            break
        main_group = root(0)
        i, j = min(((i, j) for i in range(len(places)) for j in range(len(places))
                    if root(i) == main_group and root(j) != main_group), key=lambda p: km(places[p[0]], places[p[1]]))
        pairs.add((min(i, j), max(i, j)))
    hilly = ZONES[HILLY_ZONE][0]
    walk_links = []
    for i, j in sorted(pairs):
        steep = (places[i]["zone"] == hilly or places[j]["zone"] == hilly) and rng.random() < 0.5
        walk_links.append([places[i]["name"], places[j]["name"], steep])

    return {
        "name": "Demo City",
        "note": "Generated by generate_data.py. x and y are kilometres. Times are 24-hour HH:MM.",
        "seed": seed,
        "zones": [z[0] for z in ZONES],
        "transport": TRANSPORT,
        "places": places,
        "walk_links": walk_links,
        "bus_lines": {name: [ZONES[z][0] + " Station" for z in seq] for name, seq in LINES},
    }


# ------------------------------------------------------------ hidden "true" model
DEMAND = 2.0                            # how busy the city is overall
DAY_FACTOR = {"weekday": 0.7, "weekend": 1.05}
SEASON_FACTOR = {"off": 0.75, "peak": 1.1}


def true_hour_profile(place, hour):
    """How busy a place usually is at this hour, from 0 to 1."""
    h = hour + 0.5
    if place["open"] >= "16:00":                                         # evening venues
        return 0.2 + 0.8 * math.exp(-((h - 19.5) / 1.3) ** 2)
    if place["type"] == "attraction":
        v = 0.22 + 0.78 * math.exp(-((h - 11.75) / 1.6) ** 2) + 0.55 * math.exp(-((h - 15.5) / 1.4) ** 2)
        return min(1.0, v * (max(0.3, 1 - (h - 18) * 0.25) if h > 18 else 1))
    if place["type"] == "restaurant":
        return min(1.0, 0.12 + 0.88 * math.exp(-((h - 13.0) / 0.8) ** 2) + 0.75 * math.exp(-((h - 19.75) / 0.9) ** 2))
    return min(1.0, 0.3 + 0.5 * math.exp(-((h - 17) / 2.0) ** 2) + 0.25 * math.exp(-((h - 12) / 1.5) ** 2))


def true_load(rng, place, day, season, weather, hour, event):
    """Visitors divided by capacity. Above 1 the place is over-full and a queue builds."""
    base = 0.12 + 0.78 * place["popularity"] ** 1.6
    rain = 1.0 if weather == "clear" else (1.35 if place["indoor"] else 0.5)
    near_event = 1.6 if event and event["zone"] == place["zone"] and event["from"] <= hour < event["to"] else 1.0
    return (DEMAND * base * DAY_FACTOR[day] * SEASON_FACTOR[season] * rain * true_hour_profile(place, hour)
            * near_event * math.exp(rng.gauss(0, 0.15)))


def queue_waits(arrivals, capacity, slot_min=60):
    """Fluid queue: people who cannot get in during one hour wait into the next.
    Returns the average wait (minutes) in each hour."""
    q, waits = 0.0, []
    for a in arrivals:
        waits.append(slot_min * max(0.0, q + 0.5 * a - 0.5 * capacity) / capacity)
        q = max(0.0, q + a - capacity)
    return waits


def crowd_label(load):
    return "low" if load < 0.55 else ("medium" if load < 0.85 else "high")


def wait_label(minutes):
    return "short" if minutes < 5 else ("medium" if minutes < 20 else "long")


def sample_day(rng, d, zones):
    day = "weekday" if d % 7 < 5 else "weekend"
    season = ["off", "peak"][(d // 30) % 2]                        # 30 days off-season, 30 days peak, ...
    weather = "rain" if rng.random() < (0.25 if season == "off" else 0.15) else "clear"
    event = None
    if rng.random() < 0.15:                                        # a public event somewhere in town
        start, end = rng.choice([(10, 14), (16, 21), (17, 21)])
        event = {"zone": rng.choice(zones), "from": start, "to": end}
    return day, season, weather, event


def make_history(city, seed=11):
    rng = random.Random(seed)
    places = [p for p in city["places"] if p["type"] in ("attraction", "local", "restaurant")]
    days, rows, transit = [], [], []
    for d in range(DAYS):
        day, season, weather, event = sample_day(rng, d, city["zones"])
        days.append((day, season, weather, event))
        for p in places:
            open_hours = [h for h in HOURS if p["open"] <= "%02d:00" % h < p["close"]]
            loads = [true_load(rng, p, day, season, weather, h, event) for h in open_hours]
            visitors = [round(load * p["capacity_per_hour"]) for load in loads]
            queues = queue_waits(visitors, p["capacity_per_hour"])
            for h, load, v, q in zip(open_hours, loads, visitors, queues):
                wait = round(q + 40 * max(0.0, load - 0.5) ** 1.5, 1)       # queue + crowding at the entrance
                rows.append([d, day, season, weather, h, block_of(h * 60), p["name"], fame_of(p), setting_of(p),
                             event["zone"] if event else "", v, crowd_label(load), wait, wait_label(wait)])
        for line, stations in city["bus_lines"].items():
            line_zones = {s.replace(" Station", "") for s in stations}
            for h in HOURS:
                peak = h in (8, 9, 17, 18)
                on_line = bool(event and event["zone"] in line_zones and event["from"] <= h < event["to"])
                mult = (2.0 if weather == "rain" else 1.0) * (1.5 if on_line else 1.0) * (1.5 if peak else 1.0)
                u = rng.random()
                if u < min(0.25, 0.02 * mult):
                    delay = rng.uniform(15, 30)                                  # major delay
                elif u < min(0.25, 0.02 * mult) + min(0.5, 0.08 * mult):
                    delay = rng.uniform(4, 10)                                   # minor delay
                else:
                    delay = 0.0
                delay = round(delay, 1)
                transit.append([d, line, h, weather, int(peak), int(on_line), delay, "late" if delay >= 5 else "on_time"])
    return rows, transit


HISTORY_COLUMNS = ["day", "Day", "Season", "Weather", "hour", "Block", "place", "Fame", "Setting", "event_zone",
                   "visitors", "Crowd", "wait_min", "Wait"]
TRANSIT_COLUMNS = ["day", "line", "hour", "Weather", "peak", "event_on_line", "delay_min", "BusDelay"]


# ------------------------------------------------------------------ travellers
ARCHETYPES = {
    "culture": {"history": .9, "art": .7, "architecture": .8, "spiritual": .4, "crafts": .3, "food": .3},
    "foodie": {"food": 1, "shopping": .5, "crafts": .5, "history": .3, "art": .2},
    "nature": {"nature": 1, "family": .4, "spiritual": .4, "food": .3},
    "family": {"family": 1, "nature": .6, "history": .3, "food": .4, "art": .2},
    "shopper": {"shopping": 1, "crafts": .7, "food": .5, "art": .3, "architecture": .2},
    "spiritual": {"spiritual": 1, "history": .6, "architecture": .6, "nature": .3},
    "explorer": {},
}
THEMES = ["history", "art", "nature", "food", "shopping", "spiritual", "architecture", "family", "crafts"]
WILLINGNESS = {"low": (2.0, 6.0), "medium": (4.0, 4.0), "high": (6.0, 2.0)}       # Beta(a, b)


def make_travellers(city, n=300, seed=4242):
    rng = random.Random(seed)
    hotels = [p["name"] for p in city["places"] if p["type"] == "hotel"]
    famous = [p for p in city["places"] if p["type"] == "attraction" and p["popularity"] >= 0.8]
    out = []
    for k in range(n):
        kind = rng.choice(list(ARCHETYPES))
        interests = {}
        for t in THEMES:
            v = ARCHETYPES[kind].get(t, 0.5 if kind == "explorer" else 0.0) + rng.gauss(0, 0.12)
            if v >= 0.25:
                interests[t] = round(min(1.0, v), 2)
        wheelchair = rng.random() < 0.08
        taxi_ok = rng.random() < 0.6 and not wheelchair
        level = rng.choice(list(WILLINGNESS))
        profile = {
            "name": "%s traveller %d" % (kind.capitalize(), k + 1), "archetype": kind,
            "hotel": rng.choice(hotels), "start": rng.choice(["09:00", "09:30", "10:00", "10:30"]),
            "end": rng.choice(["18:00", "18:30", "19:00", "19:30", "20:00"]),
            "budget": rng.choice([1500, 2500, 3500, 5000, 7000]), "interests": interests or {"history": 0.5},
            "max_places": rng.choice([3, 4, 4, 5]), "must_visit": [], "accessible_only": wheelchair,
            "lunch": rng.random() < 0.9, "modes": ["walk", "bus", "taxi"] if taxi_ok else ["walk", "bus"],
            "willingness": level, "flexibility": round(rng.betavariate(*WILLINGNESS[level]), 3),
        }
        if rng.random() < 0.3:                                    # some insist on one famous sight
            options = [p for p in famous if p["accessible"] or not wheelchair]
            profile["must_visit"] = [max(options, key=lambda p: interest_utility(p, profile))["name"]]
        out.append(profile)
    return out


# ------------------------------------------------------------------ writing files
def generate_all(seed=7, quiet=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    city = make_city(seed)
    with open(CITY_FILE, "w", encoding="utf-8") as f:
        json.dump(city, f, indent=1)
    rows, transit = make_history(city, seed + 4)
    with open(HISTORY_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HISTORY_COLUMNS)
        w.writerows(rows)
    with open(TRANSIT_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TRANSIT_COLUMNS)
        w.writerows(transit)
    travellers = make_travellers(city, seed=seed + 4235)
    with open(TRAVELLERS_FILE, "w", encoding="utf-8") as f:
        json.dump(travellers, f, indent=1)
    if not quiet:
        kinds = {}
        for p in city["places"]:
            kinds[p["type"]] = kinds.get(p["type"], 0) + 1
        print("data/city.json            %d places (%s), %d footpaths, %d bus lines" % (
            len(city["places"]), ", ".join("%d %s" % (n, k) for k, n in kinds.items()), len(city["walk_links"]),
            len(city["bus_lines"])))
        print("data/history.csv          %d rows (%d days x opening hours x places)" % (len(rows), DAYS))
        print("data/transit_history.csv  %d rows" % len(transit))
        print("data/travellers.json      %d travellers" % len(travellers))
    return city


def ensure_data():
    """Generate the data files if any of them is missing."""
    if not all(os.path.exists(f) for f in ALL_FILES):
        print("Generating the datasets in data/ (first run only) ...")
        generate_all()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the city and all datasets")
    ap.add_argument("--seed", type=int, default=7)
    generate_all(ap.parse_args().seed)
