"""Shared fixtures: the generated city and a running engine."""
import os

from tourism.core.city import City
from tourism.core.config import ROOT, engine_path, load_config

CITY_PATH = os.path.join(ROOT, "data", "city.json")
HISTORY_PATH = os.path.join(ROOT, "data", "history.csv.gz")


def city():
    if not os.path.exists(CITY_PATH):
        from tourism.data.city_generator import write_city
        write_city(CITY_PATH, load_config()["city"]["seed"])
    return City.load(CITY_PATH)


def engine():
    from tourism.engine.client import EngineClient
    city()
    return EngineClient(engine_path(load_config()), CITY_PATH)
