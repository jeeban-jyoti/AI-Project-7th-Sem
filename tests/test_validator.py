import unittest

from tests.helpers import city
from tourism.coordinator.validator import validate_plan
from tourism.core.types import TravellerProfile


class ValidatorTest(unittest.TestCase):
    def test_catches_violations(self):
        c = city()
        hotel = c.hotels[0]
        fort = next(p for p in c.pois if p["name"] == "Royal Fort")
        prof = TravellerProfile("X", "x", {}, budget=100, start_node=hotel, end_node=hotel, start_min=540, end_min=600,
                                max_walk_m=500, must_visit=[c.pois[1]["node"]], modes=["walk", "transit"])
        plan = {"actions": [
            {"type": "travel", "from": hotel, "to": fort["node"], "start": 540, "end": 560, "money": 50, "walk_m": 900,
             "segments": [{"mode": "car", "from": hotel, "to": fort["node"], "dist_m": 900}]},
            {"type": "visit", "poi": fort["node"], "start": 500, "end": 1100, "money": 400},
            {"type": "travel", "from": fort["node"], "to": hotel, "start": 1100, "end": 1120, "money": 0, "walk_m": 0,
             "segments": []}]}
        v = " | ".join(validate_plan(c, prof, plan))
        for expected in ("car", "opening hours", "starts before", "after", "budget", "walking", "must-visit"):
            self.assertIn(expected, v)


if __name__ == "__main__":
    unittest.main()
