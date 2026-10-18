import random
import unittest

from tests.helpers import city, engine
from tourism.coordinator.validator import validate_plan
from tourism.core.config import load_config
from tourism.data.personas import generate_travellers
from tourism.m1_search.candidates import select_candidates


class EngineIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.city = city()
        cls.engine = engine()
        cls.cfg = load_config()
        P, S = len(cls.city.pois), cls.city.n_slots
        zeros = [[0.0] * S for _ in range(P)]
        cls.engine.set_context("t", zeros, zeros, zeros, {}, True)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def test_astar_is_optimal_and_cheaper_than_ucs(self):
        rng = random.Random(3)
        nodes = [p["node"] for p in self.city.pois]
        w = {"time": 1.0, "money": 0.01, "co2": 1.0, "transfer": 2.0}
        for _ in range(25):
            a, b = rng.sample(nodes, 2)
            u = self.engine.route(a, b, 600, ["walk", "transit", "car"], w, "ucs")
            s = self.engine.route(a, b, 600, ["walk", "transit", "car"], w, "astar")
            self.assertAlmostEqual(u["cost"], s["cost"], places=6)
            self.assertLessEqual(s["expanded"], u["expanded"])

    def test_csp_and_planner_produce_valid_plans(self):
        strat = self.cfg["strategies"]["sustainable"]
        for profile, _ in generate_travellers(self.city, 6, 11):
            cands = select_candidates(profile, self.city, self.cfg["planning"])
            sol = self.engine.call("solve", context="t", profile=profile.to_engine(1.6), candidates=cands,
                                   weights=strat["weights"], options=self.cfg["planning"]["csp"])
            self.assertEqual(sol["status"], "ok")
            goals = [c for c in cands if c["poi"] in {a["poi"] for a in sol["activities"]}]
            plan = self.engine.call("plan", context="t", profile=profile.to_engine(1.6), weights=strat["weights"],
                                    goals=goals, hint_order=[a["poi"] for a in sol["activities"]],
                                    state={"node": profile.start_node, "time": profile.start_min, "budget": profile.budget})
            self.assertEqual(validate_plan(self.city, profile, plan), [], profile.id)

    def test_impossible_request_reports_conflict(self):
        profile = generate_travellers(self.city, 1, 5)[0][0]
        night = next(p for p in self.city.pois if p["name"] == "Night Food Street")
        profile.must_visit = [night["node"]]
        profile.end_min = 900
        cands = select_candidates(profile, self.city, self.cfg["planning"])
        sol = self.engine.call("solve", context="t", profile=profile.to_engine(1.6), candidates=cands,
                               weights=self.cfg["strategies"]["sustainable"]["weights"], options=self.cfg["planning"]["csp"])
        self.assertEqual(sol["status"], "unsolvable")
        self.assertTrue(any(c.get("poi") == night["node"] for c in sol["conflicts"]))


if __name__ == "__main__":
    unittest.main()
