import unittest

from tourism.sim.simulator import gini_coefficient, satisfaction
from tourism.sim.truth import queue_waits


class SimulationTest(unittest.TestCase):
    def test_queue_builds_and_drains(self):
        w = queue_waits([0, 150, 150, 0, 0], capacity=100, slot_min=15)
        self.assertEqual(w[0], 0.0)
        self.assertGreater(w[2], w[1])          # backlog carries over
        self.assertLess(w[4], w[3])             # and drains

    def test_gini(self):
        self.assertAlmostEqual(gini_coefficient([1, 1, 1, 1]), 0.0)
        self.assertGreater(gini_coefficient([0, 0, 0, 10]), 0.7)

    def test_crowding_reduces_satisfaction(self):
        calm = [{"utility": 8, "rho": 0.6, "wait": 0}]
        packed = [{"utility": 8, "rho": 1.5, "wait": 20}]
        self.assertGreater(satisfaction(calm, 30), satisfaction(packed, 30))

    def test_full_day_runs(self):
        from tourism.eval.workspace import Workspace
        ws = Workspace(need_models=False)
        try:
            m = ws.run_day("S0_normal", "conventional", 1, 15)["metrics"]
        finally:
            ws.close()
        self.assertEqual(m["travellers"], 15)
        self.assertGreaterEqual(m["feasibility"], 0.5)
        self.assertGreater(m["activities"], 1)
        self.assertEqual(m["violations"], 0)


if __name__ == "__main__":
    unittest.main()
