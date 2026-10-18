import math
import unittest

from tests.helpers import city
from tourism.m3_uncertainty.forecaster import DayForecast
from tourism.m3_uncertainty.online import OnlineCrowdUpdater, _gammainc_lower_reg


class OnlineUpdateTest(unittest.TestCase):
    def setUp(self):
        self.city = city()
        P, S = len(self.city.pois), self.city.n_slots
        self.fc = DayForecast([[0.5] * S for _ in range(P)], [[0.0] * S for _ in range(P)], [[0.0] * S for _ in range(P)])

    def test_incomplete_gamma(self):
        for x in (0.1, 1.0, 3.0, 10.0):
            self.assertAlmostEqual(_gammainc_lower_reg(1.0, x), 1 - math.exp(-x), places=7)

    def test_conjugate_update(self):
        up = OnlineCrowdUpdater(self.city, self.fc, prior_strength=4.0, detection=0.5, forget=1.0)
        lam = up.expected_arrivals(0, 10)
        up.observe(0, 10, 30)
        self.assertAlmostEqual(up.alpha[0], 34.0)
        self.assertAlmostEqual(up.beta[0], 4.0 + 0.5 * lam)

    def test_detects_a_surge_but_not_a_normal_day(self):
        up = OnlineCrowdUpdater(self.city, self.fc, detection=0.6)
        for s in range(8, 16):
            lam = up.expected_arrivals(0, s)
            up.observe(0, s, int(round(2.0 * 0.6 * lam)))   # twice the expected crowd
            up.observe(1, s, int(round(1.0 * 0.6 * up.expected_arrivals(1, s))))
        self.assertIn(0, up.surges())
        self.assertNotIn(1, up.surges())
        self.assertGreater(up.updated_forecast(16).load[0][20], self.fc.load[0][20] * 1.5)


if __name__ == "__main__":
    unittest.main()
