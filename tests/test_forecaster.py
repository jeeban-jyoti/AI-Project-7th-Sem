import os
import unittest

from tests.helpers import HISTORY_PATH, city
from tourism.m3_uncertainty.forecaster import CrowdForecaster, evaluate_forecasters
from tourism.sim.truth import DayContext


@unittest.skipUnless(os.path.exists(HISTORY_PATH), "run ./run.sh data first")
class ForecasterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.city = city()
        cls.fc = CrowdForecaster.fit(cls.city, HISTORY_PATH, HISTORY_PATH.replace("history.csv.gz", "transit_history.csv"), 100)

    def test_distributions_are_normalised(self):
        d = self.fc.forecast_day(DayContext("weekend", "peak", "clear", rain_prob=0.3))
        for row in d.probs[:5]:
            for p in row:
                self.assertAlmostEqual(sum(p), 1.0, places=6)
        self.assertAlmostEqual(sum(self.fc.delay_probs("rain", True, True)), 1.0, places=6)

    def test_context_matters(self):
        busy = self.fc.forecast_day(DayContext("weekend", "peak", "clear"))
        quiet = self.fc.forecast_day(DayContext("weekday", "off", "rain"))
        self.assertGreater(sum(map(sum, busy.load)), 1.5 * sum(map(sum, quiet.load)))

    def test_beats_historical_average(self):
        ev = evaluate_forecasters(self.city, HISTORY_PATH, 100, {"bayes": self.fc, "hist": self.fc.baseline})
        self.assertLess(ev["bayes"]["mae_load"], ev["hist"]["mae_load"])
        self.assertLess(ev["bayes"]["brier"], ev["hist"]["brier"])


if __name__ == "__main__":
    unittest.main()
