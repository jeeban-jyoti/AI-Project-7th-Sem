import itertools
import random
import unittest

from tourism.m3_uncertainty.bayes_net import BayesianNetwork


def sprinkler():
    bn = BayesianNetwork()
    bn.add_node("Rain", ["no", "yes"])
    bn.add_node("Sprinkler", ["off", "on"], ["Rain"])
    bn.add_node("Wet", ["dry", "wet"], ["Rain", "Sprinkler"])
    bn.set_cpt("Rain", [[0.8, 0.2]])
    bn.set_cpt("Sprinkler", [[0.6, 0.4], [0.99, 0.01]])
    bn.set_cpt("Wet", [[1.0, 0.0], [0.1, 0.9], [0.2, 0.8], [0.01, 0.99]])
    return bn


def joint(bn, r, s, w):
    return bn.cpt_row("Rain", [])[r] * bn.cpt_row("Sprinkler", [r])[s] * bn.cpt_row("Wet", [r, s])[w]


class BayesNetTest(unittest.TestCase):
    def test_variable_elimination_matches_enumeration(self):
        bn = sprinkler()
        post = bn.query("Rain", {"Wet": "wet"})
        num = [sum(joint(bn, r, s, 1) for s in range(2)) for r in range(2)]
        z = sum(num)
        for r in range(2):
            self.assertAlmostEqual(post[r], num[r] / z, places=9)
        self.assertAlmostEqual(sum(bn.query("Sprinkler", {"Wet": 1})), 1.0, places=9)

    def test_soft_evidence_is_a_mixture(self):
        bn = sprinkler()
        p = bn.query("Wet", soft_evidence={"Rain": [0.5, 0.5]})
        # likelihood [0.5, 0.5] on Rain leaves the prior unchanged
        prior = bn.query("Wet")
        for a, b in zip(p, prior):
            self.assertAlmostEqual(a, b, places=9)

    def test_learning_recovers_cpts(self):
        rng = random.Random(1)
        bn = sprinkler()
        rows = []
        for _ in range(20000):
            r = int(rng.random() < 0.2)
            s = int(rng.random() < (0.01 if r else 0.4))
            w = int(rng.random() < bn.cpt_row("Wet", [r, s])[1])
            rows.append((r, s, w))
        learned = BayesianNetwork()
        learned.add_node("Rain", ["no", "yes"])
        learned.add_node("Sprinkler", ["off", "on"], ["Rain"])
        learned.add_node("Wet", ["dry", "wet"], ["Rain", "Sprinkler"])
        learned.fit_counts(["Rain", "Sprinkler", "Wet"], rows)
        self.assertAlmostEqual(learned.cpt_row("Rain", [])[1], 0.2, delta=0.02)
        self.assertAlmostEqual(learned.cpt_row("Sprinkler", [0])[1], 0.4, delta=0.03)
        for row in itertools.product(range(2), range(2)):
            self.assertAlmostEqual(sum(learned.cpt_row("Wet", list(row))), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
