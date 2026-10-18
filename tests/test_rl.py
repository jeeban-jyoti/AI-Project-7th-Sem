import random
import unittest

from tourism.m4_rl.agents import LinUCBAgent, QLearningAgent
from tourism.m4_rl.features import N_STATES, STATE_DIMS, feature_vector, state_index


class RLTest(unittest.TestCase):
    def test_state_index_is_a_bijection(self):
        import itertools
        seen = set()
        for st in itertools.product(*[range(d) for d in STATE_DIMS]):
            seen.add(state_index(st))
        self.assertEqual(len(seen), N_STATES)
        self.assertEqual(min(seen), 0)
        self.assertEqual(max(seen), N_STATES - 1)

    def test_q_learning_finds_best_action(self):
        rng = random.Random(0)
        agent = QLearningAgent(alpha=0.3, gamma=0.0, seed=1)
        means = {0: 0.0, 1: 0.5, 2: -0.3}
        for _ in range(3000):
            a = agent.act(7, (0,) * len(STATE_DIMS), [0, 1, 2], None, eps=0.3)
            agent.update(7, a, means[a] + rng.gauss(0, 0.3), 7, [0, 1, 2], True)
        self.assertEqual(agent.greedy(7, [0, 1, 2]), 1)

    def test_linucb_learns_context(self):
        rng = random.Random(0)
        agent = LinUCBAgent(alpha=0.5)
        a_state = (0, 3, 2, 3, 0, 0, 1)   # crowded -> action 2 is best
        b_state = (0, 0, 0, 0, 0, 0, 0)   # quiet   -> action 0 is best
        for _ in range(400):
            for st, best in ((a_state, 2), (b_state, 0)):
                a = agent.act(0, st, [0, 2], None, eps=1.0)
                agent.update_bandit(st, a, (1.0 if a == best else 0.0) + rng.gauss(0, 0.1))
        self.assertEqual(agent.act(0, a_state, [0, 2], None, 0.0), 2)
        self.assertEqual(agent.act(0, b_state, [0, 2], None, 0.0), 0)
        self.assertEqual(len(feature_vector(a_state)), sum(STATE_DIMS) + 1)


if __name__ == "__main__":
    unittest.main()
