from unittest import TestCase
from sew import SEW, exponential_weighting
import numpy as np


class TestSEW(TestCase):

    def test_exponential_weighting(self):
        probs = exponential_weighting(timestep=50,
                                      rews=np.array([[5, 3, 2, 1], [1, 2, 3, 5]]),
                                      suboptimality_gap=0.05)
        self.assertEqual(len(probs), 2)
        self.assertEqual(len(probs[0]), 4)
        self.assertEqual(len(probs[1]), 4)

        self.assertAlmostEqual(probs[0][0], 0.41, 2)
        self.assertAlmostEqual(probs[0][1], 0.25, 2)
        self.assertAlmostEqual(probs[0][2], 0.19, 2)
        self.assertAlmostEqual(probs[0][3], 0.15, 2)

        self.assertAlmostEqual(probs[1][0], 0.15, 2)
        self.assertAlmostEqual(probs[1][1], 0.19, 2)
        self.assertAlmostEqual(probs[1][2], 0.25, 2)
        self.assertAlmostEqual(probs[1][3], 0.41, 2)

    def test_compute_reward(self):
        rew = SEW.compute_reward(1, 2, 0.99)
        self.assertEqual(rew, 1)

        rew = SEW.compute_reward(2, 1, 0.99)
        self.assertEqual(rew, -1)

        rew = SEW.compute_reward(1, 2, 1.99)
        self.assertEqual(rew, 0)

    def test_init(self):
        s = SEW(T=64, daily_auctions_number=1)
        self.assertEqual(s.time_horizon, 64)
        self.assertEqual(s.L, 3)

        self.assertEqual(len(s.M), 3)
        self.assertSequenceEqual(list(s.M), [4, 8, 16])

        self.assertEqual(len(s.U), 3)
        self.assertSequenceEqual(list(s.U), [3, 7, 15])

        self.assertEqual(len(s.W), 3)
        self.assertSequenceEqual(list(s.W), [1, 3, 7])

        self.assertEqual(len(s.I), 3)
        self.assertEqual(len(s.I[0]), 4)
        self.assertSequenceEqual(s.I[0], [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)])
        self.assertEqual(len(s.I[1]), 8)
        self.assertSequenceEqual(s.I[1],
                                 [(0.0, 0.125), (0.125, 0.25), (0.25, 0.375), (0.375, 0.5), (0.5, 0.625), (0.625, 0.75),
                                  (0.75, 0.875), (0.875, 1.0)])
        self.assertEqual(len(s.I[2]), 16)
        self.assertSequenceEqual(s.I[2],
                                 [(0.0, 0.0625), (0.0625, 0.125), (0.125, 0.1875), (0.1875, 0.25), (0.25, 0.3125),
                                  (0.3125, 0.375), (0.375, 0.4375), (0.4375, 0.5), (0.5, 0.5625), (0.5625, 0.625),
                                  (0.625, 0.6875), (0.6875, 0.75), (0.75, 0.8125), (0.8125, 0.875), (0.875, 0.9375),
                                  (0.9375, 1.0)])

        self.assertEqual(len(s.b), 3)
        self.assertEqual(len(s.b[0]), 1)
        self.assertSequenceEqual(s.b[0], [0.5])  # Might contain errors
        self.assertEqual(len(s.b[1]), 3)
        self.assertSequenceEqual(list(s.b[1]), [0.25, 0.5, 0.75])  # Might contain errors
        self.assertEqual(len(s.b[2]), 7)
        self.assertSequenceEqual(list(s.b[2]), [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])  # Might contain errors

        self.check_visiting_t(s)

        self.assertEqual(len(s.R), 3)

        self.assertEqual(len(s.R[0]), 4)

        self.assertEqual(len(s.R[0][0]), 3)
        self.assertEqual(len(s.R[0][1]), 3)
        self.assertEqual(len(s.R[0][2]), 3)
        self.assertEqual(len(s.R[0][3]), 3)

        self.assertSequenceEqual(s.R[0][0], [0, 0, 0])
        self.assertSequenceEqual(s.R[0][1], [0, 0, 0])
        self.assertSequenceEqual(s.R[0][2], [0, 0, 0])
        self.assertSequenceEqual(s.R[0][3], [0, 0, 0])

        self.assertEqual(len(s.R[1]), 8)

        self.assertEqual(len(s.R[1][0]), 7)
        self.assertEqual(len(s.R[1][1]), 7)
        self.assertEqual(len(s.R[1][2]), 7)
        self.assertEqual(len(s.R[1][3]), 7)
        self.assertEqual(len(s.R[1][4]), 7)
        self.assertEqual(len(s.R[1][5]), 7)
        self.assertEqual(len(s.R[1][6]), 7)
        self.assertEqual(len(s.R[1][7]), 7)

        self.assertSequenceEqual(s.R[1][0], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[1][1], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[1][2], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[1][3], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[1][4], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[1][5], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[1][6], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[1][7], [0, 0, 0, 0, 0, 0, 0])

        self.assertEqual(len(s.R[2]), 16)

        self.assertEqual(len(s.R[2][0]), 15)
        self.assertEqual(len(s.R[2][1]), 15)
        self.assertEqual(len(s.R[2][2]), 15)
        self.assertEqual(len(s.R[2][3]), 15)
        self.assertEqual(len(s.R[2][4]), 15)
        self.assertEqual(len(s.R[2][5]), 15)
        self.assertEqual(len(s.R[2][6]), 15)
        self.assertEqual(len(s.R[2][7]), 15)
        self.assertEqual(len(s.R[2][8]), 15)
        self.assertEqual(len(s.R[2][9]), 15)
        self.assertEqual(len(s.R[2][10]), 15)
        self.assertEqual(len(s.R[2][11]), 15)
        self.assertEqual(len(s.R[2][12]), 15)
        self.assertEqual(len(s.R[2][13]), 15)
        self.assertEqual(len(s.R[2][14]), 15)
        self.assertEqual(len(s.R[2][15]), 15)

        self.assertSequenceEqual(s.R[2][0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][3], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][4], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][6], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][7], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][8], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][10], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][11], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][12], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][13], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][14], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][15], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.assertEqual(len(s.R_dash), 3)

        self.assertEqual(len(s.R_dash[0]), 4)

        self.assertEqual(len(s.R_dash[0][0]), 1)
        self.assertEqual(len(s.R_dash[0][1]), 1)
        self.assertEqual(len(s.R_dash[0][2]), 1)
        self.assertEqual(len(s.R_dash[0][3]), 1)

        self.assertSequenceEqual(s.R_dash[0][0], [0])
        self.assertSequenceEqual(s.R_dash[0][1], [0])
        self.assertSequenceEqual(s.R_dash[0][2], [0])
        self.assertSequenceEqual(s.R_dash[0][3], [0])

        self.assertEqual(len(s.R_dash[1]), 8)

        self.assertEqual(len(s.R_dash[1][0]), 3)
        self.assertEqual(len(s.R_dash[1][1]), 3)
        self.assertEqual(len(s.R_dash[1][2]), 3)
        self.assertEqual(len(s.R_dash[1][3]), 3)
        self.assertEqual(len(s.R_dash[1][4]), 3)
        self.assertEqual(len(s.R_dash[1][5]), 3)
        self.assertEqual(len(s.R_dash[1][6]), 3)
        self.assertEqual(len(s.R_dash[1][7]), 3)

        self.assertSequenceEqual(s.R_dash[1][0], [0, 0, 0])
        self.assertSequenceEqual(s.R_dash[1][1], [0, 0, 0])
        self.assertSequenceEqual(s.R_dash[1][2], [0, 0, 0])
        self.assertSequenceEqual(s.R_dash[1][3], [0, 0, 0])
        self.assertSequenceEqual(s.R_dash[1][4], [0, 0, 0])
        self.assertSequenceEqual(s.R_dash[1][5], [0, 0, 0])
        self.assertSequenceEqual(s.R_dash[1][6], [0, 0, 0])
        self.assertSequenceEqual(s.R_dash[1][7], [0, 0, 0])

        self.assertEqual(len(s.R_dash[2]), 16)

        self.assertEqual(len(s.R_dash[2][0]), 7)
        self.assertEqual(len(s.R_dash[2][1]), 7)
        self.assertEqual(len(s.R_dash[2][2]), 7)
        self.assertEqual(len(s.R_dash[2][3]), 7)
        self.assertEqual(len(s.R_dash[2][4]), 7)
        self.assertEqual(len(s.R_dash[2][5]), 7)
        self.assertEqual(len(s.R_dash[2][6]), 7)
        self.assertEqual(len(s.R_dash[2][7]), 7)
        self.assertEqual(len(s.R_dash[2][8]), 7)
        self.assertEqual(len(s.R_dash[2][9]), 7)
        self.assertEqual(len(s.R_dash[2][10]), 7)
        self.assertEqual(len(s.R_dash[2][11]), 7)
        self.assertEqual(len(s.R_dash[2][12]), 7)
        self.assertEqual(len(s.R_dash[2][13]), 7)
        self.assertEqual(len(s.R_dash[2][14]), 7)
        self.assertEqual(len(s.R_dash[2][15]), 7)

        self.assertSequenceEqual(s.R_dash[2][0], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][1], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][2], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][3], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][4], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][5], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][6], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][7], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][8], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][9], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][10], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][11], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][12], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][13], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][14], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][15], [0, 0, 0, 0, 0, 0, 0])

    def check_visiting_t(self, s):
        self.assertEqual(len(s.visiting_T), 3)
        self.assertEqual(len(s.visiting_T[0]), 4)
        self.assertSequenceEqual(s.visiting_T[0], [0, 0, 0, 0])
        self.assertEqual(len(s.visiting_T[1]), 8)
        self.assertSequenceEqual(s.visiting_T[1], [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(len(s.visiting_T[2]), 16)
        self.assertSequenceEqual(s.visiting_T[2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_infer_m(self):
        s = SEW(1)
        np.random.seed(42)
        vt = np.random.uniform()
        self.assertEqual(vt, .3745401188473625)
        inferred_m = s.infer_m(0, vt)
        self.assertEqual(inferred_m, 1)
        inferred_m = s.infer_m(1, vt)
        self.assertEqual(inferred_m, 2)
        inferred_m = s.infer_m(5, vt)
        self.assertEqual(inferred_m, 47)

    def test_initialize_ms(self):
        s, vt = self.init_sew_single_example()
        self.assertEqual(vt, .3745401188473625)
        s.initialize_ms(vt)
        self.assertEqual(len(s.ms), 3)
        self.assertSequenceEqual(s.ms, (1, 2, 5))

    @staticmethod
    def init_sew_single_example() -> tuple:
        s = SEW(T=64, daily_auctions_number=1)
        np.random.seed(42)
        vt = np.random.uniform()
        return s, vt

    def test_update_t(self):
        s, vt = self.init_sew_single_example()
        s.initialize_ms(vt)
        s.update_T()
        self.check_updated_visiting_t(s)

    def check_updated_visiting_t(self, s):
        self.assertSequenceEqual(s.visiting_T[0], [0, 1, 0, 0])
        self.assertSequenceEqual(s.visiting_T[1], [0, 0, 1, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.visiting_T[2], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_compute_prob(self):
        s, vt = self.init_sew_single_example()
        s.initialize_ms(vt)
        s.update_T()
        s.compute_prob()

        self.assertEqual(len(s.prob), 3)

        self.assertEqual(len(s.prob[0]), 1)
        self.assertEqual(len(s.prob[0][0]), 4)
        self.assertSequenceEqual(list(s.prob[0][0]), [1 / 4, 1 / 4, 1 / 4, 1 / 4])

        self.assertEqual(len(s.prob[1]), 3)

        self.assertEqual(len(s.prob[1][0]), 4)
        self.assertEqual(len(s.prob[1][1]), 4)
        self.assertEqual(len(s.prob[1][2]), 4)

        self.assertSequenceEqual(tuple(s.prob[1][0]), (1 / 4, 1 / 4, 1 / 4, 1 / 4))
        self.assertSequenceEqual(tuple(s.prob[1][1]), (1 / 4, 1 / 4, 1 / 4, 1 / 4))
        self.assertSequenceEqual(tuple(s.prob[1][2]), (1 / 4, 1 / 4, 1 / 4, 1 / 4))

        self.assertEqual(len(s.prob[2]), 7)

        self.assertEqual(len(s.prob[2][0]), 4)
        self.assertEqual(len(s.prob[2][1]), 4)
        self.assertEqual(len(s.prob[2][2]), 4)
        self.assertEqual(len(s.prob[2][3]), 4)
        self.assertEqual(len(s.prob[2][4]), 4)
        self.assertEqual(len(s.prob[2][5]), 4)
        self.assertEqual(len(s.prob[2][6]), 4)

        self.assertSequenceEqual(tuple(s.prob[2][0]), (1 / 4, 1 / 4, 1 / 4, 1 / 4))
        self.assertSequenceEqual(tuple(s.prob[2][1]), (1 / 4, 1 / 4, 1 / 4, 1 / 4))
        self.assertSequenceEqual(tuple(s.prob[2][2]), (1 / 4, 1 / 4, 1 / 4, 1 / 4))
        self.assertSequenceEqual(tuple(s.prob[2][3]), (1 / 4, 1 / 4, 1 / 4, 1 / 4))
        self.assertSequenceEqual(tuple(s.prob[2][4]), (1 / 4, 1 / 4, 1 / 4, 1 / 4))
        self.assertSequenceEqual(tuple(s.prob[2][5]), (1 / 4, 1 / 4, 1 / 4, 1 / 4))
        self.assertSequenceEqual(tuple(s.prob[2][6]), (1 / 4, 1 / 4, 1 / 4, 1 / 4))

    def test_produce_action(self):
        s, vt = self.init_sew_single_example()
        s.initialize_ms(vt)
        s.update_T()
        s.compute_prob()
        action = s.compute_bt()
        self.assertEqual(action, .5)

    def test_initialize(self):
        s, vt = self.init_sew_single_example()
        s.initialize_ms(vt)
        s.update_T()
        self.check_updated_visiting_t(s)
        s.initialize()
        self.check_visiting_t(s)

    def test_learn(self):
        s, vt = self.init_sew_single_example()
        s.get_bid_price(vt)
        np.random.seed(43)
        mt = np.random.uniform()
        s.update(mt)

        self.assertEqual(len(s.R), 3)

        self.assertEqual(len(s.R[0]), 4)

        self.assertEqual(len(s.R[0][0]), 3)
        self.assertEqual(len(s.R[0][1]), 3)
        self.assertEqual(len(s.R[0][2]), 3)
        self.assertEqual(len(s.R[0][3]), 3)

        self.assertSequenceEqual(s.R[0][0], [0, 0, 0])
        self.assertAlmostEqual(s.R[0][1][0], 0.1050, 4)
        self.assertAlmostEqual(s.R[0][1][1], -0.1255, 4)
        self.assertAlmostEqual(s.R[0][1][2], -0.3755, 4)
        self.assertSequenceEqual(s.R[0][2], [0, 0, 0])
        self.assertSequenceEqual(s.R[0][3], [0, 0, 0])

        self.assertEqual(len(s.R[1]), 8)

        self.assertEqual(len(s.R[1][0]), 7)
        self.assertEqual(len(s.R[1][1]), 7)
        self.assertEqual(len(s.R[1][2]), 7)
        self.assertEqual(len(s.R[1][3]), 7)
        self.assertEqual(len(s.R[1][4]), 7)
        self.assertEqual(len(s.R[1][5]), 7)
        self.assertEqual(len(s.R[1][6]), 7)
        self.assertEqual(len(s.R[1][7]), 7)

        self.assertSequenceEqual(s.R[1][0], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[1][1], [0, 0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(s.R[1][2][0], 0.1715, 4)
        self.assertAlmostEqual(s.R[1][2][1], 0.1245, 4)
        self.assertAlmostEqual(s.R[1][2][2], -0.0005, 4)
        self.assertAlmostEqual(s.R[1][2][3], -0.1255, 4)
        self.assertAlmostEqual(s.R[1][2][4], -0.2505, 4)
        self.assertAlmostEqual(s.R[1][2][5], -0.3755, 4)
        self.assertAlmostEqual(s.R[1][2][6], -0.5005, 4)
        self.assertSequenceEqual(s.R[1][3], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[1][4], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[1][5], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[1][6], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[1][7], [0, 0, 0, 0, 0, 0, 0])

        self.assertEqual(len(s.R[2]), 16)

        self.assertEqual(len(s.R[2][0]), 15)
        self.assertEqual(len(s.R[2][1]), 15)
        self.assertEqual(len(s.R[2][2]), 15)
        self.assertEqual(len(s.R[2][3]), 15)
        self.assertEqual(len(s.R[2][4]), 15)
        self.assertEqual(len(s.R[2][5]), 15)
        self.assertEqual(len(s.R[2][6]), 15)
        self.assertEqual(len(s.R[2][7]), 15)
        self.assertEqual(len(s.R[2][8]), 15)
        self.assertEqual(len(s.R[2][9]), 15)
        self.assertEqual(len(s.R[2][10]), 15)
        self.assertEqual(len(s.R[2][11]), 15)
        self.assertEqual(len(s.R[2][12]), 15)
        self.assertEqual(len(s.R[2][13]), 15)
        self.assertEqual(len(s.R[2][14]), 15)
        self.assertEqual(len(s.R[2][15]), 15)

        self.assertSequenceEqual(s.R[2][0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][3], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][4], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(s.R[2][5][0], 0, 4)
        self.assertAlmostEqual(s.R[2][5][1], 0.2495, 4)
        self.assertAlmostEqual(s.R[2][5][2], 0.1870, 4)
        self.assertAlmostEqual(s.R[2][5][3], 0.1245, 4)
        self.assertAlmostEqual(s.R[2][5][4], 0.0620, 4)
        self.assertAlmostEqual(s.R[2][5][5], -0.0005, 4)
        self.assertAlmostEqual(s.R[2][5][6], -0.0630, 4)
        self.assertAlmostEqual(s.R[2][5][7], -0.1255, 4)
        self.assertAlmostEqual(s.R[2][5][8], -0.1880, 4)
        self.assertAlmostEqual(s.R[2][5][9], -0.2505, 4)
        self.assertAlmostEqual(s.R[2][5][10], -0.3130, 4)
        self.assertAlmostEqual(s.R[2][5][11], -0.3755, 4)
        self.assertAlmostEqual(s.R[2][5][12], -0.4380, 4)
        self.assertAlmostEqual(s.R[2][5][13], -0.5005, 4)
        self.assertAlmostEqual(s.R[2][5][14], -0.5630, 4)
        self.assertSequenceEqual(s.R[2][6], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][7], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][8], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][10], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][11], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][12], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][13], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][14], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R[2][15], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.assertEqual(len(s.R_dash), 3)

        self.assertEqual(len(s.R_dash[0]), 4)

        self.assertEqual(len(s.R_dash[0][0]), 1)
        self.assertEqual(len(s.R_dash[0][1]), 1)
        self.assertEqual(len(s.R_dash[0][2]), 1)
        self.assertEqual(len(s.R_dash[0][3]), 1)

        self.assertSequenceEqual(s.R_dash[0][0], [0])
        self.assertAlmostEqual(s.R_dash[0][1][0], -0.1255, 4)
        self.assertSequenceEqual(s.R_dash[0][2], [0])
        self.assertSequenceEqual(s.R_dash[0][3], [0])

        self.assertEqual(len(s.R_dash[1]), 8)

        self.assertEqual(len(s.R_dash[1][0]), 3)
        self.assertEqual(len(s.R_dash[1][1]), 3)
        self.assertEqual(len(s.R_dash[1][2]), 3)
        self.assertEqual(len(s.R_dash[1][3]), 3)
        self.assertEqual(len(s.R_dash[1][4]), 3)
        self.assertEqual(len(s.R_dash[1][5]), 3)
        self.assertEqual(len(s.R_dash[1][6]), 3)
        self.assertEqual(len(s.R_dash[1][7]), 3)

        self.assertSequenceEqual(s.R_dash[1][0], [0, 0, 0])
        self.assertSequenceEqual(s.R_dash[1][1], [0, 0, 0])
        self.assertAlmostEqual(s.R_dash[1][2][0], 0.1245, 4)
        self.assertAlmostEqual(s.R_dash[1][2][1], -0.1255, 4)
        self.assertAlmostEqual(s.R_dash[1][2][2], -0.3755, 4)
        self.assertSequenceEqual(s.R_dash[1][3], [0, 0, 0])
        self.assertSequenceEqual(s.R_dash[1][4], [0, 0, 0])
        self.assertSequenceEqual(s.R_dash[1][5], [0, 0, 0])
        self.assertSequenceEqual(s.R_dash[1][6], [0, 0, 0])
        self.assertSequenceEqual(s.R_dash[1][7], [0, 0, 0])

        self.assertEqual(len(s.R_dash[2]), 16)

        self.assertEqual(len(s.R_dash[2][0]), 7)
        self.assertEqual(len(s.R_dash[2][1]), 7)
        self.assertEqual(len(s.R_dash[2][2]), 7)
        self.assertEqual(len(s.R_dash[2][3]), 7)
        self.assertEqual(len(s.R_dash[2][4]), 7)
        self.assertEqual(len(s.R_dash[2][5]), 7)
        self.assertEqual(len(s.R_dash[2][6]), 7)
        self.assertEqual(len(s.R_dash[2][7]), 7)
        self.assertEqual(len(s.R_dash[2][8]), 7)
        self.assertEqual(len(s.R_dash[2][9]), 7)
        self.assertEqual(len(s.R_dash[2][10]), 7)
        self.assertEqual(len(s.R_dash[2][11]), 7)
        self.assertEqual(len(s.R_dash[2][12]), 7)
        self.assertEqual(len(s.R_dash[2][13]), 7)
        self.assertEqual(len(s.R_dash[2][14]), 7)
        self.assertEqual(len(s.R_dash[2][15]), 7)

        self.assertSequenceEqual(s.R_dash[2][0], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][1], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][2], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][3], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][4], [0, 0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(s.R_dash[2][5][0], 0.2495, 4)
        self.assertAlmostEqual(s.R_dash[2][5][1], 0.1245, 4)
        self.assertAlmostEqual(s.R_dash[2][5][2], -0.0005, 4)
        self.assertAlmostEqual(s.R_dash[2][5][3], -0.1255, 4)
        self.assertAlmostEqual(s.R_dash[2][5][4], -0.2505, 4)
        self.assertAlmostEqual(s.R_dash[2][5][5], -0.3755, 4)
        self.assertAlmostEqual(s.R_dash[2][5][6], -0.5005, 4)
        self.assertSequenceEqual(s.R_dash[2][6], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][7], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][8], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][9], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][10], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][11], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][12], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][13], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][14], [0, 0, 0, 0, 0, 0, 0])
        self.assertSequenceEqual(s.R_dash[2][15], [0, 0, 0, 0, 0, 0, 0])
