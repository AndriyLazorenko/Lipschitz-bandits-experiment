from typing import Tuple
import numpy as np
from numpy import power, exp2
from scipy.special import softmax

from three_d.algorithms.abstract_algorithm import Algorithm


class SEW(Algorithm):
    def __init__(self, T, daily_auctions_number: int = 10000):
        """
        Method to initialize SEW algorithm and all the relevant variables.
        Args:
            T: int: number of days for SEW evaluation. Use 60 or less to test on synthetic data, or regenerate it
            daily_auctions_number: int: number of auctions per day. 10000 is relevant for synthetic data
        """
        super().__init__(T)
        self.daily_auctions_number = daily_auctions_number
        self.time_horizon = self.T * daily_auctions_number  # page 20 line 2 SEW. Time horizon means all auctions,
        # not number of days. Adjustments needed to be made to ensure time horizon corresponds to simulated data
        self.L = int(np.log2(np.sqrt(self.time_horizon)))  # page 20 line 2 SEW
        assert isinstance(self.L, int)
        assert self.L > 0
        self.M = np.array([power(2, l + 1) for l in range(1, self.L + 1)])  # page 20 line 4 SEW
        self.U = np.array([power(2, l + 1) - 1 for l in range(1, self.L + 1)])  # page 20 line 4 SEW
        self.W = np.array([power(2, l) - 1 for l in range(1, self.L + 1)])  # page 20 line 4 SEW

        self.I = np.asarray([[
            ((m - 1) / self.M[l], (m) / self.M[l]) for m in range(1, self.M[l] + 1)
        ] for l in range(self.L)
        ])  # page 20 lines 4-5 SEW

        # initialization below might contain errors. See tests for examples
        self.b = list()
        for l in range(1, self.L + 1):
            w_arr = [w for w in range(1, self.W[l - 1] + 1)]
            self.b.append(np.multiply(exp2(-l), w_arr))
        self.b = np.asarray(self.b)  # page 20 line 5 SEW

        self.visiting_T = np.asarray(
            [[0 for m in range(1, self.M[l] + 1)] for l in range(self.L)])  # page 20 line 7 SEW

        # initialization below might contain index errors
        self.R = np.asarray([[[
            0 for u in range(1, self.U[l] + 1)
        ] for m in range(1, self.M[l] + 1)
        ] for l in range(self.L)
        ])  # page 20 line 8 SEW

        # initialization below might contain index errors
        self.R_dash = np.asarray([[[
            0 for w in range(1, self.W[l] + 1)
        ] for m in range(1, self.M[l] + 1)
        ] for l in range(self.L)
        ])  # page 20 line 8 SEW

        self.ms = None
        self.prob = None
        self.vt = None

    def initialize(self):
        """
        Method used to reset all variables for several separate iterations of algorithm.

        """
        self.__init__(self.T, self.daily_auctions_number)

    def get_bid_price(self, vt: float) -> float:
        """
        Retrieves bt given vt using SEW
        Args:
            vt: float: an oracle-based prediction of value of a specific bid (taken from HBT)

        Returns:
            action: float: a specific, normalized bidding price bt
        """
        self.vt = vt  # page 20 line 11 SEW
        self.initialize_ms(vt)  # page 20 line 14 SEW
        self.update_T()  # page 20 line 14 SEW
        self.compute_prob()  # page 20 lines 15-16 SEW
        action = self.compute_bt()  # page 20 lines 19-29 SEW
        return action

    def compute_bt(self) -> float:  # Not sure that this method works as intended. Need to revise logic
        """
        Computes bt using probabilites taken from EW subroutine.
        Returns:
            bt: float:
        """
        w_star = 1  # page 20 line 19 SEW
        for l in range(1, self.L + 1):  # page 20 line 20 SEW
            distribution = self.prob[l - 1][w_star - 1]  # page 20 line 21 SEW
            s = np.random.choice([1, 2, 3, 4], p=distribution)  # page 20 line 21 SEW
            if s == 4:  # page 20 line 22 SEW
                bt = self.b[l - 1][w_star - 1]  # page 20 line 23 SEW
                return bt  # page 20 line 23 SEW
            elif l < self.L:  # page 20 line 24 SEW
                w_star = 2 * (w_star - 1) + s  # page 20 line 25 SEW; index error might be present
            else:  # page 20 line 26 SEW
                bt = power(2., -self.L - 1) * (
                            2 * (w_star - 1) + s)  # page 20 line 27 SEW; index error might be present
                return bt  # page 20 line 27 SEW

    def compute_prob(self):
        """
        Computes probabilities tensor using EW subroutine for all levels
        """
        self.prob = list()
        [self.compute_prob_for_a_level(l) for l in range(1, self.L + 1)]

    def compute_prob_for_a_level(self, l):
        """
        Computes probabilities tensor using EW subroutine for a level
        Args:
            l: int: level
        """
        rewards = np.array([
            [self.R[l - 1][self.ms[l - 1]][2 * w - 1 - 1],
             self.R[l - 1][self.ms[l - 1]][2 * w - 1],
             self.R[l - 1][self.ms[l - 1]][2 * w + 1 - 1],
             self.R_dash[l - 1][self.ms[l - 1]][w - 1]
             ] for w in range(1, self.W[l - 1] + 1)
        ])
        entry_l = exponential_weighting(timestep=self.visiting_T[l - 1][self.ms[l - 1]],
                                        suboptimality_gap=power(2., 1 - l),
                                        rews=rewards)
        self.prob.append(entry_l)

    def update_T(self):
        """
        Updates tensor of historical interval visits, visiting_T
        """
        for l in range(self.L):
            self.visiting_T[l][self.ms[l]] += 1

    def initialize_ms(self, vt: float):
        """
        Initializes ms (indices of a specific range respective to level split)
        Args:
            vt: float
        """
        self.ms = [self.infer_m(l, vt) for l in range(self.L)]

    def infer_m(self, l: int, vt: float):
        """
        Infers an index for a specific range to which vt belongs
        Args:
            l: int: level
            vt: float

        Returns:
            m: int: an index of a specific index for a specific range to which vt belongs
        """
        for m in range(self.M[l]):
            if self.I[l][m][0] < vt <= self.I[l][m][1]:
                return m
            elif vt == 0.0:
                return 0

    def update(self, mt: float):  # page 20 line 31 SEW
        """
        A method that updates the weights to be used in EW subroutine calculation.
        Args:
            mt: float: minimum winning bid
        """
        r_dash_cached = [0 for i in range(self.L)]
        r_cached = [0 for i in range(self.L)]
        for l in range(self.L, 0, -1):  # page 20 line 32 SEW
            self._w_cycle_update(l, mt, r_dash_cached)  # page 20 lines 32-33 SEW
            self._u_cycle_update(l, mt, r_cached, r_dash_cached)  # page 20 lines 34-41 SEW

    def _w_cycle_update(self, l, mt, r_dash_cached):  # page 20 line 33 SEW
        dash_cache = self.compute_reward_vec(self.b[l - 1], self.vt, mt)
        self.R_dash[l - 1][self.ms[l - 1]] += dash_cache  # might contain errors on addition of arrays
        r_dash_cached[l - 1] = dash_cache

    def _u_cycle_update(self, l, mt, r_cached, r_dash_cached):
        u_arr = np.array([u for u in range(1, self.U[l - 1] + 1)])
        if l == self.L:  # page 20 line 34 SEW
            cache = SEW.compute_reward_vec(exp2(-self.L - 1) * u_arr, self.vt, mt)  # page 20 line 35 SEW
        else:  # page 20 line 36 SEW
            a = np.array(r_cached[l][2 * (u_arr - 1) + 0])  # page 20 line 37 SEW
            b = np.array(r_cached[l][2 * (u_arr - 1) + 1])  # page 20 line 37 SEW
            c = np.array(r_cached[l][2 * (u_arr - 1) + 2])  # page 20 line 37 SEW
            b = np.array([a, b, c, r_dash_cached[l]])  # page 20 line 37 SEW
            cache = np.einsum('ij,ij->j', self.prob[l].T, b)  # page 20 line 37 SEW
        self.R[l - 1][self.ms[l - 1]] += cache  # page 20 lines 35, 39; might contain errors on addition of arrays
        r_cached[l - 1] = cache  # page 20 line 37

    @staticmethod
    def compute_reward(bt: float, vt: float, mt: float):  # page 6 formula 1: reward calculation
        return (vt - bt) if mt < bt else 0

    # TODO: add tests
    @staticmethod
    def compute_reward_vec(bt: np.array, vt: float, mt: float) -> np.array:  # page 6 formula 1: reward calculation
        """A vectorized form of reward calculation (more time-efficient)"""
        return (vt - bt) * (mt < bt)

    def get_arm_value(self) -> Tuple:
        raise NotImplementedError("Arm value is not defined for SEW. Use get_action method instead")

    def learn(self, action: Tuple, timestep: int, reward: float):
        raise NotImplementedError("Learn method is not used in SEW. Use update method instead")

    def get_arms_batch(self) -> list:
        raise NotImplementedError("Batch processing doesn't make sense with regard to SEW")

    def batch_learn(self, actions: list, timesteps: list, rewards: list):
        raise NotImplementedError("Batch learning makes no sense for SEW")


def exponential_weighting(timestep: int,
                          rews: np.array,
                          suboptimality_gap: float
                          ) -> np.array:  # page 21 line 2 EW
    """
    An exponential weighting subroutine implementation from page 21 algorithm 3 (EW)
    Args:
        timestep: int: number of times algorithm already visited the interval
        rews: np.array: reward vector of shape 4 x W[l]
        suboptimality_gap: float

    Returns:
        prob: np.array(4) x W[l]

    """
    learning_rate = min([1 / 4, np.sqrt(np.log(4) / (np.dot(timestep, suboptimality_gap)))])  # page 21 line 4
    return softmax(learning_rate * rews, axis=1)  # page 21 lines 3,5-8 EW
