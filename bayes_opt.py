import numpy as np
from skopt import gp_minimize
from skopt.plots import plot_regret

from run import Testbed
import matplotlib.pyplot as plt


class BayesianSolution:
    stochasticity = True
    alpha = 3.1
    action_cost = 0
    heavy_tails = True
    noise_modulation = .50
    # true_minimum = -0.9  # 0 or -0.9
    true_minimum = 0

    def get_regret(self, arm: list):
        arm = arm[0]
        # rew = Testbed.triangular_reward(arm)
        rew = Testbed.quadratic_reward(arm)
        rew = Testbed.augment_reward(rew,
                                     stochasticity=self.stochasticity,
                                     alpha=self.alpha,
                                     action_cost=self.action_cost,
                                     heavy_tails=self.heavy_tails,
                                     noise_modulation=self.noise_modulation)
        return -rew

    def plot_regret(self):
        x = np.linspace(0, 1, 400).reshape(-1, 1)
        fx = [self.get_regret(x_i) for x_i in x]
        plt.plot(x, fx, "r--", label="regret")
        plt.legend()
        plt.grid()
        plt.show()

    def main(self):
        res = gp_minimize(self.get_regret,  # the function to minimize
                          [(0.0, 1.0)],  # the bounds on each dimension of x
                          acq_func="LCB",  # the acquisition function
                          n_calls=60,  # the number of evaluations of f
                          n_initial_points=0,
                          n_random_starts=10,  # the number of random initialization points
                          noise=0.1 ** 2,  # the noise level (optional)
                          random_state=1234)  # the random seed
        print(res)
        print("x^*=%.4f, f(x^*)=%.4f" % (res.x[0], res.fun))
        plot_regret(res, true_minimum=self.true_minimum)
        plt.show()


if __name__ == "__main__":
    b = BayesianSolution()
    b.plot_regret()
    b.main()
