import numpy as np

from two_d.algorithms.algorithms_map import get_algorithms

from utils.paths import scenario
from utils.rewards import Rewards
from utils.scenario_generator import ScenarioGenerator
import pandas as pd

from utils.testbed import Testbed


class Testbed2D(Testbed):

    def __init__(self, time_horizon: int = 60, trials: int = 40, delta: float = 0.1, alpha: float = 3.1,
                 epsilon: int = 1, action_cost: int = 0, c_zooming: float = 0.01, c_adtm: float = 0.1,
                 c_admm: float = 0.1, warmup_days_bandits: int = 4, warmup_days_bayesian: int = 4,
                 search_interval: tuple = (0, 1), stochasticity: bool = True, heavy_tails: bool = False,
                 noise_modulation: float = .3, reward_type: str = "triangular", img_filepath: str = None,
                 is_sequential_learning: bool = True, batch_size: int = 4, verbosity: int = 1):
        """

        Args:
            time_horizon: int: a time horizon of a simulated experiment
            trials: int: number of independent trials
            delta: float:
            alpha: float: property of pareto distribution used in heavy tailed reward simulation
            epsilon: int:
            action_cost: int: a cost associated with any action on every step, regardless of the result
            c_zooming: float:
            c_adtm: float:
            c_admm: float:
            warmup_days_bandits: int:
            warmup_days_bayesian: int:
            search_interval: tuple:
            stochasticity: bool:
            heavy_tails: bool:
            noise_modulation: float:
            reward_type: str: {"triangular", "quadratic", "article"}
            img_filepath: str:
            is_sequential_learning: bool:
            batch_size: int
            verbosity: int
        """

        super().__init__(search_interval, time_horizon, trials, alpha, action_cost, warmup_days_bayesian, stochasticity,
                         heavy_tails, noise_modulation, reward_type, img_filepath, is_sequential_learning, batch_size,
                         verbosity)

        # compute upper bounds for moments of different orders
        a_hat = max(abs(-action_cost), abs(-action_cost - 0.4))
        sigma_second = max(alpha / ((alpha - 1) ** 2 * (alpha - 2)), 1 / (36 * np.sqrt(2)))
        nu_second = max(a_hat ** 2 + sigma_second, np.power(12 * np.sqrt(2), -(1 + epsilon)))
        nu_third = a_hat ** 3 + 2 * alpha * (alpha + 1) / (
                (alpha - 1) ** 3 * (alpha - 2) * (alpha - 3)) + 3 * a_hat * sigma_second
        warmup_steps_bandits = warmup_days_bandits if is_sequential_learning else warmup_days_bandits * batch_size
        warmup_steps_bayesian = warmup_days_bayesian if is_sequential_learning else warmup_days_bayesian * batch_size

        self.algorithms = get_algorithms(time_horizon=time_horizon,
                                         batch_size=batch_size,
                                         search_interval=search_interval,
                                         delta=delta,
                                         c_zooming=c_zooming,
                                         nu_third=nu_third,
                                         c_adtm=c_adtm,
                                         nu_second=nu_second,
                                         epsilon=epsilon,
                                         c_admm=c_admm,
                                         sigma_second=sigma_second,
                                         reward_type=reward_type,
                                         warmup_bandits=warmup_steps_bandits,
                                         warmup_bayesian=warmup_steps_bayesian)
        self.initialize_rewards()

    # configure parameters of experiments
    def initialize_rewards(self):
        sc = ScenarioGenerator(self.time_horizon)
        sc.generate_scale_persist()
        df = pd.read_csv(scenario)
        self.rewards = Rewards(df, self.reward_type)

    def simulate_a_day_sew(self, alg, algo_index: int, inst_reward: np.array, day_number: int):
        pass
