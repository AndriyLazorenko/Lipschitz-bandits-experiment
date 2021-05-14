import numpy as np
import pandas as pd
from tqdm import tqdm

from three_d.algorithms.algorithms_map import get_algorithms
from utils import timer

from utils.paths import scenario
from utils.rewards import Rewards, augment_reward
from utils.scenario_generator import ScenarioGenerator

from utils.testbed import Testbed


class Testbed3D(Testbed):
    def __init__(self, time_horizon: int = 60, trials: int = 40, alpha: float = 3.1, action_cost: int = 0,
                 warmup_days_bayesian: int = 4, search_interval: tuple = ((0., 2.), (0., 10.)), search_interval_2d: tuple = (0.0, 1.0),
                 stochasticity: bool = True, heavy_tails: bool = False, noise_modulation: float = .3,
                 reward_type: str = "triangular", img_filepath: str = None, is_sequential_learning: bool = True,
                 batch_size: int = 4, verbosity: int = 1):
        """

        Args:
            time_horizon: int: a time horizon of a simulated experiment
            trials: int: number of independent trials
            action_cost: int: a cost associated with any action on every step, regardless of the result
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
        warmup_steps_bayesian = warmup_days_bayesian if is_sequential_learning else warmup_days_bayesian * batch_size

        self.algorithms = get_algorithms(time_horizon=time_horizon,
                                         batch_size=batch_size,
                                         search_interval=search_interval,
                                         search_interval_2d = search_interval_2d,
                                         reward_type=reward_type,
                                         warmup_bayesian=warmup_steps_bayesian)
        self.initialize_rewards()

    def initialize_rewards(self):
        sc = ScenarioGenerator(self.time_horizon)
        sc.generate_scale_persist()
        df = pd.read_csv(scenario)
        self.rewards = Rewards(df, self.reward_type)

    def simulate_a_day_sew(self, alg, algo_index: int, inst_reward: np.array, day_number: int, verbosity: int = 0):
        df = self.rewards.df.loc[self.rewards.df.day == day_number].copy()
        vt = df.vt.to_numpy()
        mt = df.mt.to_numpy()
        daily_reward = np.zeros(len(vt))

        if verbosity > 0:
            trial_range = tqdm(vt)
        else:
            trial_range = vt

        for ind, value in enumerate(trial_range):
            bid_price = alg.get_bid_price(value)
            reward = alg.compute_reward(bid_price, value, mt[ind])
            daily_reward[ind] = reward
            alg.update(mt[ind])

        daily_reward = np.sum(daily_reward)
        reward = augment_reward(daily_reward,
                                self.stochasticity,
                                self.alpha,
                                self.action_cost,
                                self.heavy_tails,
                                self.noise_modulation
                                )
        print(f"Reward on timestep {day_number} for SEW: {reward}")
        inst_reward[algo_index, day_number] = reward





