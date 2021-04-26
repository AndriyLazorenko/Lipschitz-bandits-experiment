import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from algorithms.algorithms_map import get_algorithms
from rewards_2d import Rewards2D

from utils.paths import scenario
from utils.scenario_generator import ScenarioGenerator
import pandas as pd


class Testbed2D:

    # configure parameters of experiments
    def __init__(self,
                 time_horizon: int = 60,
                 trials: int = 40,
                 delta: float = 0.1,
                 alpha: float = 3.1,
                 epsilon: int = 1,
                 action_cost: int = 0,
                 c_zooming: float = 0.01,  # searched within {1, 0.1, 0.01} and `0.01` is the best choice 0.0009
                 c_adtm: float = 0.1,  # searched within {1, 0.1, 0.01} and `0.1` is the best choice 0.009
                 c_admm: float = 0.1,  # searched within {1, 0.1, 0.01} and `0.1` is the best choice 0.1
                 warmup_days_bandits: int = 4,  # it is not advised to try values > 4
                 warmup_days_bayesian: int = 4,
                 search_interval: tuple = (0, 1),  # (0.33, 1)
                 stochasticity: bool = True,
                 heavy_tails: bool = False,
                 noise_modulation: float = .3,
                 reward_type: str = "triangular",
                 img_filepath: str = None,
                 is_sequential_learning: bool = True,
                 batch_size: int = 4,
                 verbosity: int = 1
                 ):
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

        self.time_horizon = time_horizon
        self.trials = trials
        self.action_cost = action_cost
        self.alpha = alpha
        self.search_interval = search_interval
        self.stochasticity = stochasticity
        self.heavy_tails = heavy_tails
        self.noise_modulation = noise_modulation
        self.img_fpath = img_filepath
        self.is_sequential_learning = is_sequential_learning
        self.batch_size = batch_size
        self.verbosity = verbosity

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
        self.cum_reward = None

        sc = ScenarioGenerator(time_horizon)
        sc.generate_scale_persist()
        df = pd.read_csv(scenario)

        self.reward_type = reward_type
        self.rewards = Rewards2D(df, reward_type)

    def simulate(self):
        """
        Simulates several algorithm's performance on search for optimal arm.
        There is a separate function calculating regret and rewards.
        Regret is calculated based on knowledge of the shape of regret as a function of optimal parameter.
        Reward is calculated based on that shape as well as stochastic heavy-tailed component and fixed cost associated
        with pulling an arm.

        In a real-life setting we don't know regret and cannot compare two algorithms based on their regrets.
        We can only compare them based on rewards. Therefore a refactoring was applied to reflect performance
        in terms of cumulative reward (the more the merrier)

        Returns:
            avg_cum_regret:

        """
        cum_reward = np.zeros((len(self.algorithms), self.time_horizon + 1))

        if self.verbosity > 0:
            trial_range = tqdm(range(self.trials))
        else:
            trial_range = range(self.trials)

        for trial in trial_range:
            inst_reward = np.zeros((len(self.algorithms), self.time_horizon + 1))
            for alg in self.algorithms:
                alg.initialize()
            if self.is_sequential_learning:
                self._sequential_learning(inst_reward)
            else:
                self._batch_learning(inst_reward)
            cum_reward += np.cumsum(inst_reward, axis=-1)
        avg_cum_regret = cum_reward / self.trials
        self.cum_reward = avg_cum_regret

    @staticmethod
    def chunks(lst: list, n: int):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def _batch_learning(self, inst_reward):
        if self.reward_type == "article":
            raise NotImplementedError("Batch learning is not supported by article reward")
        batches = self.chunks(list(range(1, self.time_horizon + 1)), self.batch_size)
        if self.verbosity > 1:
            batches = tqdm(batches)
        for batch in batches:
            for algo_index, alg in enumerate(self.algorithms):
                arms = alg.get_arms_batch()
                rewards = [self.rewards.get_reward(arm, 0) for arm in arms]
                rewards = [self.rewards.augment_reward(rew,
                                                       self.stochasticity,
                                                       self.alpha,
                                                       self.action_cost,
                                                       self.heavy_tails,
                                                       self.noise_modulation) for rew in rewards]
                for ind, timestep in enumerate(batch):
                    inst_reward[algo_index, timestep] = rewards[ind]
                alg.batch_learn(actions=arms, timesteps=batch, rewards=rewards)

    def _sequential_learning(self, inst_reward):
        if self.verbosity > 1:
            timestep_range = tqdm(range(1, self.time_horizon + 1))
        else:
            timestep_range = range(1, self.time_horizon + 1)
        for timestep in timestep_range:
            for algo_index, alg in enumerate(self.algorithms):
                arm = alg.get_arm_value()
                reward = self.rewards.get_reward(arm, timestep)
                # reward consists of constant factor less regret plus stochastic reward factor from pareto distribution
                reward = self.rewards.augment_reward(reward,
                                                     self.stochasticity,
                                                     self.alpha,
                                                     self.action_cost,
                                                     self.heavy_tails,
                                                     self.noise_modulation
                                                     )
                inst_reward[algo_index, timestep] = reward
                alg.learn(arm, timestep, reward)  # algorithm is observing the reward and changing priors

    def plot(self):
        """
        Plots the results and saves the image

        Returns:

        """
        plt.figure(figsize=(7, 4))
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        names = [f'{alg.__class__.__name__}' for alg in self.algorithms]
        linestyles = ['-', '--', '-.', ':', "-", "--", "-.", ":", '-']
        for result, name, linestyle in zip(self.cum_reward, names, linestyles):
            plt.plot(result, label=name, linewidth=2.0, linestyle=linestyle)
        plt.legend(loc='upper left', frameon=True, fontsize=10)
        plt.xlabel('t', labelpad=1, fontsize=15)
        plt.ylabel('cumulative reward', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid()
        print("Saving figure...")
        fname = self.img_fpath
        print(fname)
        plt.savefig(fname,
                    dpi=500,
                    bbox_inches='tight'
                    )
        plt.close()
        print("Saved!")
