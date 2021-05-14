import abc
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from utils.rewards import augment_reward


class Testbed(metaclass=abc.ABCMeta):

    # configure parameters of experiments
    def __init__(self,
                 search_interval: tuple,  # (0.33, 1)
                 time_horizon: int = 60,
                 trials: int = 40,
                 alpha: float = 3.1,
                 action_cost: int = 0,
                 warmup_days_bayesian: int = 4,
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
            alpha: float: property of pareto distribution used in heavy tailed reward simulation
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

        self.algorithms = None
        self.cum_reward = None
        self.reward_type = reward_type
        self.rewards = None

    @abc.abstractmethod
    def initialize_rewards(self):
        pass

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
                rewards = [augment_reward(rew,
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
                if alg.__class__.__name__ == "SEW":
                    self.simulate_a_day_sew(alg, algo_index, inst_reward, timestep)
                else:
                    self.simulate_a_day(alg, algo_index, inst_reward, timestep)

    @abc.abstractmethod
    def simulate_a_day_sew(self, alg, algo_index: int, inst_reward: np.array, day_number: int):
        pass

    def simulate_a_day(self, alg, algo_index: int, inst_reward: np.array, day_number: int):
        arm = alg.get_arm_value()
        reward = self.rewards.get_reward(arm, day_number)
        # reward consists of constant factor less regret plus stochastic reward factor from pareto distribution
        reward = augment_reward(reward,
                                self.stochasticity,
                                self.alpha,
                                self.action_cost,
                                self.heavy_tails,
                                self.noise_modulation
                                )
        inst_reward[algo_index, day_number] = reward
        alg.learn(arm, day_number, reward)  # algorithm is observing the reward and changing priors

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
