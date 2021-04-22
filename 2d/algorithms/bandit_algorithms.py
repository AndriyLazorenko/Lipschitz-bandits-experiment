from abc import ABCMeta
from algorithms.abstract_algorithm import Algorithm
import numpy as np


class Bandit(Algorithm, metaclass=ABCMeta):
    def __init__(self, T, batch_size, arm_interval):
        super().__init__(T, batch_size, arm_interval)
        self.current_arm_idx = 0

    def initialize(self):
        self.current_arm_idx = 0
        self.active_arms = []

    def random_exploration(self):
        self.active_arms.append(self.rg.uniform(self.arm_interval[0], self.arm_interval[1]))
        self.current_arm_idx = len(self.active_arms) - 1

    def init_active_arms(self, init_policy: str = "Random", root_power: float = 1 / 2) -> int:
        n_arms = int(np.power(self.T, root_power))
        if init_policy == "Random":
            self.active_arms = [self.inverse_interval_scaler(self.rg.uniform()) for _ in range(n_arms)]
        elif init_policy == "Equidistant":
            self.active_arms = [self.inverse_interval_scaler(n / (n_arms - 1)) for n in range(n_arms)]
        return n_arms

    def get_arms_batch(self) -> list:
        return [self.get_arm_value() for _ in range(self.batch_size)]

    def batch_learn(self, actions: list, timesteps: list, rewards: list):
        for ind, action in enumerate(actions):
            self.learn(action, timesteps[ind], rewards[ind])


class EpsilonGreedy(Bandit):
    def __init__(self, T, batch_size, arm_interval,
                 warmup: int = 4,
                 epsilon: float = 0.1):
        super().__init__(T, batch_size, arm_interval)
        self.warmup = warmup
        self.epsilon = epsilon
        self.map = None

    def initialize(self):
        super(EpsilonGreedy, self).initialize()
        self.map = {}

    def get_arm_value(self) -> float:
        if len(self.active_arms) < self.warmup:
            self.random_exploration()
        else:
            if self.rg.uniform() < self.epsilon:
                self.random_exploration()
            else:
                self.greedy_exploitation()
        return self.active_arms[self.current_arm_idx]

    def learn(self, action: float, timestep: int, reward: float):
        self.map[action] = reward

    def greedy_exploitation(self):
        self.current_arm_idx = self.active_arms.index(max(self.map, key=self.map.get))


class ThompsonSampling(Bandit):
    def __init__(self, T, batch_size, arm_interval, init_policy: str = "Random"):
        super().__init__(T, batch_size, arm_interval)
        self.init_policy = init_policy
        self.a = None
        self.b = None

    def initialize(self):
        super(ThompsonSampling, self).initialize()
        n_arms = self.init_active_arms(self.init_policy)
        self.a = np.ones(n_arms) * 30
        self.b = np.ones(n_arms) * 30

    def get_arm_value(self) -> float:
        beta_params = zip(self.a, self.b)

        try:
            # Perform random draw for all arms based on their params (a,b)
            all_draws = [self.rg.beta(i[0], i[1]) for i in beta_params]  # might throw errors
            # TODO: refactor errors here

            # return index of arm with the highest draw
            self.current_arm_idx = all_draws.index(max(all_draws))
        except ValueError as err:
            print(f"A terrible error has occured in Thompson sampling! {err}")
        finally:
            return self.active_arms[self.current_arm_idx]

    def learn(self, action: float, timestep: int, reward: float):
        action_idx = self.active_arms.index(action) #  problem with scaling and inverse scaling: precision is lost
        # assert action_idx == self.current_arm_idx
        # TODO: this representation of successes and failures is deeply flawed on rewards from R1
        self.a[action_idx] += reward
        self.b[action_idx] += 1 - reward


class UCB(Bandit):
    def __init__(self, T, batch_size, arm_interval,
                 c: int = 2,
                 init_policy: str = "Random"):
        super().__init__(T, batch_size, arm_interval)
        self.c = c
        self.init_policy = init_policy
        self.counter = 0
        self.actions = None
        self.rewards = None

    def initialize(self):
        super(UCB, self).initialize()
        self.counter = 0
        n_arms = self.init_active_arms(self.init_policy)
        self.actions = np.zeros(n_arms)
        self.rewards = np.zeros(n_arms)

    def get_arm_value(self) -> float:
        self.counter += 1
        sample_mean = self.rewards / self.actions
        ucb = np.sqrt(self.c * np.log10(self.counter) / self.actions)
        p = sample_mean + ucb
        self.current_arm_idx = np.argmax(p)
        return self.active_arms[self.current_arm_idx]

    def learn(self, action: float, timestep: int, reward: float):
        action_idx = self.active_arms.index(action)
        # assert action_idx == self.current_arm_idx
        self.rewards[action_idx] += reward
        self.actions[action_idx] += 1
