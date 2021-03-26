import abc
from abc import ABCMeta
from time import sleep

import numpy as np
from skopt import Optimizer


class Algorithm(metaclass=abc.ABCMeta):
    def __init__(self, T):
        self.T = T
        self.active_arms = None

    @abc.abstractmethod
    def initialize(self):
        """
        A method called to (re)set all fields and data structures

        """
        pass

    @abc.abstractmethod
    def get_arm_index(self) -> int:
        """
        A method retrieving the arm index based on the algorithm's logic
        Returns:
            arm_index: int

        """
        pass

    @abc.abstractmethod
    def learn(self, timestep: int, reward: float):
        """
        A method implementing "learning" of the algorithm based on reward obtained from an arm pull.
        Args:
            timestep: int:
            reward: float:

        """
        pass


class ZoomingAlgorithm(Algorithm, metaclass=ABCMeta):  # abstract class of `ZoomingAlgorithm`
    def __init__(self, delta, T, c):
        super().__init__(T)
        self.delta = delta
        self.c = c
        self.pulled_idx = None
        self.mu = None
        self.n = None
        self.r = None

    def get_uncovered(self):
        covered = [[arm - r, arm + r] for arm, r in zip(self.active_arms, self.r)]
        if not covered:
            return [0, 1]
        covered.sort(key=lambda x: x[0])
        low = 0
        for interval in covered:
            if interval[0] <= low:
                low = max(low, interval[1])
                if low >= 1:
                    return None
            else:
                return [low, interval[0]]
        return [low, 1]

    def compute_arm_index(self):
        uncovered = self.get_uncovered()
        if uncovered is None:
            score = [mu + 2 * r for mu, r in zip(self.mu, self.r)]
            self.pulled_idx = np.argmax(score)
        else:
            new_arm = np.random.uniform(*uncovered)
            self.active_arms.append(new_arm)
            self.mu.append(0)
            self.n.append(0)
            self.r.append(0)
            self.pulled_idx = len(self.active_arms) - 1


class Random(Algorithm):
    def get_arm_index(self):
        if self.active_arms:
            self.active_arms.pop()
        self.active_arms.append(np.random.uniform())
        return 0

    def __init__(self, T):
        super().__init__(T)

    def initialize(self):
        self.active_arms = []

    def learn(self, timestep: int, reward: float):
        # Random algorithm doesn't learn anything
        pass


class BayesianOptimization(Algorithm):
    def __init__(self, T,
                 dim: tuple = (0.0, 1.0),
                 warmup: int = 4,
                 acq_func: str = "LCB"):
        super().__init__(T)
        self.dim = dim
        self.points = warmup
        self.acq_func = acq_func
        self.opt = None
        self.current_arm_idx = 0

    def initialize(self):
        self.current_arm_idx = 0
        self.active_arms = []
        self.opt = Optimizer(dimensions=[self.dim],
                             n_initial_points=self.points,
                             acq_func=self.acq_func)

    def learn(self, timestep: int, reward: float):
        self.opt.tell([self.active_arms[self.current_arm_idx]], -reward)

    def get_arm_index(self):
        self.active_arms.append(self.opt.ask()[0])
        self.current_arm_idx = len(self.active_arms) - 1
        return self.current_arm_idx


class Bandit(Algorithm, metaclass=ABCMeta):
    def __init__(self, T):
        super().__init__(T)
        self.current_arm_idx = 0

    def initialize(self):
        self.current_arm_idx = 0
        self.active_arms = []

    def random_exploration(self):
        self.active_arms.append(np.random.uniform())
        return len(self.active_arms) - 1


class ThompsonSampling(Bandit):
    def __init__(self, T, init_policy: str = "Random"):
        super().__init__(T)
        self.init_policy = init_policy
        self.a = None
        self.b = None

    def initialize(self):
        super(ThompsonSampling, self).initialize()
        n_arms = int(np.power(self.T, 1 / 2))
        if self.init_policy == "Random":
            self.active_arms = [np.random.uniform() for _ in range(n_arms)]
        elif self.init_policy == "Equidistant":
            self.active_arms = [n / (n_arms - 1) for n in range(n_arms)]
        self.a = np.ones(n_arms) * 30
        self.b = np.ones(n_arms) * 30

    def get_arm_index(self):
        beta_params = zip(self.a, self.b)
        # Perform random draw for all arms based on their params (a,b)
        all_draws = [np.random.beta(i[0], i[1]) for i in beta_params]

        # return index of arm with the highest draw
        self.current_arm_idx = all_draws.index(max(all_draws))
        return self.current_arm_idx

    def learn(self, timestep: int, reward: float):
        self.a[self.current_arm_idx] += reward
        self.b[self.current_arm_idx] += 1-reward


class UCB(Bandit):
    def __init__(self, T, c: int = 2, init_policy: str = "Random"):
        super().__init__(T)
        self.c = c
        self.init_policy = init_policy
        self.counter = 0
        self.actions = None
        self.rewards = None

    def initialize(self):
        super(UCB, self).initialize()
        self.counter = 0
        n_arms = int(np.power(self.T, 1 / 2))
        if self.init_policy == "Random":
            self.active_arms = [np.random.uniform() for _ in range(n_arms)]
        elif self.init_policy == "Equidistant":
            self.active_arms = [n / (n_arms - 1) for n in range(n_arms)]
        self.actions = np.zeros(n_arms)
        self.rewards = np.zeros(n_arms)

    def get_arm_index(self):
        self.counter += 1
        sample_mean = self.rewards / self.actions
        ucb = np.sqrt(self.c * np.log10(self.counter) / self.actions)
        p = sample_mean + ucb
        self.current_arm_idx = np.argmax(p)
        return self.current_arm_idx

    def learn(self, timestep: int, reward: float):
        self.rewards[self.current_arm_idx] += reward
        self.actions[self.current_arm_idx] += 1


class EpsilonGreedy(Bandit):
    def __init__(self, T, warmup: int = 4, epsilon: float = 0.1):
        super().__init__(T)
        self.warmup = warmup
        self.epsilon = epsilon
        self.map = None

    def initialize(self):
        super(EpsilonGreedy, self).initialize()
        self.map = {}

    def learn(self, timestep: int, reward: float):
        self.map[self.active_arms[self.current_arm_idx]] = reward

    def get_arm_index(self):
        if len(self.active_arms) < self.warmup:
            self.current_arm_idx = self.random_exploration()
        else:
            if np.random.uniform() < self.epsilon:
                self.current_arm_idx = self.random_exploration()
            else:
                self.current_arm_idx = self.greedy_exploitation()
        return self.current_arm_idx

    def greedy_exploitation(self):
        return self.active_arms.index(max(self.map, key=self.map.get))

    @staticmethod
    def argmax(iterable):
        return max(enumerate(iterable), key=lambda x: x[1])[0]


class Optimal(Algorithm):
    def __init__(self, T, reward):
        super().__init__(T)
        self.reward = reward

    def initialize(self):
        self.active_arms = []

    def get_arm_index(self):
        if self.active_arms:
            self.active_arms.pop()
        if self.reward == "triangular":
            self.active_arms.append(0.4)
        elif self.reward == "quadratic":
            self.active_arms.append(0.7)
        return 0

    def learn(self, timestep: int, reward: float):
        # Optimal algorithm doesn't learn. It knows
        pass


class Zooming(ZoomingAlgorithm):
    def __init__(self, delta, T, c, nu):
        super().__init__(delta, T, c)
        self.nu = nu

    def get_arm_index(self):
        self.compute_arm_index()
        return self.pulled_idx

    def initialize(self):
        self.active_arms = []
        self.mu = []
        self.n = []
        self.r = []

    def learn(self, timestep: int, reward: float):
        idx = self.pulled_idx
        self.mu[idx] = (self.mu[idx] * self.n[idx] + reward) / (self.n[idx] + 1)
        self.n[idx] += 1
        for i, n in enumerate(self.n):
            self.r[i] = self.c * self.nu * np.power(timestep, 1 / 3) / np.sqrt(n)


class ADTM(ZoomingAlgorithm):
    def __init__(self, delta, T, c, nu, epsilon):
        super().__init__(delta, T, c)
        self.nu = nu
        self.epsilon = epsilon

    def initialize(self):
        self.active_arms = []
        self.mu = []
        self.n = []
        self.r = []

    def get_arm_index(self):
        self.compute_arm_index()
        return self.pulled_idx

    def learn(self, timestep, reward):
        idx = self.pulled_idx
        threshold = np.power(self.nu * (self.n[idx] + 1) / np.log(self.T ** 2 / self.delta), 1 / (1 + self.epsilon))
        if abs(reward) > threshold:
            reward = 0
        self.mu[idx] = (self.mu[idx] * self.n[idx] + reward) / (self.n[idx] + 1)
        self.n[idx] += 1
        self.r[idx] = self.c * 4 * np.power(self.nu, 1 / (1 + self.epsilon)) * np.power(
            np.log(self.T ** 2 / self.delta) / self.n[idx], self.epsilon / (1 + self.epsilon))


class ADMM(ZoomingAlgorithm):
    def __init__(self, delta, T, c, sigma, epsilon):
        super().__init__(delta, T, c)
        self.sigma = sigma
        self.epsilon = epsilon
        self.h = None
        self.replay = None

    def initialize(self):
        self.active_arms = []
        self.mu = []
        self.n = []
        self.r = []
        self.h = []
        self.replay = False

    def get_arm_index(self):
        if self.replay:
            pass  # remain `self.pulled_idx` unchanged
        else:
            uncovered = self.get_uncovered()
            if uncovered is None:
                score = [mu + 2 * r for mu, r in zip(self.mu, self.r)]
                self.pulled_idx = np.argmax(score)
            else:
                new_arm = np.random.uniform(*uncovered)
                self.active_arms.append(new_arm)
                self.mu.append(0)
                self.n.append(0)
                self.r.append(0)
                self.h.append([])
                self.pulled_idx = len(self.active_arms) - 1
        return self.pulled_idx

    def learn(self, timestep, reward):
        def MME(rewards):
            M = int(np.floor(8 * np.log(self.T ** 2 / self.delta) + 1))
            B = int(np.floor(len(rewards) / M))
            means = np.zeros(M)
            for m in range(M):
                means[m] = np.mean(rewards[m * B:(m + 1) * B])
            return np.median(means)

        idx = self.pulled_idx
        self.h[idx].append(reward)
        self.n[idx] += 1
        self.r[idx] = self.c * np.power(12 * self.sigma, 1 / (1 + self.epsilon)) * np.power(
            (16 * np.log(self.T ** 2 / self.delta) + 2) / self.n[idx], self.epsilon / (1 + self.epsilon))
        if self.n[idx] < 16 * np.log(self.T ** 2 / self.delta) + 2:
            self.replay = True
        else:
            self.replay = False
            self.mu[idx] = MME(self.h[idx])
