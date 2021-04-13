import abc
from abc import ABCMeta
from time import sleep

import numpy as np
from skopt import Optimizer


class Algorithm(metaclass=abc.ABCMeta):
    def __init__(self, T, batch_size: int = 4, arm_interval: tuple = (0., 1.)):
        self.T = T
        self.arm_interval = arm_interval
        self.rg = np.random.default_rng()
        self.active_arms = None
        self.batch_size = batch_size

    @abc.abstractmethod
    def initialize(self):
        """
        A method called to (re)set all fields and data structures

        """
        pass

    @abc.abstractmethod
    def get_arm_value(self) -> float:
        """
        A method retrieving a specific arm's value based on the algorithm's logic

        Returns:
            multiplier: float

        """
        pass

    @abc.abstractmethod
    def get_arms_batch(self) -> list:
        """
        A method retrieving a batch of arms based on algorithm's logic

        Returns:
            batch: list

        """
        pass

    @abc.abstractmethod
    def learn(self, action: float, timestep: int, reward: float):
        """
        A method implementing "learning" of the algorithm based on reward obtained from an arm pull.
        Args:
            action: float:
            timestep: int:
            reward: float:

        """
        pass

    @abc.abstractmethod
    def batch_learn(self, actions: list, timesteps: list, rewards: list):
        """
        A method implementing "batch learning" of the algorithm based on rewards obtained from batch of arm pulls
        Args:
            actions: list:
            timesteps: list:
            rewards: list

        Returns:

        """
        pass

    def interval_scaler(self, arm: float) -> float:
        """
        Transforms arm from custom interval to [0,1] interval
        Args:
            arm: float:

        Returns:
            scaled_arm: float

        """
        interval_length = self.arm_interval[1] - self.arm_interval[0]
        scaled_arm = (arm - self.arm_interval[0]) / interval_length
        return scaled_arm

    def inverse_interval_scaler(self, scaled_arm: float) -> float:
        """
        Transorms arm from scaled interval [0,1] to custom interval
        Args:
            scaled_arm: float:

        Returns:
            arm: float:

        """
        interval_length = self.arm_interval[1] - self.arm_interval[0]
        arm = scaled_arm * interval_length + self.arm_interval[0]
        return arm


class Random(Algorithm):
    def __init__(self, T, batch_size, arm_interval):
        super().__init__(T, batch_size, arm_interval)

    def initialize(self):
        # Random algorithm doesn't initialize anything
        pass

    def get_arm_value(self) -> float:
        return self.inverse_interval_scaler(self.rg.uniform())

    def get_arms_batch(self) -> list:
        return [self.get_arm_value() for _ in range(self.batch_size)]

    def learn(self, action: float, timestep: int, reward: float):
        # Random algorithm doesn't learn anything
        pass

    def batch_learn(self, actions: list, timesteps: list, rewards: list):
        pass


class Optimal(Algorithm):
    def __init__(self, T, batch_size, arm_interval, reward):
        super().__init__(T, batch_size, arm_interval)
        self.reward = reward

    def initialize(self):
        pass

    def get_arm_value(self) -> float:
        if self.reward == "triangular":
            return 0.4
        elif self.reward == "quadratic":
            return 0.7
        else:
            raise NotImplementedError("Such reward doesn't have optimal algorithm implemented!")

    def get_arms_batch(self) -> list:
        return [self.get_arm_value() for _ in range(self.batch_size)]

    def learn(self, action: float, timestep: int, reward: float):
        # Optimal algorithm doesn't need to learn. It knows
        pass

    def batch_learn(self, actions: list, timesteps: list, rewards: list):
        pass


class BayesianOptimization(Algorithm):
    def __init__(self, T, batch_size, arm_interval,
                 warmup: int = 4,
                 acq_func: str = "LCB"):
        super().__init__(T, batch_size, arm_interval)
        self.warmup = warmup + 1
        self.acq_func = acq_func
        self.opt = None
        self.current_arm_idx = 0

    def initialize(self):
        self.current_arm_idx = 0
        self.active_arms = []
        self.opt = Optimizer(dimensions=[self.arm_interval],
                             n_initial_points=self.warmup,
                             acq_func=self.acq_func)

    def get_arm_value(self) -> float:
        self.active_arms.append(self.opt.ask()[0])
        self.current_arm_idx = len(self.active_arms) - 1
        return self.active_arms[self.current_arm_idx]

    def get_arms_batch(self) -> list:
        assert self.warmup > self.batch_size
        batch = self.opt.ask(n_points=self.batch_size)
        batch = [a[0] for a in batch]
        return batch

    def learn(self, action: float, timestep: int, reward: float):
        self.opt.tell([action], -reward)

    def batch_learn(self, actions: list, timesteps: list, rewards: list):
        regrets = [-rew for rew in rewards]
        actions = [[action] for action in actions]
        self.opt.tell(actions, regrets)


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


class ZoomingAlgorithm(Algorithm, metaclass=ABCMeta):  # abstract class of `ZoomingAlgorithm`
    def __init__(self, T, batch_size, arm_interval, delta, c):
        super().__init__(T, batch_size, arm_interval)
        self.delta = delta
        self.c = c
        self.pulled_idx = None
        self.mu = None
        self.n = None
        self.r = None

    def get_uncovered(self):
        covered = [[arm - r, arm + r] for arm, r in zip(self.active_arms, self.r)]
        if not covered:
            return [0, 1]  # self.arm_interval[0], self.arm_interval[1]
        covered.sort(key=lambda x: x[0])
        low = 0  # self.arm_interval[0]
        for interval in covered:
            if interval[0] <= low:
                low = max(low, interval[1])
                if low >= 1:  # self.arm_interval[1]
                    return None
            else:
                return [low, interval[0]]
        return [low, 1]  # self.arm_interval[1]

    def compute_arm_index(self):
        uncovered = self.get_uncovered()
        if uncovered is None:
            score = [mu + 2 * r for mu, r in zip(self.mu, self.r)]
            self.pulled_idx = np.argmax(score)
        else:
            new_arm = self.rg.uniform(*uncovered)
            new_arm = self.inverse_interval_scaler(new_arm)  # Interval transform
            self.active_arms.append(new_arm)
            self.mu.append(0)
            self.n.append(0)
            self.r.append(0)
            self.pulled_idx = len(self.active_arms) - 1

    def get_arms_batch(self) -> list:
        return [self.get_arm_value() for _ in range(self.batch_size)]

    def batch_learn(self, actions: list, timesteps: list, rewards: list):
        for ind, action in enumerate(actions):
            self.learn(action, timesteps[ind], rewards[ind])


class Zooming(ZoomingAlgorithm):
    def __init__(self, T, batch_size, arm_interval, delta, c, nu):
        super().__init__(T, batch_size, arm_interval, delta, c)
        self.nu = nu

    def initialize(self):
        self.active_arms = []
        self.mu = []
        self.n = []
        self.r = []

    def get_arm_value(self) -> float:
        self.compute_arm_index()
        print(self.active_arms)
        sleep(5)
        return self.active_arms[self.pulled_idx]

    def learn(self, action: float, timestep: int, reward: float):
        action_idx = self.active_arms.index(action)
        # assert action_idx == self.pulled_idx
        self.mu[action_idx] = (self.mu[action_idx] * self.n[action_idx] + reward) / (self.n[action_idx] + 1)
        self.n[action_idx] += 1
        for i, n in enumerate(self.n):
            self.r[i] = self.c * self.nu * np.power(timestep, 1 / 3) / np.sqrt(n)


class ADTM(ZoomingAlgorithm):
    def __init__(self, T, batch_size, arm_interval, delta, c, nu, epsilon):
        super().__init__(T, batch_size, arm_interval, delta, c)
        self.nu = nu
        self.epsilon = epsilon

    def initialize(self):
        self.active_arms = []
        self.mu = []
        self.n = []
        self.r = []

    def get_arm_value(self) -> float:
        self.compute_arm_index()
        return self.active_arms[self.pulled_idx]

    def learn(self, action: float, timestep: int, reward: float):
        action_idx = self.active_arms.index(action)
        threshold = np.power(self.nu * (self.n[action_idx] + 1) / np.log(self.T ** 2 / self.delta),
                             1 / (1 + self.epsilon))
        if abs(reward) > threshold:
            reward = 0
        self.mu[action_idx] = (self.mu[action_idx] * self.n[action_idx] + reward) / (self.n[action_idx] + 1)
        self.n[action_idx] += 1
        self.r[action_idx] = self.c * 4 * np.power(self.nu, 1 / (1 + self.epsilon)) * np.power(
            np.log(self.T ** 2 / self.delta) / self.n[action_idx], self.epsilon / (1 + self.epsilon))


class ADMM(ZoomingAlgorithm):
    def __init__(self, T, batch_size, arm_interval, delta, c, sigma, epsilon):
        super().__init__(T, batch_size, arm_interval, delta, c)
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

    def get_arm_value(self) -> float:
        if self.replay:
            pass  # remain `self.pulled_idx` unchanged
        else:
            uncovered = self.get_uncovered()
            if uncovered is None:
                score = [mu + 2 * r for mu, r in zip(self.mu, self.r)]
                self.pulled_idx = np.argmax(score)
            else:
                new_arm = self.rg.uniform(*uncovered)
                new_arm = self.inverse_interval_scaler(new_arm)  # Inverse interval
                self.active_arms.append(new_arm)
                self.mu.append(0)
                self.n.append(0)
                self.r.append(0)
                self.h.append([])
                self.pulled_idx = len(self.active_arms) - 1
        return self.active_arms[self.pulled_idx]

    def learn(self, action: float, timestep: int, reward: float):
        def MME(rewards):
            M = int(np.floor(8 * np.log(self.T ** 2 / self.delta) + 1))
            B = int(np.floor(len(rewards) / M))
            means = np.zeros(M)
            for m in range(M):
                means[m] = np.mean(rewards[m * B:(m + 1) * B])
            return np.median(means)

        action_idx = self.active_arms.index(action)
        self.h[action_idx].append(reward)
        self.n[action_idx] += 1
        self.r[action_idx] = self.c * np.power(12 * self.sigma, 1 / (1 + self.epsilon)) * np.power(
            (16 * np.log(self.T ** 2 / self.delta) + 2) / self.n[action_idx], self.epsilon / (1 + self.epsilon))
        if self.n[action_idx] < 16 * np.log(self.T ** 2 / self.delta) + 2:
            self.replay = True
        else:
            self.replay = False
            self.mu[action_idx] = MME(self.h[action_idx])
