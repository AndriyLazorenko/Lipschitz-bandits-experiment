from abc import ABCMeta
from two_d.algorithms.abstract_algorithm import Algorithm
import numpy as np


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
