import abc
from typing import Tuple

import numpy as np


class Algorithm(metaclass=abc.ABCMeta):
    def __init__(self, T, batch_size: int = 4, arm_intervals=None):
        if arm_intervals is None:
            arm_intervals = [(0., 1.), (0., 10.)]
        self.T = T
        self.arm_intervals = arm_intervals
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
    def get_arm_value(self) -> Tuple:
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
    def learn(self, action: Tuple, timestep: int, reward: float):
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

    def interval_scaler(self, arms: tuple) -> tuple:
        """
        Transforms arms from custom intervals to [0,1] x [0,10] interval
        Args:
            arms: tuple

        Returns:
            scaled_arms: tuple

        """
        scaled_arms = list()
        for i in range(2):
            arm_interval = self.arm_intervals[i]
            interval_length = arm_interval[1] - arm_interval[0]
            scaled_arm = (arms[i] - arm_interval[0]) / interval_length
            if i == 1:
                scaled_arm = scaled_arm * 10
            scaled_arms.append(scaled_arm)
        return tuple(scaled_arms)

    def inverse_interval_scaler(self, scaled_arms: tuple) -> tuple:
        """
        Transforms arms from scaled interval [0,1] x [0,10] to custom intervals
        Args:
            scaled_arms: tuple:

        Returns:
            arms: tuple:

        """
        arms = list()
        for i in range(2):
            arm_interval = self.arm_intervals[i]
            interval_length = arm_interval[1] - arm_interval[0]
            scaled_arm = scaled_arms[i]
            if i == 1:
                scaled_arm = scaled_arm / 10
            arm = scaled_arm * interval_length + arm_interval[0]
            arms.append(arm)
        return tuple(arms)
