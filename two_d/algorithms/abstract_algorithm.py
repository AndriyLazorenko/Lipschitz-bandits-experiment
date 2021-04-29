import abc

import numpy as np


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
        Transforms arm from scaled interval [0,1] to custom interval
        Args:
            scaled_arm: float:

        Returns:
            arm: float:

        """
        interval_length = self.arm_interval[1] - self.arm_interval[0]
        arm = scaled_arm * interval_length + self.arm_interval[0]
        return arm
