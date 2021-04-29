from typing import Tuple
import numpy as np
import pandas as pd
from three_d.algorithms.abstract_algorithm import Algorithm
from utils.paths import scenario


class Article(Algorithm):

    def __init__(self, T, batch_size, arm_interval):
        super().__init__(T, batch_size, arm_interval)
        self.current_arm = None
        self.df = None

    def initialize(self):
        self.current_arm = (
            self.rg.uniform(self.arm_intervals[0][0], self.arm_intervals[0][1]),
            self.rg.uniform(self.arm_intervals[1][0], self.arm_intervals[1][0])
        )

    def get_arm_value(self) -> Tuple:
        return self.current_arm

    def learn(self, _action: Tuple, timestep: int, _reward: float):
        self.df = pd.read_csv(scenario)
        df = self.df.loc[self.df.day == timestep].copy()
        arms_x = np.linspace(0, 1, 30)
        arms_y = np.logspace(-2, 1, 30)
        arms = ((x, y) for x in arms_x for y in arms_y)
        rewards = []
        for arm in arms:
            df['bt'] = np.log10(1 + df.vt * arm[0] * arm[1]) / arm[1]
            df['profit'] = df.apply(lambda row: row.vt - row.bt if row.bt > row.mt else 0, axis=1)
            reward = df.profit.sum()
            rewards.append((arm, reward))
        sorted_rewards = sorted(rewards, key=lambda x: x[1], reverse=True)
        self.current_arm = sorted_rewards[0][0]

    def get_arms_batch(self) -> list:
        raise NotImplementedError(f"Batch processing is not implemented for {self.__name__()} algorithm")

    def batch_learn(self, actions: list, timesteps: list, rewards: list):
        raise NotImplementedError(f"Batch processing is not implemented for {self.__name__()} algorithm")
