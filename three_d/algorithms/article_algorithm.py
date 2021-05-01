from typing import Tuple
import numpy as np
import pandas as pd
from three_d.algorithms.abstract_algorithm import Algorithm
from utils.paths import scenario
import utils.timer as timer
from time import sleep


class NLBS(Algorithm):

    def __init__(self, T, batch_size, arm_interval):
        super().__init__(T, batch_size, arm_interval)
        self.current_arm = None
        self.df = None

    def initialize(self):
        self.current_arm = (
            self.rg.uniform(1, 2),
            self.rg.uniform(1, 7)
        )

    def get_arm_value(self) -> Tuple:
        return self.current_arm

    def learn(self, _action: Tuple, timestep: int, _reward: float):
        self.df = pd.read_csv(scenario)
        df = self.df.loc[self.df.day == timestep].copy()
        vt = df.vt.to_numpy()
        mt = df.mt.to_numpy()
        arms_x = np.linspace(0, 2, 80)  # Grid search interval over parameter theta1
        arms_y = np.linspace(0.01, 7, 80)  # Grid search interval over parameter theta2
        x, y = np.meshgrid(arms_x, arms_y)
        # arms = ((x, y) for x in arms_x for y in arms_y)
        x = x.reshape(-1)
        y = y.reshape(-1)
        # print(x[0])
        # print(y[0])
        arms = np.vstack((x, y)).T
        rewards = []
        for arm in arms:
            bt = np.divide(np.log10(np.add(1, np.multiply(np.multiply(vt, arm[0]), arm[1]))), arm[1])
            mask = bt <= mt
            profit = np.subtract(vt, bt)
            masked_profit = np.ma.masked_where(mask, profit)
            reward = masked_profit.filled(0).sum()
            reward = np.nan_to_num(reward)
            # df['bt'] = np.log10(1 + df.vt * arm[0] * arm[1]) / arm[1]
            # df['profit'] = df.apply(lambda row: row.vt - row.bt if row.bt > row.mt else 0, axis=1)
            # reward = df.profit.sum()
            rewards.append((arm, reward))
        sorted_rewards = sorted(rewards, key=lambda x: x[1], reverse=True)
        self.current_arm = tuple(sorted_rewards[0][0])
        print(f"New arm on timestep {timestep} for NLBS: {self.current_arm}, current reward: {_reward}")

    def get_arms_batch(self) -> list:
        raise NotImplementedError(f"Batch processing is not implemented for {self.__name__()} algorithm")

    def batch_learn(self, actions: list, timesteps: list, rewards: list):
        raise NotImplementedError(f"Batch processing is not implemented for {self.__name__()} algorithm")
