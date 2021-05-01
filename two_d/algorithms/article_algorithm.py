import numpy as np
import pandas as pd
from two_d.algorithms.abstract_algorithm import Algorithm
from utils.paths import scenario
import utils.timer as timer


class LBS(Algorithm):
    def __init__(self, T, batch_size, arm_interval):
        super().__init__(T, batch_size, arm_interval)
        self.current_arm = None
        self.df = None

    def initialize(self):
        self.current_arm = self.rg.uniform(self.arm_interval[0], self.arm_interval[1])

    def get_arm_value(self) -> float:
        return self.current_arm

    def learn(self, _action: float, timestep: int, _reward: float):
        self.df = pd.read_csv(scenario)
        df = self.df.loc[self.df.day == timestep].copy()
        vt = df.vt.to_numpy()
        mt = df.mt.to_numpy()
        arms = np.divide(np.linspace(1, 100, num=200), 100)
        rewards = []
        for arm in arms:
            bt = np.multiply(vt, arm)
            mask = bt <= mt
            profit = np.subtract(vt, bt)
            masked_profit = np.ma.masked_where(mask, profit)
            reward = masked_profit.filled(0).sum()
            reward = np.nan_to_num(reward)
            # df['bt'] = df.vt * arm
            # df['profit'] = df.apply(lambda row: row.vt - row.bt if row.bt > row.mt else 0, axis=1)
            # rew = df.profit.sum()
            rewards.append((arm, reward))
        sorted_rewards = sorted(rewards, key=lambda x: x[1], reverse=True)
        self.current_arm = sorted_rewards[0][0]
        print(f"New arm for LBS: {self.current_arm}, current reward: {_reward}")

    def get_arms_batch(self) -> list:
        raise NotImplementedError(f"Batch processing is not implemented for {self.__name__()} algorithm")

    def batch_learn(self, actions: list, timesteps: list, rewards: list):
        raise NotImplementedError(f"Batch processing is not implemented for {self.__name__()} algorithm")
