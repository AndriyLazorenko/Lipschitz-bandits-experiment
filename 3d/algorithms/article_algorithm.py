import pandas as pd
from algorithms.abstract_algorithm import Algorithm
from utils.paths import scenario


class Article(Algorithm):

    # TODO: refactor completely
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
        arms = [(i+1)/100 for i in range(100)]
        rewards = []
        for arm in arms:
            df['bt'] = df.vt * arm
            df['profit'] = df.apply(lambda row: row.vt - row.bt if row.bt > row.mt else 0, axis=1)
            reward = df.profit.sum()
            rewards.append((arm, reward))
        sorted_rewards = sorted(rewards, key=lambda x: x[1], reverse=True)
        self.current_arm = sorted_rewards[0][0]

    def get_arms_batch(self) -> list:
        raise NotImplementedError(f"Batch processing is not implemented for {self.__name__()} algorithm")

    def batch_learn(self, actions: list, timesteps: list, rewards: list):
        raise NotImplementedError(f"Batch processing is not implemented for {self.__name__()} algorithm")
