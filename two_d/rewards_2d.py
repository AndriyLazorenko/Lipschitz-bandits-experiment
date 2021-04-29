import numpy as np
import pandas as pd

from utils.rewards import Rewards


class Rewards2D(Rewards):
    def __init__(self, df: pd.DataFrame, reward_type: str):
        super().__init__(df, reward_type)

    @staticmethod
    def quadratic_reward(arm: float) -> float:
        return max(0.1, 0.9 - 3.2 * (0.7 - arm) ** 2)

    @staticmethod
    def triangular_reward(arm: float) -> float:
        # custom regret function, triangular regret is selected
        instant_regret = min(abs(np.subtract(arm, 0.4)), abs(np.subtract(arm, 0.8)))
        instant_reward = np.negative(instant_regret)
        return instant_reward

    def article_reward(self, arm: float, timestep: int) -> float:
        df = self.df.loc[self.df.day == timestep].copy()
        df['bt'] = df.vt * arm
        df['profit'] = df.apply(lambda row: row.vt - row.bt if row.bt > row.mt else 0, axis=1)
        return df.profit.sum()
