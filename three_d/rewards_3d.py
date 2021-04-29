import numpy as np
import pandas as pd

from typing import Tuple

from utils.rewards import Rewards


class Rewards3D(Rewards):
    def __init__(self, df: pd.DataFrame, reward_type: str):
        super().__init__(df, reward_type)

    @staticmethod
    def quadratic_reward(arms: Tuple[float, float]) -> float:
        return -100 * (arms[1] - arms[0] ** 2) ** 2 + (1 - arms[0]) ** 2  # Rosenbrock function

    @staticmethod
    def triangular_reward(arms: Tuple[float, float]) -> float:
        return -100 * np.sqrt(np.abs(arms[0] - 0.01*arms[1]**2)) + 0.01 * np.abs(10 - arms[1])  # Bukin function #6

    def article_reward(self, arms: Tuple[float, float], timestep: int) -> float:
        df = self.df.loc[self.df.day == timestep].copy()
        df['bt'] = np.log10(1 + df.vt * arms[0] * arms[1])/arms[1]
        df['profit'] = df.apply(lambda row: row.vt - row.bt if row.bt > row.mt else 0, axis=1)
        return df.profit.sum()
