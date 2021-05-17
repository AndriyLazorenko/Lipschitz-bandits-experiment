import numpy as np
import pandas as pd
from scipy.stats import pareto
from typing import Tuple
# from utils import timer


class Rewards:
    def __init__(self,
                 df: pd.DataFrame,
                 reward_type: str,
                 ):
        self.df = df
        self.reward_type = reward_type

    @staticmethod
    def quadratic_reward(arm: float) -> float:
        return max(0.1, 0.9 - 3.2 * (0.7 - arm) ** 2)

    @staticmethod
    def triangular_reward(arm: float) -> float:
        # custom regret function, triangular regret is selected
        instant_regret = min(abs(np.subtract(arm, 0.4)), abs(np.subtract(arm, 0.8)))
        instant_reward = np.negative(instant_regret)
        return instant_reward

    @staticmethod
    def rosenbrock_reward(arms: Tuple[float, float]) -> float:
        return -100 * (arms[1] - arms[0] ** 2) ** 2 + (1 - arms[0]) ** 2  # Rosenbrock function

    @staticmethod
    def bukin_reward(arms: Tuple[float, float]) -> float:
        return -100 * np.sqrt(np.abs(arms[0] - 0.01 * arms[1] ** 2)) + 0.01 * np.abs(10 - arms[1])  # Bukin function #6

    def article_LBS_reward(self, arm: float, timestep: int) -> float:
        df = self.df.loc[self.df.day == timestep].copy()
        vt = df.vt.to_numpy()
        mt = df.mt.to_numpy()
        bt = np.multiply(vt, arm)
        mask = bt <= mt
        profit = np.subtract(vt, bt)
        masked_profit = np.ma.masked_where(mask, profit)
        reward = masked_profit.filled(0).sum()
        reward = np.nan_to_num(reward)
        # df['bt'] = df.vt * arm
        # df['profit'] = df.apply(lambda row: row.vt - row.bt if row.bt > row.mt else 0, axis=1)
        # rew = df.profit.sum()
        # assert reward == rew
        return reward

    def article_NLBS_reward(self, arms: Tuple[float, float], timestep: int) -> float:
        df = self.df.loc[self.df.day == timestep].copy()
        vt = df.vt.to_numpy()
        mt = df.mt.to_numpy()
        bt = np.divide(np.log10(np.add(1, np.multiply(np.multiply(vt, arms[0]), arms[1]))), arms[1])
        mask = bt <= mt
        profit = np.subtract(vt, bt)
        masked_profit = np.ma.masked_where(mask, profit)
        reward = masked_profit.filled(0).sum()
        reward = np.nan_to_num(reward)
        # df['bt'] = np.log10(1 + df.vt * arms[0] * arms[1]) / arms[1]
        # df['profit'] = df.apply(lambda row: row.vt - row.bt if row.bt > row.mt else 0, axis=1)
        # rew = df.profit.sum()
        # print(reward, rew)
        # assert rew == reward
        return reward

    def get_reward(self, arm, timestep: int) -> float:
        """
        A method that routes reward calculation
        Args:
            arm: float:
            timestep: int

        Returns:

        """
        if self.reward_type == "triangular":
            return self.triangular_reward(arm)
        elif self.reward_type == "quadratic":
            return self.quadratic_reward(arm)
        elif self.reward_type == "article_LBS":
            return self.article_LBS_reward(arm, timestep)
        elif self.reward_type == "bukin":
            return self.bukin_reward(arm)
        elif self.reward_type == "rosenbrock":
            return self.rosenbrock_reward(arm)
        elif self.reward_type == "article_NLBS":
            return self.article_NLBS_reward(arm, timestep)
        elif self.reward_type == "article":
            if isinstance(arm, tuple):
                return self.article_NLBS_reward(arm, timestep)
            elif isinstance(arm, list):
                return self.article_NLBS_reward(tuple(arm), timestep)
            elif isinstance(arm, float):
                return self.article_LBS_reward(arm, timestep)
            else:
                raise IOError(f"Incorrect input: {arm} of type: {type(arm)}")
        else:
            raise NotImplementedError(f"'{self.reward_type.capitalize()}' reward type is not implemented")


def augment_reward(reward: float,
                   stochasticity: bool,
                   alpha: float,
                   action_cost: int,
                   heavy_tails: bool = True,
                   noise_modulation: float = .3
                   ) -> float:
    """
    A method to augment reward
    Args:
        reward: float
        stochasticity: bool:
        alpha: float:
        action_cost: int:
        heavy_tails: bool:
        noise_modulation: float:

    Returns:
        reward: float

    """
    if stochasticity:
        if heavy_tails:
            stochastic_factor = pareto.rvs(alpha) - alpha / (alpha - 1)
            reward += stochastic_factor
        else:
            stochastic_factor = np.random.uniform(1 - noise_modulation, 1 + noise_modulation)
            reward *= stochastic_factor
    reward -= action_cost
    return reward
