import abc
import numpy as np
import pandas as pd
from scipy.stats import pareto


class Rewards(metaclass=abc.ABCMeta):
    def __init__(self,
                 df: pd.DataFrame,
                 reward_type: str,
                 ):
        self.df = df
        self.reward_type = reward_type

    @staticmethod
    @abc.abstractmethod
    def triangular_reward(arm):
        pass

    @staticmethod
    @abc.abstractmethod
    def quadratic_reward(arm):
        pass

    @abc.abstractmethod
    def article_reward(self, arm, timestep: int):
        pass

    def get_reward(self, arm: float, timestep: int) -> float:
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
        elif self.reward_type == "article":
            return self.article_reward(arm, timestep)
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
