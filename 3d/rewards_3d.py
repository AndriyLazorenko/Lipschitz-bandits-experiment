import numpy as np
import pandas as pd
from scipy.stats import pareto


class Rewards3D:
    # TODO: refactor completely
    def __init__(self,
                 df: pd.DataFrame,
                 reward_type: str,
                 ):
        self.df = df
        self.reward_type = reward_type

    @staticmethod
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

    def get_reward(self, arm: float, timestep: int) -> float:
        """
        A method that routes reward calculation
        Args:
            arm: float:

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
