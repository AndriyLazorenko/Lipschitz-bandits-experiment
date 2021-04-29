from typing import Tuple

from three_d.algorithms.abstract_algorithm import Algorithm


class Optimal(Algorithm):
    def __init__(self, T, batch_size, arm_interval, reward):
        super().__init__(T, batch_size, arm_interval)
        self.reward = reward

    def initialize(self):
        pass

    def get_arm_value(self) -> Tuple[float, float]:
        if self.reward == "triangular":
            return 1.0, 10.0
        elif self.reward == "quadratic":
            return 1.0, 1.0
        else:
            raise NotImplementedError(f"{self.reward.capitalize()} reward doesn't have optimal algorithm implemented!")

    def get_arms_batch(self) -> list:
        return [self.get_arm_value() for _ in range(self.batch_size)]

    def learn(self, action: float, timestep: int, reward: float):
        # Optimal algorithm doesn't need to learn. It knows
        pass

    def batch_learn(self, actions: list, timesteps: list, rewards: list):
        pass
