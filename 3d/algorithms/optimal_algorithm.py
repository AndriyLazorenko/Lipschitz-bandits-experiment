from algorithms.abstract_algorithm import Algorithm


class Optimal(Algorithm):
    # TODO: refactor after simulation reward functions are reengineered
    def __init__(self, T, batch_size, arm_interval, reward):
        super().__init__(T, batch_size, arm_interval)
        self.reward = reward

    def initialize(self):
        pass

    def get_arm_value(self) -> float:
        if self.reward == "triangular":
            return 0.4
        elif self.reward == "quadratic":
            return 0.7
        else:
            raise NotImplementedError("Such reward doesn't have optimal algorithm implemented!")

    def get_arms_batch(self) -> list:
        return [self.get_arm_value() for _ in range(self.batch_size)]

    def learn(self, action: float, timestep: int, reward: float):
        # Optimal algorithm doesn't need to learn. It knows
        pass

    def batch_learn(self, actions: list, timesteps: list, rewards: list):
        pass
