from two_d.algorithms.abstract_algorithm import Algorithm


class Random(Algorithm):
    def __init__(self, T, batch_size, arm_interval):
        super().__init__(T, batch_size, arm_interval)

    def initialize(self):
        # Random algorithm doesn't initialize anything
        pass

    def get_arm_value(self) -> float:
        return self.inverse_interval_scaler(self.rg.uniform())

    def get_arms_batch(self) -> list:
        return [self.get_arm_value() for _ in range(self.batch_size)]

    def learn(self, action: float, timestep: int, reward: float):
        # Random algorithm doesn't learn anything
        pass

    def batch_learn(self, actions: list, timesteps: list, rewards: list):
        pass
