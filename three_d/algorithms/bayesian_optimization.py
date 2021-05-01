from three_d.algorithms.abstract_algorithm import Algorithm
from skopt import Optimizer
from typing import Tuple


class BayesianOptimizationDoubleParam(Algorithm):
    def __init__(self, T, batch_size, arm_interval,
                 warmup: int = 4,
                 acq_func: str = "LCB"):
        super().__init__(T, batch_size, arm_interval)
        self.warmup = warmup + 1
        self.acq_func = acq_func
        self.opt = None

    def initialize(self):
        self.opt = Optimizer(dimensions=self.arm_intervals,
                             n_initial_points=self.warmup,
                             acq_func=self.acq_func)

    def get_arm_value(self) -> Tuple:
        current_arm = self.opt.ask()
        assert len(current_arm) == 2
        return current_arm

    def get_arms_batch(self) -> list:
        assert self.warmup > self.batch_size
        batch = self.opt.ask(n_points=self.batch_size)
        return batch

    def learn(self, action: Tuple, timestep: int, reward: float):
        assert len(action) == 2
        self.opt.tell(action, -reward)

    def batch_learn(self, actions: list, timesteps: list, rewards: list):
        regrets = [-rew for rew in rewards]
        assert len(actions[0]) == 2
        self.opt.tell(actions, regrets)
