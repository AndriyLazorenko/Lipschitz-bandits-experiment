from algorithms.abstract_algorithm import Algorithm
from skopt import Optimizer


class BayesianOptimization(Algorithm):
    # TODO: refactor completely
    def __init__(self, T, batch_size, arm_interval,
                 warmup: int = 4,
                 acq_func: str = "LCB"):
        super().__init__(T, batch_size, arm_interval)
        self.warmup = warmup + 1
        self.acq_func = acq_func
        self.opt = None
        self.current_arm_idx = 0

    def initialize(self):
        self.current_arm_idx = 0
        self.active_arms = []
        self.opt = Optimizer(dimensions=[self.arm_interval],
                             n_initial_points=self.warmup,
                             acq_func=self.acq_func)

    def get_arm_value(self) -> float:
        self.active_arms.append(self.opt.ask()[0])
        self.current_arm_idx = len(self.active_arms) - 1
        return self.active_arms[self.current_arm_idx]

    def get_arms_batch(self) -> list:
        assert self.warmup > self.batch_size
        batch = self.opt.ask(n_points=self.batch_size)
        batch = [a[0] for a in batch]
        return batch

    def learn(self, action: float, timestep: int, reward: float):
        self.opt.tell([action], -reward)

    def batch_learn(self, actions: list, timesteps: list, rewards: list):
        regrets = [-rew for rew in rewards]
        actions = [[action] for action in actions]
        self.opt.tell(actions, regrets)
