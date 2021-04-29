import numpy as np
from rewards import augment_reward
from two_d.rewards_2d import Rewards2D
from three_d.rewards_3d import Rewards3D
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

from utils.paths import reward_shapes_folder, reward_plots_info


class RewardVisualizer:
    def __init__(self,
                 stochasticity: bool = False,
                 heavy_tails: bool = True,
                 noise_modulation: float = .50,
                 reward_type: str = "triangular",
                 dimension: int = 2
                 ):
        self.stochasticity = stochasticity
        self.heavy_tails = heavy_tails
        self.noise_modulation = noise_modulation
        self.reward_type = reward_type
        self.dimension = dimension
        self.alpha = 3.1
        self.action_cost = 0

    def compose_filename(self) -> str:
        stochasticity = "stochasticity_on" if self.stochasticity else "stochasticity_off"
        heavy_tails = "heavy_tails" if self.heavy_tails else f"{self.noise_modulation * 100}%noise"
        heavy_tails = "_" + heavy_tails if self.stochasticity else ""
        fname = f'{reward_shapes_folder}{self.dimension}d_{self.reward_type}_{stochasticity}{heavy_tails}.png'
        return fname

    def get_reward(self, arm) -> float:
        rew = None
        if self.reward_type == "triangular" and self.dimension == 2:
            rew = Rewards2D.triangular_reward(arm)
        elif self.reward_type == "quadratic" and self.dimension == 2:
            rew = Rewards2D.quadratic_reward(arm)
        elif self.reward_type == "triangular" and self.dimension == 3:
            rew = Rewards3D.triangular_reward(arm)
        elif self.reward_type == "quadratic" and self.dimension == 3:
            rew = Rewards3D.quadratic_reward(arm)
        return rew

    def get_augmented_reward(self, arm: list):
        if self.dimension == 2:
            arm = arm[0]
        rew = self.get_reward(arm)
        rew = augment_reward(rew,
                             stochasticity=self.stochasticity,
                             alpha=self.alpha,
                             action_cost=self.action_cost,
                             heavy_tails=self.heavy_tails,
                             noise_modulation=self.noise_modulation)
        return rew

    def plot_reward(self):
        if self.dimension == 2:
            x = np.linspace(0, 1, 400).reshape(-1, 1)
            fx = [self.get_augmented_reward(x_i) for x_i in x]
            plt.plot(x, fx, "r--", label="reward")

        else:
            x = np.linspace(-1, 1, 400)
            y = np.linspace(-5, 10, 400)
            x, y = np.meshgrid(x, y)

            def f(a, b):
                return self.get_augmented_reward([a, b])

            z = f(x, y)
            ax = plt.axes(projection='3d')
            ax.plot_surface(x, y, z, rstride=1, cstride=1,
                            cmap=cm.coolwarm, edgecolor='none', antialiased=False)
            # ax.plot_trisurf(x, y, z, cmap='viridis', linewidth=.2, antialiased=True)  # 'viridis' # plt.cm.CMRmap
        plt.grid()
        fname = self.compose_filename()
        plt.savefig(fname,
                    dpi=500,
                    bbox_inches='tight')
        plt.close()
        print("Saved!")


def plot_one(params: dict):
    r = RewardVisualizer(**params)
    r.plot_reward()


def plot_all():
    default_params = {
        "dimension": 2,
        "reward_type": "quadratic",
        "stochasticity": False,
        "heavy_tails": True,
        "noise_modulation": .25
    }
    plots = list()
    for dimension in {2, 3}:
        default_params = default_params.copy()
        default_params["dimension"] = dimension
        for reward_type in {"triangular", "quadratic"}:
            default_params = default_params.copy()
            default_params["reward_type"] = reward_type
            for stochasticity in {True, False}:
                default_params = default_params.copy()
                default_params["stochasticity"] = stochasticity
                if stochasticity:
                    for heavy_tails in {True, False}:
                        default_params = default_params.copy()
                        default_params["heavy_tails"] = heavy_tails
                        if not heavy_tails:
                            for noise_modulation in {.25, .50}:
                                default_params = default_params.copy()
                                default_params["noise_modulation"] = noise_modulation
                                plots.append(default_params)
                        else:
                            plots.append(default_params)
                else:
                    plots.append(default_params)
    df = pd.DataFrame(plots)
    df.to_csv(reward_plots_info, index=False)
    for plot in plots:
        plot_one(plot)


if __name__ == '__main__':
    # plot_one()
    plot_all()
