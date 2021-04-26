from _ast import Tuple

from scipy.stats import pareto, chi2, gamma
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import os

from utils.paths import scenario


class ScenarioGenerator:
    def __init__(self,
                 gamma_vt: float = 1.0,
                 gamma_mt: float = 2.0,
                 num_impressions_won_daily: int = 10000,
                 num_days: int = 60):
        self.num_days = num_days
        self.gamma_vt = gamma_vt
        self.gamma_mt = gamma_mt
        self.num_impressions_won_daily = num_impressions_won_daily
        self.df = None

    def generate_scenario(self) -> Tuple(float, float):
        vt = gamma.rvs(self.gamma_vt, scale=1 / 2)
        mt = gamma.rvs(self.gamma_mt, scale=1 / 2)
        return vt, mt

    def generate_a_day(self, day_index: int = 0) -> list:
        pairs = list()
        for i in range(self.num_impressions_won_daily):
            scenario = self.generate_scenario()
            pairs.append((scenario[0], scenario[1], day_index))
        return pairs

    def generate_n_days(self):
        days_range = tqdm(range(self.num_days))
        for i in days_range:
            pairs = self.generate_a_day(i + 1)
            if i == 0:
                self.df = pd.DataFrame(pairs, columns=['vt', 'mt', 'day'])
            else:
                df_2 = pd.DataFrame(pairs, columns=['vt', 'mt', 'day'])
                self.df = self.df.append(df_2)

    def scale_prices(self):
        mi = min([self.df.vt.min(), self.df.mt.min()])
        ma = max([self.df.vt.max(), self.df.mt.max()])
        self.df.vt = (self.df.vt - mi) / (ma - mi)
        self.df.mt = (self.df.mt - mi) / (ma - mi)

    def persist(self):
        self.df.to_csv(scenario, index=False)

    def load(self):
        self.df = pd.read_csv(scenario)

    def generate_scale_persist(self, is_regenerate: bool = False):
        if is_regenerate or not os.path.isfile(scenario):
            self.generate_n_days()
            self.scale_prices()
            self.persist()

    @staticmethod
    def visualize():
        df = pd.read_csv(scenario)
        df.vt.plot.hist(bins=100, alpha=0.5)
        df.mt.plot.hist(bins=100, alpha=0.5)
        plt.show()


if __name__ == '__main__':
    sg = ScenarioGenerator()
    # sg.generate_n_days()
    # sg.scale_prices()
    # sg.persist()
    sg.visualize()
