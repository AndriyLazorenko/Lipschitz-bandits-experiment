from algorithms.random_algorithm import Random
from algorithms.optimal_algorithm import Optimal
from algorithms.bayesian_optimization import BayesianOptimization
from algorithms.bandit_algorithms import UCB, ThompsonSampling, EpsilonGreedy
from algorithms.zooming_algorithms import ADMM, ADTM, Zooming
from algorithms.article_algorithm import Article

import json
from paths import algorithms_2d_config


def get_algorithms(time_horizon: int,
                   batch_size: int,
                   search_interval: tuple,
                   delta: float,
                   c_zooming: float,
                   nu_third: float,
                   c_adtm: float,
                   nu_second: float,
                   epsilon: float,
                   c_admm: float,
                   sigma_second: float,
                   reward_type: str,
                   warmup_bandits: int,
                   warmup_bayesian: int
                   ) -> list:

    with open(algorithms_2d_config) as json_file:
        algorithms = json.load(json_file)

    algo_map = {
        "zooming": Zooming(time_horizon, batch_size, search_interval, delta, c_zooming, nu_third),
        "adtm": ADTM(time_horizon, batch_size, search_interval, delta, c_adtm, nu_second, epsilon),
        "admm": ADMM(time_horizon, batch_size, search_interval, delta, c_admm, sigma_second, epsilon),
        "random": Random(time_horizon, batch_size, search_interval),
        "optimal": Optimal(time_horizon, batch_size, search_interval, reward_type),
        "epsilon_greedy": EpsilonGreedy(time_horizon, batch_size, search_interval, warmup=warmup_bandits),
        "bayesian_optimization": BayesianOptimization(time_horizon, batch_size, search_interval,
                                                      warmup=warmup_bayesian),
        "thompson_sampling": ThompsonSampling(time_horizon, batch_size, search_interval),
        "ucb": UCB(time_horizon, batch_size, search_interval),
        "article": Article(time_horizon, batch_size, search_interval)
    }
    for_ret = list()
    for algo in algorithms:
        try:
            for_ret.append(algo_map[algo])
        except KeyError as err:
            raise NotImplementedError(f"An algorithm {algo} is not implemented!")
    return for_ret
