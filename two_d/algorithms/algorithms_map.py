from two_d.algorithms.random_algorithm import RandomSingleParam
from two_d.algorithms.optimal_algorithm import Optimal
from two_d.algorithms.bayesian_optimization import BayesianOptimizationSingleParam
from two_d.algorithms.bandit_algorithms import UCB, ThompsonSampling, EpsilonGreedy
from two_d.algorithms.zooming_algorithms import ADMM, ADTM, Zooming
from two_d.algorithms.article_algorithm import LBS

import json
from utils.paths import algorithms_2d_config


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
        "random": RandomSingleParam(time_horizon, batch_size, search_interval),
        "optimal": Optimal(time_horizon, batch_size, search_interval, reward_type),
        "epsilon_greedy": EpsilonGreedy(time_horizon, batch_size, search_interval, warmup=warmup_bandits),
        "bayesian_optimization": BayesianOptimizationSingleParam(time_horizon, batch_size, search_interval,
                                                                 warmup=warmup_bayesian),
        "thompson_sampling": ThompsonSampling(time_horizon, batch_size, search_interval),
        "ucb": UCB(time_horizon, batch_size, search_interval),
        "LBS": LBS(time_horizon, batch_size, search_interval)
    }
    for_ret = list()
    for algo in algorithms:
        if reward_type == "article":
            if algo == "thompson_sampling":
                print("Thompson sampling is bugged and doesn't work with article reward")
            else:
                append_algo(algo, algo_map, for_ret)
        elif reward_type != "article":
            if algo == "LBS":
                print("Linear bid shading is only defined for reward from article")
            else:
                append_algo(algo, algo_map, for_ret)
    return for_ret


def append_algo(algo: str, algo_map: dict, for_ret: list):
    try:
        for_ret.append(algo_map[algo])
    except KeyError as err:
        raise NotImplementedError(f"An algorithm {algo} is not implemented!")
