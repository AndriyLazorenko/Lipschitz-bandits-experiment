from three_d.algorithms.random_algorithm import RandomDoubleParam
from two_d.algorithms.bandit_algorithms import UCB, EpsilonGreedy
from two_d.algorithms.random_algorithm import RandomSingleParam
from three_d.algorithms.optimal_algorithm import Optimal
from three_d.algorithms.bayesian_optimization import BayesianOptimizationDoubleParam
from two_d.algorithms.bayesian_optimization import BayesianOptimizationSingleParam
from three_d.algorithms.article_algorithm import NLBS
from two_d.algorithms.article_algorithm import LBS

import json
from utils.paths import algorithms_3d_config


def get_algorithms(time_horizon: int,
                   batch_size: int,
                   search_interval: tuple,
                   search_interval_2d: tuple,
                   reward_type: str,
                   warmup_bayesian: int
                   ) -> list:
    with open(algorithms_3d_config) as json_file:
        algorithms = json.load(json_file)

    algo_map = {
        "optimal": Optimal(time_horizon, batch_size, search_interval, reward_type),
        "random_NLBS": RandomDoubleParam(time_horizon, batch_size, search_interval),
        "random_LBS": RandomSingleParam(time_horizon, batch_size, search_interval_2d),
        "bayesian_optimization_NLBS": BayesianOptimizationDoubleParam(time_horizon, batch_size, search_interval,
                                                                      warmup=warmup_bayesian),
        "bayesian_optimization_LBS": BayesianOptimizationSingleParam(time_horizon, batch_size, search_interval_2d,
                                                                     warmup=warmup_bayesian),
        "article_NLBS": NLBS(time_horizon, batch_size, search_interval),
        "article_LBS": LBS(time_horizon, batch_size, search_interval_2d),
        "UCB": UCB(time_horizon, batch_size, search_interval_2d),
        "EpsilonGreedy": EpsilonGreedy(time_horizon, batch_size, search_interval_2d, warmup_bayesian)
    }
    for_ret = list()
    for algo in algorithms:
        try:
            for_ret.append(algo_map[algo])
        except KeyError as err:
            raise NotImplementedError(f"An algorithm {algo} is not implemented!")
    return for_ret
