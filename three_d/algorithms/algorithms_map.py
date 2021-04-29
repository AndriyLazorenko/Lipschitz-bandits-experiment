from three_d.algorithms.random_algorithm import Random
from three_d.algorithms.optimal_algorithm import Optimal
from three_d.algorithms.bayesian_optimization import BayesianOptimization
from three_d.algorithms.article_algorithm import Article

import json
from utils.paths import algorithms_3d_config


def get_algorithms(time_horizon: int,
                   batch_size: int,
                   search_interval: tuple,
                   reward_type: str,
                   warmup_bayesian: int
                   ) -> list:

    with open(algorithms_3d_config) as json_file:
        algorithms = json.load(json_file)

    algo_map = {
        "random": Random(time_horizon, batch_size, search_interval),
        "optimal": Optimal(time_horizon, batch_size, search_interval, reward_type),
        "bayesian_optimization": BayesianOptimization(time_horizon, batch_size, search_interval,
                                                      warmup=warmup_bayesian),
        "article": Article(time_horizon, batch_size, search_interval)
    }
    for_ret = list()
    for algo in algorithms:
        try:
            for_ret.append(algo_map[algo])
        except KeyError as err:
            raise NotImplementedError(f"An algorithm {algo} is not implemented!")
    return for_ret
