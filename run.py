import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pareto
from algorithms import Zooming, ADTM, ADMM
from tqdm import tqdm


def simulate(algorithms: list,
             action_reward: int,
             alpha: float,
             time_horizon: int,
             trials: int,
             verbosity: int = 1):
    """
    Simulates several algorithm's performance on search for optimal arm.
    There is a separate function calculating regret and rewards.
    Regret is calculated based on knowledge of the shape of regret as a function of optimal parameter.
    Reward is calculated based on that shape as well as stochastic heavy-tailed component and fixed cost associated
    with pulling an arm.

    In a real-life setting we don't know regret and cannot compare two algorithms based on their regrets.
    We can only compare them based on rewards. Therefore a refactoring was applied to reflect performance
    in terms of cumulative reward (the more the merrier)

    Args:
        algorithms: list: algorithms being part of simulated experiment
        action_reward: int: a reward added to reward function on every step, regardless of the result
        alpha: float: property of pareto distribution used in heavy tailed reward simulation
        time_horizon: int: a time horizon of a simulated experiment
        trials: number of independent trials
        verbosity: int: how verbose is the algorithm

    Returns:
        avg_cum_regret:

    """
    # cum_regret = np.zeros((len(algorithms), time_horizon + 1))
    cum_reward = np.zeros((len(algorithms), time_horizon + 1))

    if verbosity > 0:
        trial_range = tqdm(range(trials))
    else:
        trial_range = range(trials)

    for trial in trial_range:
        # inst_regret = np.zeros((len(algorithms), time_horizon + 1))
        inst_reward = np.zeros((len(algorithms), time_horizon + 1))
        for alg in algorithms:
            alg.initialize()

        if verbosity > 1:
            timestep_range = tqdm(range(1, time_horizon + 1))
        else:
            timestep_range = range(1, time_horizon + 1)

        for timestep in timestep_range:
            for algo_index, alg in enumerate(algorithms):
                idx = alg.output()  # index of the arm is computed
                arm = alg.active_arms[idx]
                # custom regret function, triangular regret is selected
                instant_regret = min(abs(arm - 0.4), abs(arm - 0.8))
                instant_reward = - instant_regret
                # inst_regret[algo_index, timestep] = instant_regret
                reward = get_reward(action_reward, alpha, instant_reward, "paper")
                inst_reward[algo_index, timestep] = reward
                # reward consists of constant factor less regret plus stochastic reward factor from pareto distribution
                alg.observe(timestep, reward)  # algorithm is observing the reward and changing priors

        cum_reward += np.cumsum(inst_reward, axis=-1)
        # cum_regret += np.cumsum(inst_regret, axis=-1)
    # regret doesn't correspond to reward. Regret is used in visualizations, while reward contains additional factors
    avg_cum_regret = cum_reward / trials
    # avg_cum_regret = cum_regret / trials
    return avg_cum_regret


def get_reward(action_reward: int, alpha: float, instant_reward: float, reward_type: str = "paper"):
    if reward_type == "paper":
        return _paper_reward(action_reward, alpha, instant_reward)
    else:
        raise NotImplementedError(f"'{reward_type.capitalize()}' reward type is not implemented")


def _paper_reward(action_reward: int, alpha: float, instant_reward: float):
    return action_reward + instant_reward + pareto.rvs(alpha) - alpha / (alpha - 1)


def run_experiment(a):
    # configure parameters of experiments
    time_horizon = 20000
    trials = 40
    delta = 0.1
    alpha = 3.1
    epsilon = 1

    # compute upper bounds for moments of different orders
    a_hat = max(abs(a), abs(a - 0.4))
    sigma_second = max(alpha / ((alpha - 1) ** 2 * (alpha - 2)), 1 / (36 * np.sqrt(2)))
    nu_second = max(a_hat ** 2 + sigma_second, np.power(12 * np.sqrt(2), -(1 + epsilon)))
    nu_third = a_hat ** 3 + 2 * alpha * (alpha + 1) / (
            (alpha - 1) ** 3 * (alpha - 2) * (alpha - 3)) + 3 * a_hat * sigma_second

    # simulate
    c_zooming = 0.01  # searched within {1, 0.1, 0.01} and `0.01` is the best choice
    c_ADTM = 0.1  # searched within {1, 0.1, 0.01} and `0.1` is the best choice
    c_ADMM = 0.1  # searched within {1, 0.1, 0.01} and `0.1` is the best choice
    algorithms = [Zooming(delta, time_horizon, c_zooming, nu_third), ADTM(delta, time_horizon, c_ADTM, nu_second, epsilon),
                  ADMM(delta, time_horizon, c_ADMM, sigma_second, epsilon)]
    cum_reward = simulate(algorithms, a, alpha, time_horizon, trials)

    # plot figure
    plt.figure(figsize=(7, 4))
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    names = [f'{alg.__class__.__name__}' for alg in algorithms]
    linestyles = ['-', '--', '-.']
    for result, name, linestyle in zip(cum_reward, names, linestyles):
        plt.plot(result, label=name, linewidth=2.0, linestyle=linestyle)
    plt.legend(loc='upper left', frameon=True, fontsize=10)
    plt.xlabel('t', labelpad=1, fontsize=15)
    plt.ylabel('cumulative regret', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(f'cum_reward_{a}.png', dpi=500, bbox_inches='tight')


if __name__ == '__main__':
    run_experiment(a=0)
    # run_experiment(a=-2)
