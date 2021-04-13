from multiprocessing import Pool
from pprint import pprint

import pandas as pd

import matplotlib.pyplot as plt
from scipy.stats import pareto
from algorithms import *
from tqdm import tqdm


class Testbed:

    # configure parameters of experiments
    def __init__(self,
                 time_horizon: int = 20000,
                 trials: int = 40,
                 delta: float = 0.1,
                 alpha: float = 3.1,
                 epsilon: int = 1,
                 action_cost: int = 0,
                 c_zooming: float = 0.01,  # searched within {1, 0.1, 0.01} and `0.01` is the best choice 0.0009
                 c_adtm: float = 0.1,  # searched within {1, 0.1, 0.01} and `0.1` is the best choice 0.009
                 c_admm: float = 0.1,  # searched within {1, 0.1, 0.01} and `0.1` is the best choice 0.1
                 warmup_days_bandits: int = 4,  # it is not advised to try values > 4
                 warmup_days_bayesian: int = 4,
                 search_interval: tuple = (1, 3),  # (0.33, 1)
                 stochasticity: bool = True,
                 heavy_tails: bool = False,
                 noise_modulation: float = .3,
                 reward_type: str = "triangular",
                 img_filepath: str = None,
                 is_sequential_learning: bool = True,
                 batch_size: int = 4,
                 verbosity: int = 1
                 ):
        """

        Args:
            time_horizon: int: a time horizon of a simulated experiment
            trials: int: number of independent trials
            delta: float:
            alpha: float: property of pareto distribution used in heavy tailed reward simulation
            epsilon: int:
            action_cost: int: a cost associated with any action on every step, regardless of the result
            c_zooming: float:
            c_adtm: float:
            c_admm: float:
            warmup_days_bandits: int:
            warmup_days_bayesian: int:
            search_interval: tuple:
            stochasticity: bool:
            heavy_tails: bool:
            noise_modulation: float:
            reward_type: str: {"triangular", "quadratic", "metric-based"}
            img_filepath: str:
            is_sequential_learning: bool:
            batch_size: int
            verbosity: int
        """

        self.time_horizon = time_horizon
        self.trials = trials
        self.action_cost = action_cost
        self.alpha = alpha
        self.search_interval = search_interval
        self.stochasticity = stochasticity
        self.heavy_tails = heavy_tails
        self.noise_modulation = noise_modulation
        self.reward_type = reward_type
        self.img_fpath = img_filepath
        self.is_sequential_learning = is_sequential_learning
        self.batch_size = batch_size
        self.verbosity = verbosity

        # compute upper bounds for moments of different orders
        a_hat = max(abs(-action_cost), abs(-action_cost - 0.4))
        sigma_second = max(alpha / ((alpha - 1) ** 2 * (alpha - 2)), 1 / (36 * np.sqrt(2)))
        nu_second = max(a_hat ** 2 + sigma_second, np.power(12 * np.sqrt(2), -(1 + epsilon)))
        nu_third = a_hat ** 3 + 2 * alpha * (alpha + 1) / (
                (alpha - 1) ** 3 * (alpha - 2) * (alpha - 3)) + 3 * a_hat * sigma_second
        warmup_steps_bandits = warmup_days_bandits if is_sequential_learning else warmup_days_bandits * batch_size
        warmup_steps_bayesian = warmup_days_bayesian if is_sequential_learning else warmup_days_bayesian * batch_size

        self.algorithms = [
            # Zooming(time_horizon, self.batch_size, search_interval, delta, c_zooming, nu_third),
            # ADTM(time_horizon, self.batch_size, search_interval, delta, c_adtm, nu_second, epsilon),
            # ADMM(time_horizon, self.batch_size, search_interval, delta, c_admm, sigma_second, epsilon),
            Random(time_horizon, self.batch_size, search_interval),
            Optimal(time_horizon, self.batch_size, search_interval, self.reward_type),
            EpsilonGreedy(time_horizon, self.batch_size, search_interval, warmup=warmup_steps_bandits),
            BayesianOptimization(time_horizon, self.batch_size, search_interval, warmup=warmup_steps_bayesian),
            # ThompsonSampling(time_horizon, self.batch_size, search_interval),
            UCB(time_horizon, self.batch_size, search_interval)
        ]
        self.cum_reward = None

    def main(self):
        self.cum_reward = self.simulate()

    def simulate(self):
        """
        Simulates several algorithm's performance on search for optimal arm.
        There is a separate function calculating regret and rewards.
        Regret is calculated based on knowledge of the shape of regret as a function of optimal parameter.
        Reward is calculated based on that shape as well as stochastic heavy-tailed component and fixed cost associated
        with pulling an arm.

        In a real-life setting we don't know regret and cannot compare two algorithms based on their regrets.
        We can only compare them based on rewards. Therefore a refactoring was applied to reflect performance
        in terms of cumulative reward (the more the merrier)

        Returns:
            avg_cum_regret:

        """
        cum_reward = np.zeros((len(self.algorithms), self.time_horizon + 1))

        if self.verbosity > 0:
            trial_range = tqdm(range(self.trials))
        else:
            trial_range = range(self.trials)

        for trial in trial_range:
            inst_reward = np.zeros((len(self.algorithms), self.time_horizon + 1))
            for alg in self.algorithms:
                alg.initialize()
            if self.is_sequential_learning:
                self.sequential_learning(inst_reward)
            else:
                self.batch_learning(inst_reward)
            cum_reward += np.cumsum(inst_reward, axis=-1)
        avg_cum_regret = cum_reward / self.trials
        return avg_cum_regret

    @staticmethod
    def chunks(lst: list, n: int):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def batch_learning(self, inst_reward):
        batches = self.chunks(list(range(1, self.time_horizon + 1)), self.batch_size)
        if self.verbosity > 1:
            batches = tqdm(batches)
        for batch in batches:
            for algo_index, alg in enumerate(self.algorithms):
                arms = alg.get_arms_batch()
                rewards = [self._get_reward(arm) for arm in arms]
                rewards = [self.augment_reward(rew,
                                               self.stochasticity,
                                               self.alpha,
                                               self.action_cost,
                                               self.heavy_tails,
                                               self.noise_modulation) for rew in rewards]
                for ind, timestep in enumerate(batch):
                    inst_reward[algo_index, timestep] = rewards[ind]
                alg.batch_learn(actions=arms, timesteps=batch, rewards=rewards)

    def sequential_learning(self, inst_reward):
        if self.verbosity > 1:
            timestep_range = tqdm(range(1, self.time_horizon + 1))
        else:
            timestep_range = range(1, self.time_horizon + 1)
        for timestep in timestep_range:
            for algo_index, alg in enumerate(self.algorithms):
                arm = alg.get_arm_value()
                reward = self._get_reward(arm)
                # reward consists of constant factor less regret plus stochastic reward factor from pareto distribution
                reward = self.augment_reward(reward,
                                             self.stochasticity,
                                             self.alpha,
                                             self.action_cost,
                                             self.heavy_tails,
                                             self.noise_modulation
                                             )
                inst_reward[algo_index, timestep] = reward
                alg.learn(arm, timestep, reward)  # algorithm is observing the reward and changing priors

    @staticmethod
    def augment_reward(reward: float,
                       stochasticity: bool,
                       alpha: float,
                       action_cost: int,
                       heavy_tails: bool = True,
                       noise_modulation: float = .3
                       ) -> float:
        """
        A method to augment reward
        Args:
            reward: float
            stochasticity: bool:
            alpha: float:
            action_cost: int:
            heavy_tails: bool:
            noise_modulation: float:

        Returns:
            reward: float

        """
        if stochasticity:
            if heavy_tails:
                stochastic_factor = pareto.rvs(alpha) - alpha / (alpha - 1)
                reward += stochastic_factor
            else:
                stochastic_factor = np.random.uniform(1 - noise_modulation, 1 + noise_modulation)
                reward *= stochastic_factor
        reward -= action_cost
        return reward

    def _get_reward(self, arm: float):
        """
        A method that routes reward calculation
        Args:
            arm: float:

        Returns:

        """
        if self.reward_type == "triangular":
            return self.triangular_reward(arm)
        elif self.reward_type == "quadratic":
            return self.quadratic_reward(arm)
        elif self.reward_type == "metric-based":
            return self.metric_based_reward(arm)
        else:
            raise NotImplementedError(f"'{self.reward_type.capitalize()}' reward type is not implemented")

    @staticmethod
    def quadratic_reward(arm: float) -> float:
        return max(0.1, 0.9 - 3.2 * (0.7 - arm) ** 2)

    @staticmethod
    def triangular_reward(arm: float) -> float:
        # custom regret function, triangular regret is selected
        instant_regret = min(abs(np.subtract(arm, 0.4)), abs(np.subtract(arm, 0.8)))
        instant_reward = np.negative(instant_regret)
        return instant_reward

    def metric_based_reward(self, arm: float):
        # TODO: implement
        pass

    def _arm_converter(self, arm: float) -> float:
        # TODO: finish method
        """
        A scaling method for arm conversion. Used to convert arms from different intervals to [0,1]

        Args:
            arm: float:

        Returns:
            arm: float:

        """
        if self.search_interval[0] != 0:
            starting_difference = self.search_interval[0]
            scaling_factor = self.search_interval[1] - self.search_interval[0]
        return arm

    @staticmethod
    def harmonic_mean(a: float, b: float, c: float):
        return 3 / (1 / a + 1 / b + 1 / c)

    def plot(self):
        """
        Plots the results and saves the image

        Returns:

        """
        plt.figure(figsize=(7, 4))
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        names = [f'{alg.__class__.__name__}' for alg in self.algorithms]
        linestyles = ['-', '--', '-.', ':', "-", "--", "-.", ":", '-']
        for result, name, linestyle in zip(self.cum_reward, names, linestyles):
            plt.plot(result, label=name, linewidth=2.0, linestyle=linestyle)
        plt.legend(loc='upper left', frameon=True, fontsize=10)
        plt.xlabel('t', labelpad=1, fontsize=15)
        plt.ylabel('cumulative reward', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid()
        print("Saving figure...")
        fname = self.img_fpath
        print(fname)
        plt.savefig(fname,
                    dpi=500,
                    bbox_inches='tight'
                    )
        plt.close()
        print("Saved!")


def _run_experiment(params: dict):
    """
    Runs an experiment, plots and saves the results.
    Args:
        params:

    Returns:

    """
    print(params)
    tb = Testbed(**params)
    tb.main()
    tb.plot()


def _add_fpath(params: dict):
    """
    Function adds filepath for image storing to params dictionary
    Args:
        params: dict:

    Returns:

    """
    stochasticity = "stochasticity_on" if params['stochasticity'] else "stochasticity_off"
    heavy_tails = "heavy_tails" if params['heavy_tails'] else f"{params['noise_modulation'] * 100}%noise"
    heavy_tails = "_" + heavy_tails if params['stochasticity'] else ""
    fname = f'resources/cum_reward_{params["search_interval"]}_{params["trials"]}_{-params["action_cost"]}_{params["time_horizon"]}_{params["reward_type"]}_{stochasticity}{heavy_tails}.png'
    params['img_filepath'] = fname


def run_all():
    """
    Main method that runs multiple simulated elements, determined by json lists of arguments,
    saves parameters of each visualization along with the visualization path and stores visualizations.

    Returns:

    """
    exp_template = {'action_cost': 0,
                    'stochasticity': True,
                    'time_horizon': 60,
                    'c_zooming': .0009,
                    'c_admm': .009,
                    'c_adtm': .009,
                    'warmup_days_bayesian': 1,
                    'warmup_days_bandits': 1,
                    'reward_type': "triangular",
                    'heavy_tails': False,
                    'noise_modulation': .25,
                    'trials': 100,
                    'num_cores': 2,
                    'search_interval': (-10., 10.),
                    'is_sequential_learning': False,
                    'batch_size': 4,
                    'verbosity': 1
                    }

    num_cores = exp_template.pop('num_cores')

    experiments = list()
    for search_interval in [(-10., 10.), (1., 3.), (0.3, 1.), (0., 1.)]:
        exp = exp_template.copy()
        exp['search_interval'] = search_interval
        assert exp['search_interval'] == search_interval
        for rew_type in {"quadratic", "triangular"}:
            exp = exp.copy()
            exp['reward_type'] = rew_type
            if rew_type == 'triangular':
                exp['c_admm'] = .009
                exp['c_adtm'] = .009
                exp['c_zooming'] = .0009
            else:
                exp['c_zooming'] = .003
                exp['c_admm'] = .09
                exp['c_adtm'] = .06
            for stochasticity in {True, False}:
                exp = exp.copy()
                exp['stochasticity'] = stochasticity
                if stochasticity:
                    for is_heavy_tail in {True, False}:
                        exp = exp.copy()
                        exp['heavy_tails'] = is_heavy_tail
                        if not is_heavy_tail:
                            for noise_modulation in {.5, .25}:
                                exp = exp.copy()
                                exp['noise_modulation'] = noise_modulation
                                _add_fpath(exp)
                                experiments.append(exp)
                        else:
                            if exp['reward_type'] == 'quadratic':
                                exp['c_admm'] = .03
                                exp['c_adtm'] = .1
                            _add_fpath(exp)
                            experiments.append(exp)
                else:
                    _add_fpath(exp)
                    experiments.append(exp)

    df = pd.DataFrame(experiments)
    df.to_csv("resources/experiments.csv", index=False)

    with Pool(processes=num_cores) as p:
        p.map(_run_experiment, experiments)


def run_one():
    """
    Runs a single experiment according to experiment parameters present in experiment dictionary.
    Stores an image with experiments' results into `resources` folder.

    Returns:
    """
    experiment = {
        'action_cost': 0,
        'stochasticity': False,
        'time_horizon': 60,
        'c_zooming': .0009,
        'c_admm': .009,
        'c_adtm': .009,
        'warmup_days_bandits': 1,
        'warmup_days_bayesian': 1,
        'reward_type': "quadratic",
        'heavy_tails': True,
        'noise_modulation': .25,
        'trials': 1,
        'search_interval': (0, 1.),
        'is_sequential_learning': False,
        'batch_size': 4,
        'verbosity': 2
    }
    _add_fpath(experiment)
    _run_experiment(experiment)


if __name__ == '__main__':
    # run_all()
    run_one()
