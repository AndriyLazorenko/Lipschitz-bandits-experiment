from multiprocessing import Pool

from paths import experiment_2d_config, experiments_2d_dataframe, experiments_2d_configs, resources_2d_folder
from testbed_2d import Testbed2D
import pandas as pd
import json
from pprint import pprint


def _run_experiment(params: dict):
    """
    Runs an experiment, plots and saves the results.
    Args:
        params:

    Returns:

    """
    tb = Testbed2D(**params)
    tb.simulate()
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
    fname = f'{resources_2d_folder}cum_reward_{params["search_interval"]}_{params["trials"]}_{-params["action_cost"]}_{params["time_horizon"]}_{params["reward_type"]}_{stochasticity}{heavy_tails}.png'
    params['img_filepath'] = fname


def run_all():
    """
    Main method that runs multiple simulated elements, determined by json lists of arguments,
    saves parameters of each visualization along with the visualization path and stores visualizations.

    Returns:

    """
    with open(experiments_2d_configs) as json_file:
        exp_template = json.load(json_file)

    num_cores = exp_template.pop('num_cores')

    experiments = list()
    for rew_type in {"quadratic", "triangular", "article"}:
        exp = exp_template.copy()
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
            if rew_type == 'article' and stochasticity:
                continue
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
    df.to_csv(experiments_2d_dataframe, index=False)

    with Pool(processes=num_cores) as p:
        p.map(_run_experiment, experiments)


def run_one():
    """
    Runs a single experiment according to experiment parameters present in experiment dictionary.
    Stores an image with experiments' results into `resources` folder.

    Returns:
    """
    with open(experiment_2d_config) as json_file:
        experiment = json.load(json_file)
    _add_fpath(experiment)
    _run_experiment(experiment)


if __name__ == '__main__':
    # run_all()
    run_one()
