from multiprocessing import Pool

from utils.experiments import add_fpath, run_experiment
from utils.paths import experiments_2d_dataframe, experiments_2d_configs
import pandas as pd
import json


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
    for rew_type in {"quadratic", "triangular"}:
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
                            add_fpath(exp)
                            experiments.append(exp)
                    else:
                        if exp['reward_type'] == 'quadratic':
                            exp['c_admm'] = .03
                            exp['c_adtm'] = .1
                        add_fpath(exp)
                        experiments.append(exp)
            else:
                add_fpath(exp)
                experiments.append(exp)

    df = pd.DataFrame(experiments)
    df.to_csv(experiments_2d_dataframe, index=False)

    with Pool(processes=num_cores) as p:
        p.map(run_experiment, experiments)


if __name__ == '__main__':
    run_all()
