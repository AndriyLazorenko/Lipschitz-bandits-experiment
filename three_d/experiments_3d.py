from multiprocessing import Pool

import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.extend([os.path.join(dir_path, "..")])

from utils.experiments import add_fpath, run_experiment
from utils.paths import experiments_3d_dataframe, experiments_3d_configs
import pandas as pd
import json


def run_all():
    """
    Main method that runs multiple simulated elements, determined by json lists of arguments,
    saves parameters of each visualization along with the visualization path and stores visualizations.

    Returns:

    """
    with open(experiments_3d_configs) as json_file:
        exp_template = json.load(json_file)

    num_cores = exp_template.pop('num_cores')

    experiments = list()
    for rew_type in {"bukin", "rosenbrock"}:
        exp = exp_template.copy()
        exp['reward_type'] = rew_type
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
                        add_fpath(exp)
                        experiments.append(exp)
            else:
                add_fpath(exp)
                experiments.append(exp)

    df = pd.DataFrame(experiments)
    df.to_csv(experiments_3d_dataframe, index=False)

    with Pool(processes=num_cores) as p:
        p.map(run_experiment, experiments)


if __name__ == '__main__':
    run_all()
