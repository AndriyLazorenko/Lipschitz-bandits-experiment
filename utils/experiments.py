from three_d.testbed_3d import Testbed3D
from two_d.testbed_2d import Testbed2D
from pprint import pprint
import json

from utils.paths import resources_3d_folder, resources_2d_folder, experiment_3d_config, experiment_2d_config


def run_experiment(params: dict):
    """
    Runs an experiment, plots and saves the results.
    Args:
        params:

    Returns:

    """
    if params["dimension"] == 2:
        del params["dimension"]
        tb = Testbed2D(**params)
    else:
        del params["dimension"]
        tb = Testbed3D(**params)
    pprint(params)
    tb.simulate()
    tb.plot()


def add_fpath(params: dict):
    """
    Function adds filepath for image storing to params dictionary
    Args:
        params: dict:

    Returns:

    """
    stochasticity = "stochasticity_on" if params['stochasticity'] else "stochasticity_off"
    heavy_tails = "heavy_tails" if params['heavy_tails'] else f"{params['noise_modulation'] * 100}%noise"
    heavy_tails = "_" + heavy_tails if params['stochasticity'] else ""
    if params['dimension'] == 3:
        fname = f'{resources_3d_folder}cum_reward_{params["search_interval"]}_{params["trials"]}_{-params["action_cost"]}_{params["time_horizon"]}_{params["reward_type"]}_{stochasticity}{heavy_tails}.png'
    else:
        fname = f'{resources_2d_folder}cum_reward_{params["search_interval"]}_{params["trials"]}_{-params["action_cost"]}_{params["time_horizon"]}_{params["reward_type"]}_{stochasticity}{heavy_tails}.png'
    params['img_filepath'] = fname


def run_one(config_type: str = "3d"):
    """
    Runs a single experiment according to experiment parameters present in experiment dictionary.
    Stores an image with experiments' results into `resources` folder.

    Args:
        config_type: str: {"2d", "3d"}


    """
    if config_type == "3d":
        config_filepath = experiment_3d_config
    else:
        config_filepath = experiment_2d_config

    with open(config_filepath) as json_file:
        experiment = json.load(json_file)
    add_fpath(experiment)
    run_experiment(experiment)


if __name__ == '__main__':
    run_one()
