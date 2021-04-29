import os
from os.path import join

project_path = os.path.dirname(os.path.realpath(__file__))
resources_folder = join(project_path, "../resources/")
reward_shapes_folder = join(resources_folder, "reward_shapes/")
reward_plots_info = join(reward_shapes_folder, "info.csv")
scenario = join(resources_folder, "../resources/campaign_scenario.csv")
## 2d ##
two_d_path = join(project_path, '../two_d')
configs_2d_path = join(two_d_path, "configs")
experiment_2d_config = join(configs_2d_path, "experiment.json")
experiments_2d_configs = join(configs_2d_path, "experiments_template.json")
algorithms_2d_config = join(configs_2d_path, "algorithms.json")
resources_2d_folder = join(resources_folder, "2d/")
experiments_2d_dataframe = join(resources_2d_folder, "experiments.csv")
## 3d ##
three_d_path = join(project_path, '../three_d')
configs_3d_path = join(three_d_path, "configs")
experiment_3d_config = join(configs_3d_path, "experiment.json")
experiments_3d_configs = join(configs_3d_path, "experiments_template.json")
algorithms_3d_config = join(configs_3d_path, "algorithms.json")
resources_3d_folder = join(resources_folder, "3d/")
experiments_3d_dataframe = join(resources_3d_folder, "experiments.csv")
