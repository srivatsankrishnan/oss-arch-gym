import os
import sys

from absl import flags
from absl import app
from absl import logging

os.sys.path.insert(0, os.path.abspath('../../'))
from configs.sims import OpenLane_config 
from arch_gym.envs.envHelpers import helpers
from arch_gym.envs import OpenLane_wrapper
import envlogger
import numpy as np
import pandas as pd

# User-defined flags to run training script with 
flags.DEFINE_string ('rtl_top'                            , 'Top'                       , 'RTL Design for physical implementation.'                                                 )
flags.DEFINE_string ('reward_formulation'                 , 'POWER'                     , 'Metric for optimization.'                                                                )
flags.DEFINE_integer('max_steps'                          , 100                         , 'Number of training steps.'                                                               )
flags.DEFINE_integer('num_episodes'                       , 2                           , 'Number of training episodes.'                                                            ) 
flags.DEFINE_bool   ('use_envlogger'                      , False                       , 'Use envlogger to log the trajectory data.'                                               )  
flags.DEFINE_string ('experiment_summary_data_dir_path'   , '.'                         , 'Path to location where to save data from launched experiment.'                           )
flags.DEFINE_string ('experiment_csv_data_dir_name'       , 'randomwalk_csv_data'       , 'Name of directory to store experiment log data in csv format.'                           )
flags.DEFINE_string ('experiment_trajectory_data_dir_name', 'randomwalk_trajectory_data', 'Name of directory to store experiment trajectory data in envlogger format from DeepMind.')
FLAGS = flags.FLAGS 


def logging_logistics():
    # Create a log file name for this training run / experiment based on hyperparams
    experiment_name   = f"{FLAGS.rtl_top}_design_{FLAGS.max_steps}_steps_{FLAGS.num_episodes}_episodes" 
    csv_data_dir_path = os.path.join(FLAGS.experiment_summary_data_dir_path, 
                                     FLAGS.experiment_csv_data_dir_name,
                                     FLAGS.reward_formulation, 
                                     experiment_name)
    trajectory_data_dir_path = os.path.join(FLAGS.experiment_summary_data_dir_path, 
                                            FLAGS.experiment_trajectory_data_dir_name,
                                            FLAGS.reward_formulation, 
                                            experiment_name)
    if not os.path.exists(csv_data_dir_path):
        os.makedirs(csv_data_dir_path)
    if FLAGS.use_envlogger:
        if not os.path.exists(trajectory_data_dir_path):
            os.makedirs(trajectory_data_dir_path)

    return csv_data_dir_path, trajectory_data_dir_path


def log_fitness_to_csv(csv_data_dir_path, fitness_hist):
    """Logs fitness history to csv files.

    Args:
        csv_data_dir_path (str): path to where data will be stored in csv format
        fitness_hist (dict): dictionary containing the fitness history
    """
    # Log reward data separately to see how reward improves over time easily
    reward_df       = pd.DataFrame([fitness_hist['reward']])
    reward_csv_path = os.path.join(csv_data_dir_path, "fitness.csv")
    reward_df.to_csv(reward_csv_path, index=False, header=False, mode='a')

    # Log all fitness data (action, obs, reward) in different file to see agent trajectory over time 
    trajectory_df       = pd.DataFrame([fitness_hist])
    trajectory_csv_path = os.path.join(csv_data_dir_path, "trajectory.csv")
    trajectory_df.to_csv(trajectory_csv_path, index=False, header=False, mode='a')


def wrap_in_envlogger(env, trajectory_data_dir_name):
    """Wraps the environment in envlogger.

    Args:
        env : OpenLane gym environment
        trajectory_data_dir_name (str): path to the directory where the trajectory data will be logged by envlogger
    """
    metadata = {
            "agent_type": "Random Walker",
            "max_steps" : FLAGS.max_steps,
            "env_type"  : type(env).__name__,
            }

    logging.info('Wrapping OpenLane environment with EnvironmentLogger...')
    env = envlogger.EnvLogger(env,
                              data_directory=trajectory_data_dir_name, 
                              max_episodes_per_file=1000,
                              metadata=metadata)
    logging.info('Done wrapping OpenLane environment with EnvironmentLogger.')
    return env
    

def main(_):
    # Instantiate OpenLane Environment
    OpenLane_env = OpenLane_wrapper.make_OpenLaneEnvironment(rtl_top=FLAGS.rtl_top, reward_formulation=FLAGS.reward_formulation, max_steps=FLAGS.max_steps)

    # Setup directories for logging data from experiment in generic csv format and trajectory data in envlogger format (if desired)
    csv_data_dir_path, trajectory_data_dir_path = logging_logistics()

    # Wrap environment in envlogger if logging trajectory data in envlogger format
    if FLAGS.use_envlogger:
        OpenLane_env = wrap_in_envlogger(OpenLane_env, trajectory_data_dir_path)

    # Create a helper that will randomly generate actions for our random walker agent
    OpenLane_helper = helpers()

    # Run training 
    for episode in range(FLAGS.num_episodes):
        print(f"Beginning episode #{episode + 1}.")  # episode num is zero indexed
        
        for step in range(FLAGS.max_steps):
            # Random walker agent: generate random action (choices for OpenLane parameters)
            action = OpenLane_helper.get_OpenLane_random_action()

            # Take a step in the environment using this action and retreive reward + observation
            _, reward, _, observation = OpenLane_env.step(action)  # TimeStep object returned from .step() which is tuple of (step_type, reward, discount, observation)

            # Log the fitness of this action with its reward and observation
            fitness_hist           = {}
            fitness_hist['action'] = action
            fitness_hist['obs']    = observation
            fitness_hist['reward'] = reward if reward is not None else [-1.0]  # None returned for reward at start of each episode; log as -1 instead of None.
            log_fitness_to_csv(csv_data_dir_path, fitness_hist) 

        print(f"Ending episode #{episode + 1}.")  # episode num is zero indexed

    print(f"Training complete: {FLAGS.num_episodes} episodes finished.")

    return

if __name__ == "__main__":
    app.run(main)
