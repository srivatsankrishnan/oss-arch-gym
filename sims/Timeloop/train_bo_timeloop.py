import os
from process_params import TimeloopConfigParams
import pandas as pd
import numpy as np
os.sys.path.insert(0, os.path.abspath('../../'))
from bo.TimeloopEstimator import TimeloopEstimator
import configparser
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import sys
import os
import configs.arch_gym_configs as arch_gym_configs
sys.path.append("../../")

from absl import flags
from absl import app


# parameters file
PARAMS_FILE = arch_gym_configs.timeloop_parameters

cparser = configparser.ConfigParser()
cparser.read(PARAMS_FILE)
# get the base directory from the file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("BASE_DIR: ", BASE_DIR)

flags.DEFINE_string("script", str(BASE_DIR) + "/script", "Path to the script")
flags.DEFINE_string("output", str(BASE_DIR) + "/output", "Path to the output")
flags.DEFINE_string("arch", str(BASE_DIR) + "/arch", "Path to the arch")
flags.DEFINE_string("mapper", str(BASE_DIR) + "/mapper", "Path to the mapper")
flags.DEFINE_string("workload", str(BASE_DIR) + "/layer_shapes/AlexNet", "Path to the workload")
flags.DEFINE_string("summary_dir", ".", "Path to the log")
flags.DEFINE_string('traject_dir', 'bo_trajectories', 'Directory to trajectory data.')
flags.DEFINE_string('aco_log_dir', 'bo_logs', 'Directory to store logs.')
flags.DEFINE_bool('use_envlogger', False, 'Use EnvLogger to log environment data.')

flags.DEFINE_string("params_file", str(BASE_DIR) + "/parameters.ini", "Path to the parameters file")
flags.DEFINE_integer('num_iter', 16, 'Number of training steps.')
flags.DEFINE_integer('random_state', 2, 'Random state.')
flags.DEFINE_string('exp_config_file', 'exp_config.ini', 'Experiment config file.')
flags.DEFINE_string('reward_formulation', 'energy', 'Reward formulation')

flags.DEFINE_float('target_energy', 20444.2, 'Target energy value.')
flags.DEFINE_float('target_area', 1.7255, 'Target area value.')
flags.DEFINE_float('target_cycles', 6308563, 'Target cycles value.')

FLAGS = flags.FLAGS

def scorer(estimator, X, y=None):
    # definition of "good" score is minimum
    # but default is higher score is better so * -1 for our purposes
    return -1 * estimator.fit(X, y)


def get_search_space(timeloop_config):
    param_search_space = {}
    param_search_space.update(timeloop_config.get_all_params_flattened())

    return param_search_space


def find_best_params_test(X, parameters,
                        num_iter,
                        random_state,
                        exp_name,
                        traject_dir,
                        exp_log_dir):

    output_dir = FLAGS.output
    script = FLAGS.script
    arch = FLAGS.arch
    mapper = FLAGS.mapper
    workload = FLAGS.workload
    
    reward_formulation = FLAGS.reward_formulation
    use_envlogger = FLAGS.use_envlogger

    
    model = TimeloopEstimator(script=script, traj_dir=traject_dir, exp_name=exp_name, mapper_dir=mapper,
                 output_dir=output_dir, log=exp_log_dir, reward_formulation=reward_formulation, use_envlogger=use_envlogger,
                arch=arch, target_area= FLAGS.target_area, target_energy=FLAGS.target_energy, target_cycles=FLAGS.target_cycles,
                 workload=workload, **parameters)
    
    # use config parser to update its parameters
    config = configparser.ConfigParser()
    config.read(FLAGS.exp_config_file)
    config.set("experiment_configuration", "exp_name", str(exp_name))
    config.set("experiment_configuration", "trajectory_dir", str(traject_dir))
    config.set("experiment_configuration", "log_dir", str(exp_log_dir))
    config.set("experiment_configuration", "reward_formulation", str(FLAGS.reward_formulation))
    config.set("experiment_configuration", "use_envlogger", str(FLAGS.use_envlogger))
    config.set("experiment_configuration", "target_area", str(FLAGS.target_area))
    config.set("experiment_configuration", "target_energy", str(FLAGS.target_energy))
    config.set("experiment_configuration", "target_cycles", str(FLAGS.target_cycles))

    # write the updated config file
    with open(FLAGS.exp_config_file, 'w') as configfile:
        config.write(configfile)

    opt = BayesSearchCV(
        estimator=model,
        search_spaces=parameters,
        n_iter=num_iter,
        random_state=random_state,
        scoring=scorer,
        n_jobs=1,
        cv = 2,
    )
    opt.fit(X)
    print(opt.best_params_)

    return opt.best_params_

def main(_):
    # Dummy numpy array to pass to the optimizer
    dummyX = np.array([1,2,3,4,5,6,7,8,9,10])

    timeloop_config = TimeloopConfigParams(FLAGS.params_file)
    param_search_space = get_search_space(timeloop_config)

    # define architectural parameters to search over
    parameters = {'NUM_PEs': Categorical(['PE[0..13]', 'PE[0..27]', 'PE[0..41]', 'PE[0..55]', 'PE[0..69]', 'PE[0..83]', 'PE[0..97]', 'PE[0..111]', 'PE[0..125]', 'PE[0..139]', 'PE[0..153]', 'PE[0..167]', 'PE[0..181]', 'PE[0..195]', 'PE[0..209]', 'PE[0..223]', 'PE[0..237]', 'PE[0..251]', 'PE[0..265]', 'PE[0..279]', 'PE[0..293]', 'PE[0..307]', 'PE[0..321]', 'PE[0..335]']),
    'MAC_MESH_X': Categorical([7, 14]),
    'IFMAP_SPAD_CLASS': Categorical(['regfile', 'smartbuffer_SRAM', 'smartbuffer_RF', 'SRAM']),
    'IFMAP_SPAD_ATRIBUTES.memory_depth': Categorical([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]),
    'IFMAP_SPAD_ATRIBUTES.block-size': Categorical([1, 2, 4]),
    'IFMAP_SPAD_ATRIBUTES.read_bandwidth': Categorical([2, 4, 8, 16]),
    'IFMAP_SPAD_ATRIBUTES.write_bandwidth': Categorical([2, 4, 8, 16]),
    'PSUM_SPAD_CLASS': Categorical(['regfile', 'smartbuffer_SRAM', 'smartbuffer_RF', 'SRAM']),
    'PSUM_SPAD_ATRIBUTES.memory_depth': Categorical([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]),
    'PSUM_SPAD_ATRIBUTES.block-size': Categorical([1, 2, 4]),
    'PSUM_SPAD_ATRIBUTES.read_bandwidth': Categorical([2, 4, 8, 16]),
    'PSUM_SPAD_ATRIBUTES.write_bandwidth': Categorical([2, 4, 8, 16]),
    'WEIGHTS_SPAD_CLASS': Categorical(['regfile', 'smartbuffer_SRAM', 'smartbuffer_RF', 'SRAM']),
    'WEIGHTS_SPAD_ATRIBUTES.memory_depth': Categorical([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]),
    'WEIGHTS_SPAD_ATRIBUTES.block-size': Categorical([1, 2, 4]),
    'WEIGHTS_SPAD_ATRIBUTES.read_bandwidth': Categorical([2, 4, 8, 16]), 
    'WEIGHTS_SPAD_ATRIBUTES.write_bandwidth': Categorical([2, 4, 8, 16]),
    'DUMMY_BUFFER_CLASS': Categorical(['regfile', 'smartbuffer_SRAM', 'smartbuffer_RF', 'SRAM']),
    'DUMMY_BUFFER_ATTRIBUTES.depth': Categorical([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]),
    'DUMMY_BUFFER_ATTRIBUTES.block-size': Categorical([1, 2, 4]),
    'SHARED_GLB_CLASS': Categorical(['regfile', 'smartbuffer_SRAM', 'smartbuffer_RF', 'SRAM']),
    'SHARED_GLB_ATTRIBUTES.memory_depth':Categorical([1024, 2048, 4096, 8192, 16384, 32768, 65536]),
    'SHARED_GLB_ATTRIBUTES.n_banks': Categorical([16, 32, 64, 128]),
    'SHARED_GLB_ATTRIBUTES.block-size': Categorical([1, 2, 4]),
    'SHARED_GLB_ATTRIBUTES.read_bandwidth': Categorical([2, 4, 8, 16]),
    'SHARED_GLB_ATTRIBUTES.write_bandwidth': Categorical([2, 4, 8, 16])
    }
    
    # get workload name 
    # experiment name
    if "AlexNet" in FLAGS.workload:
        workload = "AlexNet"
    elif "resnet" in FLAGS.workload:
        workload = "ResNet"
    elif "mobilenet" in FLAGS.workload:
        workload = "mobilenet"
    
    # Construct the exp name from seed and num_iter
    exp_name = str(workload) + "_random_state_" + str(FLAGS.random_state) + "_num_iter_" + str(FLAGS.num_iter)
   
    # get the current working directory and append the exp name
    traject_dir = os.path.join(FLAGS.summary_dir, FLAGS.traject_dir, FLAGS.reward_formulation, exp_name)

    # log directories for storing exp csvs
    exp_log_dir = os.path.join(FLAGS.summary_dir, "bo_logs", FLAGS.reward_formulation, exp_name)

    print("Trajectory directory: " + traject_dir)

    find_best_params_test(dummyX, parameters,
                        FLAGS.num_iter,
                        FLAGS.random_state,
                        exp_name,
                        traject_dir,
                        exp_log_dir)

if __name__ == '__main__':
   app.run(main)
   