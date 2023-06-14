from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import make_scorer
import os
import sys
sys.path.append("../../")
from bo.SniperEstimator import SniperEstimator
import numpy as np
import time
import pandas as pd
from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_iter', 64, 'Number of training steps.')
flags.DEFINE_integer('random_state', 1, 'Random state.')
flags.DEFINE_string('data_dir', 'data/', 'Directory to store data.')

def scorer(estimator, X, y=None):
   # definition of "good" score is minimum 
   # but default is higher score is better so * -1 for our purposes 
   return -1 * estimator.fit(X, y)

def find_best_params_test(X,parameters):
    print("Data_dir", FLAGS.data_dir)
    model = SniperEstimator(
        core_dispatch_width=2,
        core_window_size=16,
        core_outstanding_loads=2,
        core_outstanding_stores=2,
        core_commit_width=2,
        core_rs_entries=2,
        l1_icache_size=4,
        l1_dcache_size=4,
        l2_cache_size=128,
        l3_cache_size=4096,
        random_state=FLAGS.random_state,
        num_iter=FLAGS.num_iter
    )

    opt = BayesSearchCV(
        estimator=model,
        search_spaces = parameters,
        n_iter=FLAGS.num_iter,
        random_state=FLAGS.random_state,
        scoring=scorer,
        n_jobs=1,
    )
    opt.fit(X)
    print(opt.best_params_)
    
    return opt.best_params_


def main(_):
    print("Running BO for ", FLAGS.num_iter, " iterations")
    # To do : Configure the workload trace here
    dummy_X = np.array([1,2,3,4,5])

    # To do : Figure out a way to make sure sampling only returns powers of 2. Else sniper
    # will crash. Currently handled in SniperEstimator
    parameters = {"core_dispatch_width": Integer(2, 5, base=2,prior='log-uniform' ),
                    "core_window_size": Integer(16, 513, base=2,prior='log-uniform'),
                    "core_outstanding_loads": Integer(32, 96, base=2,prior='log-uniform'),
                    "core_outstanding_stores": Integer(24, 64, base=2,prior='log-uniform'),
                    "core_commit_width": Integer(32, 192, base=2,prior='log-uniform'),
                    "core_rs_entries": Integer(18, 72, base=2,prior='log-uniform'),
                    "l1_icache_size": Integer(4, 129, base=2, prior='log-uniform'),
                    "l1_dcache_size": Integer(4, 129, base=2, prior='log-uniform'),
                    "l2_cache_size": Integer(128, 2049, base=2,prior='log-uniform'),
                    "l3_cache_size": Integer(4096, 16385, base=2,prior='log-uniform'),
                }
    find_best_params_test(dummy_X,parameters)

if __name__ == '__main__':
   app.run(main)