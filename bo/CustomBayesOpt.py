#!/usr/bin/env python3
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import make_scorer
import CustomEstimator
import numpy as np

def scorer(estimator, X, y=None):
   # definition of "good" score is minimum 
   # but default is higher score is better so * -1 for our purposes 
   return -1 * estimator.fit(X, y)


def find_best_params_test(X):
    model = CustomEstimator.CustomEstimator(1,2,3)
    # Note need to use scipy=1.5.2 & scikit-learn=0.23.2 for this, see:
    # https://github.com/scikit-optimize/scikit-optimize/issues/978
    opt = BayesSearchCV(
        estimator=model,
        search_spaces={
        'a_': Real(1e-6, 1e+6, prior='log-uniform'),
        'b_': Real(1e-6, 1e+1, prior='log-uniform'),
        'c_': Integer(1,8),
        'd_': Categorical(['linear', 'poly', 'constant']),
        },
        n_iter=32,
        random_state=0,
        scoring=scorer,
        n_jobs=1,
    )

    # executes bayesian optimization
    opt.fit(X)
    print(opt.best_params_)
    return opt.best_params_


if __name__ == "__main__":
   dummy_X = np.array([1,2,3,4,5])
   find_best_params_test(dummy_X)
