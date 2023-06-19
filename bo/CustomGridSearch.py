#!/usr/bin/env python3
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import CustomEstimator
import numpy as np

def scorer(estimator, X, y=None):
   # definition of "good" score is minimum 
   # but default is higher score is better so * -1 for our purposes 
   return -1 * estimator.fit(X, y)


def find_best_params_test(X):
   model = CustomEstimator.CustomEstimator(1,2,3)
   gs = GridSearchCV(estimator=model, 
                     param_grid={"a_": [1,2,3], "b_":[4,5,6], "c_":[7,8,9]}, 
                     n_jobs=1, 
                     scoring=scorer
                    )
   gs.fit(X)
   print(gs.best_params_)
   return gs.best_params_


if __name__ == "__main__":
   dummy_X = np.array([1,2,3,4,5])
   find_best_params_test(dummy_X)
