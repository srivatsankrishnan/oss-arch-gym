import numpy as np
import pandas as pd
import time
import os
os.sys.path.insert(0, os.path.abspath('../../'))

from aco.DeepSwarm.deepswarm.backends import Dataset, SniperBackend
from aco.DeepSwarm.deepswarm.deepswarm import DeepSwarm


# Dummy "training" input for POC
x_train = np.array([1,2,3,4,5])
print("Number of original training examples:", len(x_train))
print("Number of original test examples:", len(x_train))

dataset = Dataset(training_examples=x_train, training_labels=None, testing_examples=x_train, testing_labels=None)
backend = SniperBackend(dataset=dataset)
deepswarm = DeepSwarm(backend=backend)

topology = deepswarm.find_topology()
