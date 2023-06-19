#!/usr/bin/env python3

import numpy as np
from deepswarm.backends import Dataset, DRAMSysBackend
from deepswarm.deepswarm import DeepSwarm

# Dummy "training" input for POC
x_train = np.array([1,2,3,4,5])
print("Number of original training examples:", len(x_train))
print("Number of original test examples:", len(x_train))

dataset = Dataset(training_examples=x_train, training_labels=None, testing_examples=x_train, testing_labels=None)
backend = DRAMSysBackend(dataset=dataset)
deepswarm = DeepSwarm(backend=backend)
topology = deepswarm.find_topology()

