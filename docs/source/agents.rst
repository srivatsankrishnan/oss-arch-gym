Agents
======
We define “agent” as an encapsulation of the machine learning algorithm. An ML algorithm consists of “hyperparameters” and a guiding “policy”. We currently support the following agents:

* Ant Colony Optimization (ACO)
* Genetic Algorithm (GA)
* Bayesian Optimization (BO)
* Reinforcement Learning (RL)
* Random Walker (RW)
* Vizier Algorithms
   1. Random Search (``RANDOM_SEARCH``): Flat Search Spaces.
   2. Quasi-Random Search (``QUASI_RANDOM_SEARCH``): Flat Search Spaces.
   3. Grid Search (``GRID_SEARCH``): Flat Search Spaces.
   4. Emukit Bayesian Optimization (``EMUKIT_GP_EI``): Flat Search Spaces.
   5. NSGA2 (``NSGA2``) : Flat Search Spaces.

