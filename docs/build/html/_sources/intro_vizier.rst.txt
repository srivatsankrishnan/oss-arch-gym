Introduction
============

Open Source (OSS) Vizier is a Python-based interface for blackbox optimization and research, based on Google's original internal Vizier, one of the first hyperparameter tuning services designed to work at scale.

Objective
==========
We introduce ArchGym, an open-source gymnasium and easy-to-extend framework that connects a diverse range of search algorithms to
architecture simulators. The results suggest that with an unlimited number of samples, ML algorithms are equally favorable to meet the user-defined target specification
if its hyperparameters are tuned thoroughly; no one solution is necessarily better than another. We show how using a same structure of code for different algirthms, 
we can train the agent to generate optimal parameters just by varying the name of the algorithm. 

Supported Algorithms
====================

The following algorithms are currently supported the vizier version:

1. Random Search (``RANDOM_SEARCH``): Flat Search Spaces.
2. Quasi-Random Search (``QUASI_RANDOM_SEARCH``): Flat Search Spaces.
3. Grid Search (``GRID_SEARCH``): Flat Search Spaces.
4. Emukit Bayesian Optimization (``EMUKIT_GP_EI``): Flat Search Spaces.
5. NSGA2 (``NSGA2``) : Flat Search Spaces.

