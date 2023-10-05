Introduction
============
We introduce ArchGym, an open-source gymnasium and easy-to-extend framework that connects a diverse range of search algorithms to
architecture simulators. The results suggest that with an unlimited number of samples, ML algorithms are equally favorable to meet the user-defined target specification
if its hyperparameters are tuned thoroughly; no one solution is necessarily better than another. We show how using a same structure of code for different algirthms, 
we can train the agent to generate optimal parameters just by varying the name of the algorithm. 
ArchGym is a systematic and standardized framework for ML-driven research tackling architectural design space exploration. ArchGym currently supports (six) different ML-based search algorithms and three unique architecture simulators.

.. image:: ArchGym-animation.gif
    :width: 100%
    :align: center

ML Proxy Pipeline
-----------------
Architecture simulators are slow to generate data, and thus, we can use ML proxy models to speed up the process. By utilizing an accurate and high-speed proxy model, we can augment conventional slower architectural simulators while retaining their original interfaces.
Regardless of the proxy model type, all models can be encapsulated using the same interface. 

An example of a training algorithm:

.. toctree::
    :maxdepth: 1

    bayesian_ridge