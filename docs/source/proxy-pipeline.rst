ML Proxy Pipeline
=================

Architecture simulators are slow to generate data, and thus, we can use ML proxy models to speed up the process. By utilizing an accurate and high-speed proxy model, we can augment conventional slower architectural simulators while retaining their original interfaces.
Regardless of the proxy model type, all models can be encapsulated using the same interface. 

An example of a training algorithm:

.. toctree::
    :maxdepth: 1

    bayesian_ridge