Custom Environment
==================

Custom Environment - While, gym environment is primarily designed for applying reinforcement learning algorithms, we show how we can generalize that interface to use it to apply any ML algorithms. 
We show how different ways in which various machine learning techniques have been applied to design specialized hardware. 
Recreating these results and overcoming the challenges of getting to work with complex architecture simulators, its sheer slowness increases the barrier of entry for ML-aided design. 
Hence, ArchGym tries to lower this barrier by advocating for a general interface through which we can apply ML-aided design while being agnostic to type of ML algorithm or the simulator type.

Example 
*******

Let's say we have a hypothetical architecture which has four parameters, namely, number of cores (``num_cores``, type integer), frequency (``freq``, type float), memory type (``mem_type``, {DRAM, SRAM, Hybrid}) 
and memory size (``mem_size``, type integer). 
We want to use the algorithms available in OSS-Vizier to find the optimal values of these parameters. 
The file ``custom_env.py`` shows this particular customised environment.
For example, if we want to use ``RANDOM_SEARCH`` algorithm, we can use ``train_randomsearch_vizier.py`` file. The structure of this code, Detailed description can be found here (add hyperlink for this module)
If you wish to use any other of the supported algorithms, simply change line ``study_config.algorithm = vz.Algorithm.RANDOM_SEARCH`` to ``study_config.algorithm = vz.Algorithm.<ALGORITHM_NAME>``. Also, make sure to change the names of directories in the following places :
``flags.DEFINE_string('traject_dir',  '<algo_name>_trajectories',  'Directory to save the dataset.')``, 
``log_path = os.path.join(FLAGS.summary_dir, '<algo_name>_logs', FLAGS.reward_formulation, exp_name)``. This is done to ensure that the data logs in the respective algorithm directory only
and does not override other algorithms' data log. 

