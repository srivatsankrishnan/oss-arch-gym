# Custom Environment

Custom Environment is designed to generalize the Gym interface for applying any machine learning (ML) algorithms. This environment demonstrates how various ML techniques have been utilized in designing specialized hardware. It aims to recreate results and overcome the challenges associated with complex architecture simulators, which can be slow and create barriers to entry for ML-aided design. ArchGym strives to lower this barrier by providing a general interface that allows ML-aided design without being tied to a specific ML algorithm or simulator type.

## Example

Let's consider a hypothetical architecture with four parameters:

| Parameter   | Type            | Possible Values        |
|-------------|-----------------|------------------------|
| num_cores   | Integer         |                        |
| freq        | Float           |                        |
| mem_type    | Enumeration     | {DRAM, SRAM, Hybrid}   |
| mem_size    | Integer         |                        |

The goal is to use the algorithms available in OSS-Vizier to find the optimal values for these parameters. The following `custom_env.py` file demonstrates this custom environment.

For instance, if you want to use the "RANDOM_SEARCH" algorithm, you can utilize the `train_randomsearch_vizier.py` file. You can easily switch to another supported algorithm by changing the line `study_config.algorithm = vz.Algorithm.RANDOM_SEARCH` to `study_config.algorithm = vz.Algorithm.<ALGORITHM_NAME>`. Additionally, ensure that you modify directory names in the following locations to keep the data logs separate for each algorithm:

1. `flags.DEFINE_string('traject_dir',  '<algo_name>_trajectories',  'Directory to save the dataset.')`
2. `log_path = os.path.join(FLAGS.summary_dir, '<algo_name>_logs', FLAGS.reward_formulation, exp_name)`

This is done to ensure that data logs are saved in the respective algorithm directory and do not overwrite data from other algorithms.
