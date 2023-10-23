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

The goal is to use the algorithms available in OSS-Vizier to find the optimal values for these parameters. 

Run the required script in the arch-gym conda environment. These scripts are present in sims/customenv:

* **Random Walker**: ```python train_randomwalker.py```

* **Random Search**: ```python train_randomsearch_vizier.py```

* **Quasi Random**: ```python train_quasirandom_vizier.py```

* **Grid Search**: ```python train_gridsearch_vizier.py```

* **NSGA2**: ```python train_NSGA2_vizier.py```

* **EMUKIT_GP**: ```python train_EMUKIT_GP_vizier.py```
 
