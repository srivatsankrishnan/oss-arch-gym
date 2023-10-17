CFU Playground
===

CFU Playground is a full-stack open-source framework for TinyML Acceleration. The tool enables users to rapidly design and prototype tightly coupled accelerators and explore novel design space tradeoffs between CPU and accelerator. 

## Installing CFU Playground

From the repository root, run the following script:
```
./install_sim cfu
```

## Run Training Scripts

Run the required script in the arch-gym conda environment. These scripts are present in sims/CFU-Playground:

* **Random Walker**: ```python train_randomwalker_CFUPlayground.py```

* **Random Search**: ```python train_randomsearch_CFUPlayground.py```

* **Quasi Random**: ```python train_quasirandom_CFUPlayground.py```

* **Grid Search**: ```python train_gridsearch_CFUPlayground.py```

* **NSGA2**: ```python train_NSGA2_CFUPlayground.py```

* **EMUKIT_GP**: ```python train_EMUKIT_GP_CFUPlayground.py```

## Configuration options

There are various workloads available to train for in CFU Playground:

* micro_speech, magic_wand, mnv2, hps, mlcommons_tiny_v01_amond, mlcommons_tiny_v01_imgc, mlcommons_tiny_v01_kws, mlcommons_tiny_v01_vww

There are also the following embench workloads available:

* primecount, minver, aha_mont64, crc_32, cubic, edn, huffbench, matmul, md5, nbody, nettle_aes, nettle_sha256, nsichneu, picojpeg, qrduino, slre, st, statemate, tarfind, ud, wikisort

To update various parameters such as workload, num_steps, reward_formulation for the training scripts, follow these steps:

```
python training_script.py --parameter=value
```

| Script Parameter       | Values        | default|
| ----------------       | ------------- | ----------| 
|workload                | any of the workloads listed above| micro_speech
|num_steps              | any integer  | 1
|traject_dir | directory name (relative to sims/CFU-Playground) | \<training algorithm name>_trajectories
|use_envlogger| boolean | True
|reward_formulation| both, cells, cycles| both          

## Design Space

Currently, CFU-Playground allows the exploration of the following design space:

| System Parameter       | Values        |
| ----------------       | ------------- |
|Bypass                  | True, False  
|Branch Prediction       | None, Static, Dynamic, Dynamic Target                           
|Instruction Cache Size  | 0-16KiB                               
|Data Cache Size         | 0-16KiB                               
|Hardware Multiplier     | True, False                           
|Hardware Divider        | True, False                           
|Single Cycle Multiplier | True, False                           
|Single Cycle Shifter    | True, False                           
|Safe                    | True, False                           
