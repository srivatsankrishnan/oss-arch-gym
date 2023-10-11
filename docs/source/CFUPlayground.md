CFU Playground
===

CFU Playground is a full-stack open-source framework for TinyML Acceleration. The tool enables users to rapidly design and prototype tightly coupled accelerators and explore novel design space tradeoffs between CPU and accelerator. 

## Installing CFU Playground

In order to install CFU Playground, run the following script from the `oss-arch-gym` repository root:

```sh
git submodule update --init sims/CFU-Playground/CFU-Playground
```

- Move into the CFU Playrgoud directory:
```sh 
cd sims/CFU-Playground/CFU-Playground
```
- Run the following from this location:
```sh
./scripts/setup
./scripts/setup_vexriscv_build.sh
make install-sf
```
Now you should be able to use the CFU-env gym environment.

To invoke CFU Playground from arch-gym (after completing above installation steps):
```
cd oss-arch-gym/sims/CFU-Playground/
```
```
make enter-sf
```

## Run Training Scripts

Inside sims/CFU-Playground:

* **Random Walker**: ```python train_randomwalker_CFUPlayground.py```

* **Random Search**: ```python train_randomsearch_CFUPlayground.py```

* **Quasi Random**: ```python train_quasirandom_CFUPlayground.py```

* **Grid Search**: ```python train_gridsearch_CFUPlayground.py```

* **NSGA2**: ```python train_NSGA2_CFUPlayground.py```

* **EMUKIT_GP**: ```python train_EMUKIT_GP_CFUPlayground.py```




To update the parameters such as workload, num_steps, reward_formulation for the training scripts, follow these steps:
...

```
python training_script.py --parameter=value
```

## Parameter Space
| System Parameter       | Values        |
| ----------------       | ------------- |
|Bypass                  | True, False  
|CFU Enable              | True, False  
|Branch Prediction       | None, Static, Dynamic, Dynamic Target                           
|Instruction Cache Size  | 0-16KiB                               
|Data Cache Size         | 0-16KiB                               
|Hardware Multiplier     | True, False                           
|Hardware Divider        | True, False                           
|Single Cycle Multiplier | True, False                           
|Single Cycle Shifter    | True, False                           
|Safe                    | True, False                           
