## Basic Usage ##
[GAMMA: Automating the HW Mapping of DNN Models on
Accelerators via Genetic Algorithm](https://dl.acm.org/doi/10.1145/3400302.3415639)

In the basic usage, Gamma will create a map space assuming full flexibility in the underlying hardware.
### Basic Parameter ###
* fitness1: The first fitness objective (latency/ power/ energy)
* fitness2: The second fitness objective (latency/ power/ energy)
* model: The model to run (available model in data/model)
* singlelayer: The layer index of the selected model, if want to optimize only single layer. If want to optimize all layers, skip this specification.
* num_pe: Number of PEs
* l1_size: L1 size (Number of elements)
* l2_size: L2 size (Number of elements)
* slevel_min: The minimum number of parallelism
* slevel_max: The maximum number of parallelism. The number of parallelism will be in the range [slevel_min, slevel_max]
* hwconfig: Read in HW configuration from file. An example of hwconfig can be found [here](data/HWconfigs/hw_config.m). An example of using it can be found [here](../run_gamma_with_hwconfig.sh)
* epochs: Number of generation for the optimization
* outdir: The output result directory



## Advanced Usage (Constrained Map Space Exploration) ##
[A formalisim of Accelerator Flexiblity](https://dl.acm.org/doi/10.1145/3530907)

We can also specify mapping constraint if the underlying hardware is not fully flexible. Gamma will create a constraint map-space based on the given mapping constraint.
### Advanced Parameter ###
* mapping_cstr: The constraint on the map-space.
  * Make a customized mapping constraint files and put it in this path [``data/mapping_cstr``](../../data/mapping_cstr).   

For example, let's apply an mapping constraint called ``dla_map`` (located at [``data/mapping_cstr/dla_map.py``](../../data/mapping_cstr/dla_map.py))
```
python main.ppy --mapping_cstr dla_map
```

We give detailed explanations and provide examples in [``data/mapping_cstr``](../../data/mapping_cstr).


## Advanced Usage (PE-Mapping Co-exploration) ##
[DiGamma: Domain-aware Genetic Algorithm for HW-Mapping Co-optimization for DNN Accelerators](https://arxiv.org/pdf/2201.11220.pdf)

We can also support PE (HW) and mapping Co-exploration. In this mode, we use area (PE area + buffer area) as constraint.
### Advanced Parameter ###
* **num_pe: To execute in this mode, set num_pe to -1.**
* area_budget: The area budget for compute (PE) and memory (L1 buffer + L2 buffer).
* pe_limit:The upper-bound for available number of PE.

