## How to specify Mapping Constraint ##
We use an example to demonstrait how to customize a mapping constraint.

The following is ``example_map`` example in ``data/mapping_cstr``

```python
mapping_cstr = {}
mapping_cstr["L2"] = {"sp": np.array(["K"]),
                      "K":"K",
                      }
mapping_cstr["L1"] = {"sp":np.array(["C"]),
                      "sp_sz":16,
                      "order": ["C", "Y", "X", "R"],
                      "K":"K",
                      "C":1,
                      "Y":"R",
                      "X":"S",
                      }
```

In this example, we assume there are two levels of mapping L2 and L1. For example the mapping level, we can specify the following constraints:
* ``sp``: The parallelism dimension. If ``sp`` is specified, the parallelism dimension at this level will be fixed across the mapping search.
  * In this example, L1-mapping is fixed to parallelize across ``C``-dim.
* ``sp_sz``: Number of parallelism across the parallelism dimension at this mapping level.
  * In this example, L1-mapping will have ``16`` parallelism across ``C``-dim.
* ``order``: The loop order.
    * In this example, the loop order is fixed to ``C -> Y -> X -> R -> S -> K``.
    * We can also only fixed some of the outer-loop and make inner-loop flexiblie. 
        * E.g., ``order:  ["C", "Y"]`` means we only fixed the two most outer-loop to  ``C -> Y`` while the four inner-loop can have any permutations.
* ``tile size``: The tile size of each dimension. There are possible cases
    * Fixed integer tile sizes: Fixed to a specific given integer tile sizes.
      * E.g., L1-mapping `C`-dim has tile size of 1.
    *  Fixed fully un-folded tile sizes: Fixed to the same tile size as its parent mapping level.
       * E.g., L1-mapping `K`-dim will be fully un-folded and use the exact same tile size of `K`-dim in L2-mapping.
       * E.g., L1-mapping `Y`-dim use the same tile size of `R`-dim in L2-mapping. Similarly,  L1-mapping `X`-dim use the same tile size of `S`-dim in L2-mapping.
       * If it is at outer-most mapping level, then the parent mapping level refers to the actual layer dimension.
       * E.g., if the current layer has `K=64`, then the `K`-dim of L2-mapping is `64`. 
    * No specficiation: means no constraint on this dimension.
        * E.g., `R` and `S` dimensions are not specfied. Therefore there are no consrtaints on these two dimensions.

We can do the same constraint formulation for all the mapping levels (e.g., L2-mapping and L1-mapping in this example). 




### Mapping Constraint for 16 flexibility classes ###
We also give example how to specify the mapping constraints for 16 different flexibility classes, defined in [A formalisim of Accelerator Flexiblity](https://dl.acm.org/doi/10.1145/3530907) in ``data/mapping_cstr/flexiblity_classes_examples``

For example when using the example ``flexiblity_classes_example/dla_dir/dla_map_0000``
* mapping_cstr: setting ``mapping_cstr`` to ``flexiblity_classes_example.dla_dir.dla_map_0000``


### Advanced Constraint ###
We support two additional advanced constraints: accelerator HW configuration consrtraints and cost-model specific constraints. More detailed can be found in [advanced_cstr](./advanced_cstr).

We suggest normal users to use the default setting on these two constraints, except:
* Accelerator HW designer/researcher: you may want to take a look at ``accel_cstr``
* DNN cost-model developer: you may want to look at ``costmodel_cstr``