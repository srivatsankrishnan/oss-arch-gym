## Advanced Constraint Setting ##
We support two kinds of advance accelerator constraint settings: ``accel_cstr`` and ``costmodel_cstr``.

### Accelerator HW types constraints ###
Different HW choices (Buffer and NoCs) will also infer different constraints on the mapping.

* Buffer: We support ``FIFO``, ``ScratchPad``
* Reduction/ Distribution NoC: We support ``Bus``, ``Tree``, ``Temoporal``, ``ReduceAndFoward``, ``AdderTree``

We showed an example in ``dla_accel``
```python
accel_cstr = {}
accel_cstr["L2"] = {
    "inbuffer":"ScratchPad",
    "outbuffer":"ScratchPad",
    "weightbuffer":"ScratchPad",
    "distrNoc":"Systolic",
    "reduceNoc":"Temporal",
}
accel_cstr["L1"] = {
    "inbuffer": "FIFO",
    "outbuffer": "FIFO",
    "weightbuffer": "FIFO",
    "distrNoc":"Systolic",
    "reduceNoc":"ReduceAndFoward",
}

```

### Constraint from cost-model ###

We use MAESTRO as our underlying cost-model. MAESTRO as a cost model has the constraint of R and S should be fully unfolded. Therefore we add a ``maestro_cstr`` to specify this chaterestics.

When adopting GAMMA to differnt cost-model, ``maestro-cstr`` is not needed.