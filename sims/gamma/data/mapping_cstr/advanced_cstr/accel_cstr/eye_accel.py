accel_cstr = {}
accel_cstr["L2"] = {
    "inbuffer":"ScratchPad",
    "outbuffer":"ScratchPad",
    "weightbuffer":"ScratchPad",
    "distrNoc":"Bus",
    "reduceNoc":"ReduceAndFoward",
}
accel_cstr["L1"] = {
    "inbuffer": "ScratchPad",
    "outbuffer": "ScratchPad",
    "weightbuffer": "ScratchPad",
    "distrNoc":"Bus",
    "reduceNoc":"Temporal",
}




