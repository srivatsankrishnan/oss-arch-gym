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


