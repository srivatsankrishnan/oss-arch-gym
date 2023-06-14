from utils.accel_cstr_class import *

def get_accel_cstr(accel):
    accel_cstr = Accel_cstr()
    if accel == "eye":
        accel_cstr.set_cstr(level="L2", inbuffer="ScratchPad", outbuffer="ScratchPad", weightbuffer="ScratchPad", distrNoc="Bus", reduceNoc="Systolic")
        accel_cstr.set_cstr(level="L1", distrNoc="Bus", reduceNoc="Systolic")
    return accel_cstr.accel_cstr

def put_into_actual_cstr(mapping_cstr, cstr_container):
    for level, cstr_dict in mapping_cstr.items():
        cstr_container.set_constraint(level=level, cstr=cstr_dict)

def translate_to_actual_cstr(accel_cstr, cstr_container):
    for level, dicts in accel_cstr.items():
        cstr_dict = {}
        for value in dicts.values():
            if value == "FIFO":
                cstr_dict["Y"] = [1,"R"]
                cstr_dict["X"] = [1,"S"]
                # cstr_dict["Y"] = "R"
                # cstr_dict["X"] = "S"
            if value =="Bus" or value == "Tree":
                cstr_dict["sp"] = ["K", "Y", "X", "R", "S"]
            if value == "Temporal":
                cstr_dict["sp"] = ["K", "Y", "X"]
            if value =="AdderTree":
                cstr_dict["sp"] = ["C", "R", "S"]
            if value =="ReduceAndFoward":
                cstr_dict["sp"] = ["C", "R", "S"]
        cstr_container.set_constraint(level=level, cstr=cstr_dict)