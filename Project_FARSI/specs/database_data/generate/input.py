#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

from specs.LW_cl import *
gen_config = {}

# ---------------------------
# global variables
# ---------------------------
# HW database
sim_time_per_design = .25 / (3600)
gen_config["DS_output_mode"] = "DB_DS_time"  # ["DS_size", "DS_time"]
gen_config["DB_MAX_PE_CNT_range"] = range(1, 4)
gen_config["DB_MAX_TASK_CNT_range"] = range(5, 10)
gen_config["DB_MAX_PAR_TASK_CNT_range"] = range(5, 7)
gen_config["DB_DS_type"] = "exhaustive_naive"
gen_config["DB_DS_type"] = "exhaustive_reduction_DB_semantics"

# SW database
gen_config["DB_MAX_TASK_CNT"] = 10  # does include souurce and siink
gen_config["DB_MAX_PAR_TASK_CNT"] = 8 # souurce and siink would be automatically serialized

gen_config["DB_MAX_PE_CNT"] = 4
gen_config["DB_MAX_BUS_CNT"] = 4
gen_config["DB_MAX_MEM_CNT"] = 4

gen_config["DB_PE_list"] = ["A53_pe"]
gen_config["DB_BUS_list"] = ["LMEM_ic_0_ic"]
gen_config["DB_MEM_list"] = ["LMEM_0_mem"]

gen_config["DB_MIN_PE_CNT"] = 4
gen_config["DB_MIN_MEM_CNT"] = 4
gen_config["DB_MIN_BUS_CNT"] = 4

gen_config["DB_MAX_SYSTEM_to_investigate"] = 1
assert (gen_config["DB_MIN_PE_CNT"] <= gen_config["DB_MAX_PE_CNT"])
assert (gen_config["DB_MIN_BUS_CNT"] <= gen_config["DB_MAX_BUS_CNT"])
assert (gen_config["DB_MIN_MEM_CNT"] <= gen_config["DB_MAX_MEM_CNT"])
assert (gen_config["DB_MAX_PE_CNT"] >= gen_config["DB_MAX_MEM_CNT"])
assert (gen_config["DB_MAX_PE_CNT"] >= gen_config["DB_MAX_BUS_CNT"])
assert(gen_config["DB_MAX_PE_CNT"] <= gen_config["DB_MAX_TASK_CNT"] -1)  # have to make sure that is not isolated
assert(gen_config["DB_MAX_MEM_CNT"] <= gen_config["DB_MAX_TASK_CNT"] -1)  # have to make sure that is not isolated
assert(gen_config["DB_MAX_BUS_CNT"] <= gen_config["DB_MAX_TASK_CNT"] -1)  # have to make sure that is not isolated



budgets_dict = defaultdict(dict)
# some numbers for now. This doesn't matter at the momoent. 
budgets_dict["glass"] = {}
budgets_dict["glass"]["latency"] = {"synthetic": .030}
budgets_dict["glass"]["power"] = 20*10**-3
budgets_dict["glass"]["area"] = 15*10**-6
other_values_dict = defaultdict(dict)
other_values_dict["glass"]["cost"] = 10**-9  # something really small