#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

from specs.LW_cl import *
from specs.parse_libraries.parse_library import  *
import importlib
from specs.gen_synthetic_data import *


#  -----------------
#   This class helps populating the database. It uses the sw_hw_database_population mode to do so.
#   the mode can be hardcoded, parse (parse a csv) or generate (synthetic generation)
#  -----------------
class database_input_class():
    def append_tasksL(self, tasksL_list):
        # one tasks absorbs another one
        def absorb(ref_task, task):
            ref_task.set_children_nature({**ref_task.get_children_nature(), **task.get_children_nature()})
            ref_task.set_self_to_children_work_distribution({**ref_task.get_self_to_children_work_distribution(),
                                                             **task.get_self_to_children_work_distribution()})
            ref_task.set_self_to_children_work({**ref_task.get_self_to_children_work(),
                                                             **task.get_self_to_children_work()})
            ref_task.set_children(ref_task.get_children() + task.get_children())

        for el in tasksL_list:
            refine = False
            for el_ in self.tasksL:
                if el_.task_name == el.task_name:
                    refine = True
                    break
            if refine:
                absorb(el_, el)
            else:
                self.tasksL.append(el)

    def append_blocksL(self, blocksL_list):
        for el in blocksL_list:
            append = True
            for el_ in self.blocksL:
                if el_.block_instance_name == el.block_instance_name:
                    append = False
                    break
            if append:
                self.blocksL.append(el)

    def append_pe_mapsL(self, pe_mapsL):
        for el in pe_mapsL:
            append = True
            for el_ in self.pe_mapsL:
                if el_.pe_block_instance_name == el.pe_block_instance_name and el_.task_name == el.task_name:
                    append = False
                    break
            if append:
                self.pe_mapsL.append(el)

    def append_pe_scheduels(self, pe_scheduels):
        for el in pe_scheduels:
            append = True
            for el_ in self.pe_schedeulesL:
                if el_.task_name == el.task_name:
                    append = False
                    break
            if append:
                self.pe_schedeulesL.append(el)




    #glass_constraints = {"power": config.budget_dict["glass"]["power"]}  #keys:power, area, latency  example: {"area": area_budget}
    def __init__(self,
                 sw_hw_database_population={"db_mode":"hardcoded", "hw_graph_mode":"generated_from_scratch",
                                            "workloads":{},"misc_knobs":{}}):
        # some sanity checks first
        assert(sw_hw_database_population["db_mode"] in ["hardcoded", "generate", "parse"])
        assert(sw_hw_database_population["hw_graph_mode"] in ["generated_from_scratch", "generated_from_check_point", "parse", "hardcoded", "hop_mode", "star_mode"])
        if sw_hw_database_population["hw_graph_mode"] == "parse":
            assert(sw_hw_database_population["db_mode"] == "parse")

        # start parsing/generating
        if sw_hw_database_population["db_mode"] == "hardcoded":
            lib_relative_addr = config.database_data_dir.replace(config.home_dir, "")
            lib_relative_addr_pythony_fied = lib_relative_addr.replace("/",".")
            files_to_import = [lib_relative_addr_pythony_fied+".hardcoded."+workload+".input"  for workload in sw_hw_database_population["workloads"]]
            imported_databases = [importlib.import_module(el) for el in files_to_import]
        elif sw_hw_database_population["db_mode"] == "generate":
            lib_relative_addr = config.database_data_dir.replace(config.home_dir, "")
            lib_relative_addr_pythony_fied = lib_relative_addr.replace("/",".")
            files_to_import =   [lib_relative_addr_pythony_fied+".generate."+"input"  for workload in sw_hw_database_population["workloads"]]
            imported_databases = [importlib.import_module(el) for el in files_to_import]

        self.blocksL:List[BlockL] = []  # collection of all the blocks
        self.pe_schedeulesL:List[TaskScheduleL] = []
        self.tasksL: List[TaskL] = []
        self.pe_mapsL: List[TaskToPEBlockMapL] = []
        self.souurce_memory_work = {}
        self.workloads_last_task = []
        self.workload_tasks = {}
        self.task_workload = {}
        self.misc_data  = {}
        self.parallel_task_names = {}
        self.hoppy_task_names = []
        self.hardware_graph = ""
        self.task_to_hardware_mapping = ""
        self.parallel_task_count = "NA"
        self.serial_task_count = "NA"
        self.memory_boundedness_ratio = "NA"
        self.datamovement_scaling_ratio = "NA"
        self.parallel_task_type = "NA"
        self.num_of_hops = "NA"
        self.num_of_NoCs = "NA"

        # using the input files, populate the task graph and possible blocks and the mapping of tasks to blocks
        if sw_hw_database_population["db_mode"] == "hardcoded":
            if len(imported_databases) > 1:
                print("we have to fix the budets_dict collection here. support it and run")
                exit(0)
            #self.workload_tasks[sw_hw_database_population["workloads"][0]] = []
            for imported_database in imported_databases:
                self.blocksL.extend(imported_database.blocksL)
                self.tasksL.extend(imported_database.tasksL)
                self.pe_mapsL.extend(imported_database.pe_mapsL)
                self.pe_schedeulesL.extend(imported_database.pe_schedeulesL)
                self.workloads_last_task = imported_database.workloads_last_task
                self.budgets_dict = imported_database.budgets_dict
                self.other_values_dict = imported_database.other_values_dict
                self.souurce_memory_work = imported_database.souurce_memory_work
                self.misc_data["same_ip_tasks_list"] = imported_database.same_ip_tasks_list
            self.workload_tasks[list(sw_hw_database_population["workloads"])[0]] = [el.task_name for el in self.tasksL]
            for el in self.tasksL:
                self.task_workload[el.task_name] = list(sw_hw_database_population["workloads"])[0]

            self.sw_hw_database_population = sw_hw_database_population

        elif sw_hw_database_population["db_mode"] == "parse":
            for workload in sw_hw_database_population["workloads"]:
                tasksL_, data_movement = gen_task_graph(os.path.join(config.database_data_dir, "parsing"), workload+"_database - ", sw_hw_database_population["misc_knobs"])
                blocksL_, pe_mapsL_, pe_schedulesL_ = gen_hardware_library(os.path.join(config.database_data_dir, "parsing"), workload+"_database - ", workload, sw_hw_database_population["misc_knobs"])
                self.sw_hw_database_population = sw_hw_database_population
                self.append_tasksL(copy.deepcopy(tasksL_))
                self.append_blocksL(copy.deepcopy(blocksL_))
                self.append_pe_mapsL(copy.deepcopy(pe_mapsL_))
                self.append_pe_scheduels(copy.deepcopy(pe_schedulesL_))
                blah = data_movement
                self.souurce_memory_work.update(data_movement['souurce'])
                self.workload_tasks[workload] = [el.task_name for el in tasksL_]
                for el in tasksL_:
                    self.task_workload[el.task_name] = workload

                #self.souurce_memory_work += sum([sum(list(data_movement[task].values())) for task in data_movement.keys() if task == "souurce"])

            self.workloads_last_task = collect_last_task(sw_hw_database_population["workloads"], os.path.join(config.database_data_dir, "parsing"), "misc_database - ")
            self.budgets_dict, self.other_values_dict = collect_budgets(sw_hw_database_population["workloads"], sw_hw_database_population["misc_knobs"], os.path.join(config.database_data_dir, "parsing"),  "misc_database - ")
            if config.heuristic_scaling_study:
                for metric in self.budgets_dict['glass'].keys():
                    if metric == "latency":
                        continue
                    self.budgets_dict['glass'][metric] *= len(sw_hw_database_population["workloads"])

            self.misc_data["same_ip_tasks_list"] = []
        elif sw_hw_database_population["db_mode"] == "generate":
            if len(imported_databases) > 1:
                print("we have to fix the budets_dict collection here. support it and run")
                exit(0)
            self.gen_config = imported_databases[0].gen_config
            self.gen_config['parallel_task_cnt'] = sw_hw_database_population["misc_knobs"]['task_spawn']["parallel_task_cnt"]
            self.gen_config['serial_task_cnt'] = sw_hw_database_population["misc_knobs"]['task_spawn']["serial_task_cnt"]
            self.parallel_task_type = sw_hw_database_population["misc_knobs"]['task_spawn']["parallel_task_type"]
            self.parallel_task_count =self.gen_config['parallel_task_cnt']
            self.serial_task_count =self.gen_config['serial_task_cnt']
            self.datamovement_scaling_ratio = sw_hw_database_population["misc_knobs"]['task_spawn']["boundedness"][1]
            self.memory_boundedness_ratio = sw_hw_database_population["misc_knobs"]['task_spawn']["boundedness"][2]
            self.num_of_hops = sw_hw_database_population["misc_knobs"]['num_of_hops']
            self.budgets_dict = imported_databases[0].budgets_dict
            self.other_values_dict= imported_databases[0].other_values_dict
            self.num_of_NoCs = sw_hw_database_population["misc_knobs"]["num_of_NoCs"]

            other_task_count = 7

            total_task_cnt = other_task_count + max(self.gen_config["parallel_task_cnt"]-1, 0) + self.gen_config["serial_task_cnt"] + max(self.num_of_NoCs -2, 0)

            #intensity_params = ["memory_intensive", 1]
            intensity_params = sw_hw_database_population["misc_knobs"]['task_spawn']["boundedness"]


            tasksL_, data_movement, task_work_dict, parallel_task_names, hoppy_task_names = generate_synthetic_task_graphs_for_asymetric_graphs(total_task_cnt, other_task_count, self.gen_config["parallel_task_cnt"], self.gen_config["serial_task_cnt"], self.parallel_task_type, intensity_params, self.num_of_NoCs)  # memory_intensive, comp_intensive
            blocksL_, pe_mapsL_, pe_schedulesL_ = generate_synthetic_hardware_library(task_work_dict, os.path.join(config.database_data_dir, "parsing"), "misc_database - Block Characteristics.csv")
            self.tasksL.extend(tasksL_)
            self.blocksL.extend(copy.deepcopy(blocksL_))
            self.pe_mapsL.extend(pe_mapsL_)
            self.pe_schedeulesL.extend(pe_schedulesL_)
            for task in data_movement.keys():
                if task == "synthetic_souurce":
                    #self.souurce_memory_work = ["synthetic_souurce"] = sum(list(data_movement[task].values()))
                    self.souurce_memory_work = data_movement[task]

            self.workloads_last_task = {"synthetic" : [taskL.task_name for taskL in tasksL_ if len(taskL.get_children()) == 0][0]}
            self.gen_config["full_potential_tasks_list"] = list(task_work_dict.keys())
            self.misc_data["same_ip_tasks_list"] = []
            self.parallel_task_names = parallel_task_names
            self.hoppy_task_names = hoppy_task_names
            self.workload_tasks[list(sw_hw_database_population["workloads"])[0]] = [el.task_name for el in self.tasksL]
            self.sw_hw_database_population = sw_hw_database_population

            pass
        else:
            print("db_mode:" + sw_hw_database_population["db_mode"] + " is not supported" )
            exit(0)

        # get the hardware graph if need be
        if sw_hw_database_population["hw_graph_mode"] == "parse":
            self.hardware_graph = gen_hardware_graph(os.path.join(config.database_data_dir, "parsing"),
                                                     workload + "_database - ")
            self.task_to_hardware_mapping = gen_task_to_hw_mapping(os.path.join(config.database_data_dir, "parsing"),
                                                            workload + "_database - ")
        else:
            self.hardware_graph = ""
            self.task_to_hardware_mapping = ""

        # set the budget values
        config.souurce_memory_work  = self.souurce_memory_work
        self.SOCsL = []
        self.SOCL0_budget_dict = {"latency": self.budgets_dict["glass"]["latency"], "area":self.budgets_dict["glass"]["area"],
                             "power": self.budgets_dict["glass"]["power"]}  #keys:power, area, latency  example: {"area": area_budget}

        self.SOCL0_other_metrics_dict = {"cost": self.other_values_dict["glass"]["cost"]}
        self.SOCL0 = SOCL("glass", self.SOCL0_budget_dict, self.SOCL0_other_metrics_dict)
        self.SOCsL.append(self.SOCL0)

        #  -----------------
        #  HARDWARE DATABASE
        #  -----------------
        self.porting_effort = {}
        self.porting_effort["arm"] = 1
        self.porting_effort["dsp"] = 10
        self.porting_effort["ip"] = 100
        self.porting_effort["mem"] = .1
        self.porting_effort["ic"] = .1

        df = pd.read_csv(os.path.join(config.database_data_dir, "parsing", "misc_database - Common Hardware.csv"))

        # eval the expression
        def evaluate(value):
            replaced_value_1 = value.replace("^", "**")
            replaced_value_2 = replaced_value_1.replace("=","")
            return eval(replaced_value_2)

        for index, row in df.iterrows():
            temp_dict = row.to_dict()
            self.misc_data[list(temp_dict.values())[0]] = evaluate(list(temp_dict.values())[1])

        self.proj_name = config.proj_name
        # simple models to FARSI_fy the database
        self.misc_data["byte_error_margin"] = 100  # since we use work ratio, we might calculate the bytes wrong (this effect area calculation)
        self.misc_data["area_error_margin"] = 2.1739130434782608e-10
            #self.misc_data["byte_error_margin"]/ self.misc_data["ref_mem_work_over_area"]  # to tolerate the error caused by work_ratio
            #                                                                           # (use byte_error_margin for this calculation)

        arm_clock =[el.clock_freq for el in self.blocksL if el.block_subtype == "gpp"][0]
        self.misc_data["arm_work_over_energy"] = self.misc_data["ref_gpp_dhrystone_value"]/self.misc_data["arm_power_over_clock"]
        self.misc_data["ref_gpp_work_rate"] = self.misc_data["arm_work_rate"] = self.misc_data["ref_gpp_dhrystone_value"] * arm_clock
        self.misc_data["dsp_work_rate"] = self.misc_data["dsp_speed_up_coef"] * self.misc_data["ref_gpp_work_rate"]
        self.misc_data["ip_work_rate"] = self.misc_data["ip_speed_up_coef"]*self.misc_data["ref_gpp_work_rate"]
        self.misc_data["dsp_work_over_energy"] = self.misc_data["dsp_speed_up_coef"] * self.misc_data["ref_gpp_dhrystone_value"] / self.misc_data["dsp_power_over_clock"]
        self.misc_data["ip_work_over_energy"] = 8*self.misc_data["dsp_work_over_energy"]     # multiply by 5 sine we assume only 1/5 th of the instructions are MACs
        self.misc_data["ip_work_over_area"] = 5/(self.misc_data["mac_area"]*self.misc_data["mac_allocated_per_mac_operations"])
        #self.misc_data["ref_ic_work_rate"] = self.misc_data["ref_ic_width"] * self.misc_data["ref_ic_clock"]
        #self.misc_data["ref_mem_work_rate"] = self.misc_data["ref_mem_width"] * self.misc_data["ref_mem_clock"]


    def set_workloads_last_task(self, workloads_last_task):
        self.workloads_last_task = workloads_last_task

    def get_parsed_hardware_graph(self):
        return self.hardware_graph

    def get_parsed_task_to_hw_mapping(self):
        return self.task_to_hardware_mapping

    # setting the budgets dict directly
    # This needs to be done with caution
    def set_budgets_dict_directly(self, budget_dicts):
        self.budgets_dict = budget_dicts
        self.SOCsL = []
        self.SOCL0_budget_dict = {"latency": self.budgets_dict["glass"]["latency"], "area":self.budgets_dict["glass"]["area"],
                             "power": self.budgets_dict["glass"]["power"]}  #keys:power, area, latency  example: {"area": area_budget}

        self.SOCL0_other_metrics_dict = {"cost": self.other_values_dict["glass"]["cost"]}
        self.SOCL0 = SOCL("glass", self.SOCL0_budget_dict, self.SOCL0_other_metrics_dict)
        self.SOCsL.append(self.SOCL0)

    # get the budget values for the SOC
    def get_budget_dict(self, SOC_name):
        for SOC in self.SOCsL:
            if SOC.type  == SOC_name:
                return SOC.get_budget_dict()

        print("SOC_name:" + SOC_name + "not defined")
        exit(0)

    # get non budget values (e.g., cost)
    def get_other_values_dict(self):
        return self.other_values_dict

    # update the budget values
    def update_budget_dict(self, budgets_dict):
        self.budgets_dict = budgets_dict

    def update_other_values_dict(self, other_values_dict):
        self.other_values_dict = other_values_dict

    # set the porting effort (porting effort is a coefficient that determines how hard it is to port the
    #                         task for a non general purpose processor)
    def set_porting_effort_for_block(self, block_type, porting_effort):
        self.porting_effort[block_type] = porting_effort
        assert(self.porting_effort["arm"] < self.porting_effort["dsp"]), "porting effort for arm needs to be smaller than dsp"
        assert(self.porting_effort["dsp"] < self.porting_effort["ip"]), "porting effort for dsp needs to be smaller than ip"
