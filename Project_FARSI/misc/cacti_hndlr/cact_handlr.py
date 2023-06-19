#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import os
import csv
import pandas as pd
import math
import numpy as np
import time
import shutil
import subprocess
#from settings import config

# This class at the moment only handls very specific cases,
# concretely, we can provide the size of memory and get the power/area results back.
class CactiHndlr():
    def __init__(self, bin_addr, param_file , cacti_data_log_file, input_col_order, output_col_order):
        self.bin_addr = bin_addr
        self.param_file = param_file
        self.cur_mem_size = 0
        self.input_cfg = ""
        self.output_cfg = ""
        self.cur_mem_type = ""
        self.cur_cell_type = ""
        self.input_col_order = input_col_order
        self.cacti_data_log_file = cacti_data_log_file
        self.output_col_order = output_col_order
        self.cacti_data_container = CactiDataContainer(cacti_data_log_file, input_col_order, output_col_order)

    def set_cur_cell_type(self, cell_type):
        self.cur_cell_type = cell_type

    def set_cur_mem_size(self, cur_mem_size):
        self.cur_mem_size = math.ceil(cur_mem_size)

    def set_cur_mem_type(self, cur_mem_type):
        self.cur_mem_type = cur_mem_type

    def set_params(self):
        param_file_copy_name= "/".join(self.param_file.split("/")[:-1]) + "/" + self.param_file.split("/")[-1] +"_cp"
        #os.system("cp " + self.param_file + " "  + param_file_copy_name)
        shutil.copy(self.param_file, param_file_copy_name)
        time.sleep(.05)
        file1 = open(param_file_copy_name, "a")  # append mode
        size_cmd = "-size (bytes) " + str(self.cur_mem_size)
        cur_mem_type_cmd = "-cache type \"" + self.cur_mem_type + "\""
        cell_type_cmd = "-Data array cell type - \"" + self.cur_cell_type+ "\""
        file1.write(size_cmd + "\n")
        file1.write(cur_mem_type_cmd + "\n")
        file1.write(cell_type_cmd + "\n")
        file1.close()

        self.input_cfg = param_file_copy_name
        self.output_cfg = self.input_cfg +".out"

    def get_config(self):
        return {"mem_size":self.cur_mem_size, "mem_type":self.cur_mem_type, "cell_type:":self.cur_cell_type}

    def run_bin(self):
        bin_dir = "/".join(self.bin_addr.split("/")[:-1])
        os.chdir(bin_dir)
        #cmd = self.bin_addr + " " + "-infile " + self.input_cfg
        subprocess.call([self.bin_addr, "-infile", self.input_cfg])
        #os.system(cmd)

    def run_cacti(self):
        self.set_params()
        self.run_bin()

    def reset_cfgs(self):
        self.input_cfg = ""
        self.output_cfg = ""

    def parse_and_find(self, kwords):
        results_dict = {}
        ctr =0
        while not os.path.isfile(self.output_cfg) and ctr < 60:
            time.sleep(1)
            ctr +=1

        f = open(self.output_cfg)
        reader = csv.DictReader(f)
        dict_list = []
        for line in reader:
            dict_list.append(line)

        for kw in kwords:
            results_dict [kw] = []

        for dict_ in dict_list:
            for kw in results_dict.keys():
                for key in dict_.keys():
                    if key == " " +kw:
                        results_dict[kw] = dict_[key]

        f.close()
        return results_dict

    def collect_cati_data(self):
        self.run_cacti()
        results = self.parse_and_find(["Dynamic read energy (nJ)", "Dynamic write energy (nJ)", "Area (mm2)"])
        os.system("rm " + self.output_cfg)
        return results



class CactiDataContainer():
    def __init__(self, cached_data_file_addr, input_col_order, output_col_order):
        self.cached_data_file_addr = cached_data_file_addr
        self.input_col_order = input_col_order
        self.output_col_order = output_col_order
        self.prase_cached_data()

    def prase_cached_data(self):
        # create the file if doesn't exist
        if not os.path.exists(self.cached_data_file_addr):
            file = open(self.cached_data_file_addr, "w")
            for col_val in (self.input_col_order + self.output_col_order)[:-1]:
                file.write(str(col_val)+ ",")
            file.write(str((self.input_col_order + self.output_col_order)[-1])+"\n")
            file.close()

        # populate the pand data frames with it
        try:
            self.df = pd.read_csv(self.cached_data_file_addr)
        except Exception as e:
            if e.__class__.__name__ in "pandas.errors.EmptyDataError":
                self.df = pd.DataFrame(columns=self.input_col_order + self.output_col_order)
                #self.df =

    def find(self, KVs):
        df_ = self.df
        for k,v in KVs:
            df_temp = self.find_one_kv(df_, (k,v))
            if isinstance(df_temp, bool)  and df_temp == False:  # if can't be found
                return False, "_", "_", "_"
            elif df_temp.empty:
                return False, "_", "_", "_"
            df_ =  df_temp

        if len(df_.index) > 1:  # number of rows >1 means more than one equal value
            print("can not have duplicated values ")
            exit(0)

        output = [True] + [df_.iloc[0][col_name] for col_name in self.output_col_order]
        return output

    def find_one_kv(self, df, kv):
        if df.empty:
            return False

        k = kv[0]
        v = kv[1]
        result = df.loc[(df[k] == v)]
        return result

    def insert(self, key_values_):
        # if data exists, return
        if not self.df.empty:
            if self.find(key_values_)[0]:
                return

        # append the output file
        # make the output file if doesn't exist
        if not os.path.exists(self.cached_data_file_addr):
            file = open(self.cached_data_file_addr, "w")
            for col_val in self.df.columns[:-1]:
                file.write(self.df.columns + ",")
            file.write(self.df.columns[-1]+  "\n")
            file.close()

        values_ = [kv[1] for kv in key_values_]
        # add it to the pandas
        df2 = pd.DataFrame([values_], columns=self.input_col_order + self.output_col_order)
        self.df = self.df.append(df2, ignore_index=True)

        # append results to the file
        with open(self.cached_data_file_addr, "a") as output:
            for key, value in key_values_[:-1]:
                output.write(str(value) +",")
            output.write(str(values_[-1]) + "\n")


# just a test case
if __name__ == "__main__":
    cact_bin_addr = "/Users/behzadboro/Downloads/cacti/cacti"
    cacti_param_addr = "/Users/behzadboro/Downloads/cacti/farsi_gen.cfg"
    cacti_data_log_file= "/Users/behzadboro/Downloads/cacti/data_log.csv"

    cur_mem_size = 320000000
    cur_mem_type = "main memory"   # ["main memory", "ram"]
    input_col_order = ("mem_subtype", "mem_size")
    output_col_order = ("energy_per_byte", "area")
    cacti_hndlr = CactiHndlr(cact_bin_addr, cacti_param_addr, cacti_data_log_file, input_col_order, output_col_order)
    cacti_hndlr.set_cur_mem_size(cur_mem_size)
    cacti_hndlr.set_cur_mem_type(cur_mem_type)
    area_power_results = cacti_hndlr.collect_cati_data()
    print(area_power_results)
