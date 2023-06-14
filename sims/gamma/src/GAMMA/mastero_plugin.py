import os
import sys
from subprocess import Popen, PIPE
import numpy as np
import pandas as pd
import random
from math import ceil
import time
print("Import works!!")

class MasterEnv():
    
    def __init__(self, exe_file, mapping_file, noc_bw, offchip_bw, l1_size, l2_size, num_pe):
        
        
        self._executable = exe_file
        self.mapping_file = mapping_file
        self.NocBW = noc_bw
        self.offchipBW = offchip_bw
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.num_pe = num_pe
        self._executable = exe_file
    
    def compute_area_maestro(self, num_pe, l1_size, l2_size):
        MAC_AREA_MAESTRO=4470
        L2BUF_AREA_MAESTRO = 4161.536
        L1BUF_AREA_MAESTRO = 4505.1889
        L2BUF_UNIT = 32768
        L1BUF_UNIT = 64
        area = num_pe * MAC_AREA_MAESTRO + ceil(int(l2_size)/L2BUF_UNIT)*L2BUF_AREA_MAESTRO + ceil(int(l1_size)/L1BUF_UNIT)*L1BUF_AREA_MAESTRO * num_pe
        return area


    def run_maestro(self):

        command = [self._executable,
           "--Mapping_file={}.m".format(self.mapping_file),
           "--full_buffer=false",
           "--noc_bw_cstr={}".format(self.NocBW),
           "--noc_hops=1",
           "--noc_hop_latency=1",
           "--offchip_bw_cstr={}".format(self.offchipBW),
           "--noc_mc_support=true",
           "--num_pes={}".format(int(self.num_pe)),
           "--num_simd_lanes=1",
           "--l1_size_cstr={}".format(self.l1_size),
           "--l2_size_cstr={}".format(self.l2_size),
           "--print_res=false",
           "--print_res_csv_file=true",
           "--print_log_file=false",
           "--print_design_space=false",
           "--msg_print_lv=0"]

        print(command)
        process = Popen(command, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        process.wait() 
        
        # sleep for 5 seconds to make sure the file is written
       
        try:
            df = pd.read_csv("./{}.csv".format(self.mapping_file))
            layer_name = df[" Layer Number"]
            runtime = np.array(df[" Runtime (Cycles)"]).reshape(-1, 1)
            runtime_series = np.array(df[" Runtime (Cycles)"]).reshape(-1, 1)
            throughput = np.array(df[" Throughput (MACs/Cycle)"]).reshape(-1, 1)
            energy = np.array(df[" Activity count-based Energy (nJ)"]).reshape(-1, 1)
            area = np.array(df[" Area"]).reshape(-1, 1)
            power = np.array(df[" Power"]).reshape(-1, 1)
            l1_size = np.array(df[" L1 SRAM Size Req (Bytes)"]).reshape(-1, 1)
            l2_size = np.array(df["  L2 SRAM Size Req (Bytes)"]).reshape(-1, 1)
            l1_size_series = np.array(df[" L1 SRAM Size Req (Bytes)"]).reshape(-1, 1)
            l2_size_series = np.array(df["  L2 SRAM Size Req (Bytes)"]).reshape(-1, 1)
            l1_input_read = np.array(df[" input l1 read"]).reshape(-1, 1)
            l1_input_write = np.array(df[" input l1 write"]).reshape(-1, 1)
            l1_weight_read = np.array(df["filter l1 read"]).reshape(-1, 1)
            l1_weight_write = np.array(df[" filter l1 write"]).reshape(-1, 1)
            l1_output_read = np.array(df["output l1 read"]).reshape(-1, 1)
            l1_output_write = np.array(df[" output l1 write"]).reshape(-1, 1)
            l2_input_read = np.array(df[" input l2 read"]).reshape(-1, 1)
            l2_input_write = np.array(df[" input l2 write"]).reshape(-1, 1)
            l2_weight_read = np.array(df[" filter l2 read"]).reshape(-1, 1)
            l2_weight_write = np.array(df[" filter l2 write"]).reshape(-1, 1)
            l2_output_read = np.array(df[" output l2 read"]).reshape(-1, 1)
            l2_output_write = np.array(df[" output l2 write"]).reshape(-1, 1)
            mac = np.array(df[" Num MACs"]).reshape(-1, 1)
            
            activity_count = {}
            activity_count["l1_input_read"] = l1_input_read
            activity_count["l1_input_write"] = l1_input_write
            activity_count["l1_weight_read"] = l1_weight_read
            activity_count["l1_weight_write"] = l1_weight_write
            activity_count["l1_output_read"] = l1_output_read
            activity_count["l1_output_write"] = l1_output_write
            activity_count["l2_input_read"] = l2_input_read
            activity_count["l2_input_write"] = l2_input_write
            activity_count["l2_weight_read"] = l2_weight_read
            activity_count["l2_weight_write"] = l2_weight_write
            activity_count["l2_output_read"] = l2_output_read
            activity_count["l2_output_write"] = l2_output_write
            activity_count["mac_activity"] = mac
            area = self.compute_area_maestro(self.num_pe, self.l1_size, self.l2_size)
            self.observation = [np.mean(x) for x in [runtime, throughput, energy, area, l1_size, l2_size, mac, power, self.num_pe]]
        except:
            #set all the return values to -1
            runtime = [[1e20]]
            runtime_series = -1
            throughput = -1
            energy = -1
            area = -1
            power = -1
            l1_size = -1    
            l2_size = -1
            l1_size_series = -1
            l2_size_series = -1
            activity_count = -1
            mac = -1
            self.observation = [np.mean(x) for x in [runtime, throughput, energy, area, l1_size, l2_size, mac, power, self.num_pe]]
            print("Error in reading csv file")
        

        return runtime

    def reset(self):
        # if any csv and m files exists then remove the *.csv file and *.m files

        # get the file path
        file_path = os.path.dirname(os.path.realpath(__file__))
        print(file_path)
        results_file = os.path.join(file_path, self.mapping_file+".csv")
        mapping_file = os.path.join(file_path, self.mapping_file+".m")
        
        if os.path.exists(results_file):
            print("csv file exists")
            os.remove(results_file)
        if os.path.exists(mapping_file):
            print("m file exists")
            os.remove(mapping_file)
        

if __name__ == "__main__":
        
        exe_file = "../../cost_model/maestro"
        mapping_file = "1322331445"
        noc_bw = 1073741824
        offchip_bw = 1073741824
        l1_size = 1073741824
        l2_size = 1073741824
        num_pe = 1024
        
        env = MasterEnv(exe_file, mapping_file, noc_bw, offchip_bw, l1_size, l2_size, num_pe)
        runtime, runtime_series, throughput, energy, area, power, l1_size, l2_size, \
        l1_size_series, l2_size_series, activity_count = env.run_maestro()


        print("runtime: ", runtime)
        print("runtime_series: ", runtime_series)
        print("throughput: ", throughput)
        print("energy: ", energy)
        print("area: ", area)
        print("power: ", power)
        print("l1_size: ", l1_size)
        print("l2_size: ", l2_size)
        print("l1_size_series: ", l1_size_series)
        print("l2_size_series: ", l2_size_series)
        print("activity_count: ", activity_count)



  
                