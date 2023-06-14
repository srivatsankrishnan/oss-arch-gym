
import numpy as np
import copy, random
import os
from subprocess import Popen, PIPE
import pandas as pd
import pickle
m_type_dicts = {1:"CONV", 2:"DSCONV"}
ST_dicts = {0:"K", 1:"C", 2:"X", 3:"Y"}
with open("../utils/order_table.plt", "rb") as fd:
    chkpt = pickle.load(fd)
    order_table = chkpt["table"]
class MaestroEnvironment(object):
    def __init__(self,dimension, num_pe=64, fitness="latency",par_RS=False, l1_size=512, l2_size=108000, NocBW=8192000):
        super(MaestroEnvironment,self).__init__()
        self.dimension = dimension
        self.dimension_dict = {"K": dimension[0], "C": dimension[1], "Y": dimension[2], "X": dimension[3],
                               "R": dimension[4], "S": dimension[5], "T": dimension[6]}
        self.lastcluster_dict = {"K": dimension[0], "C": dimension[1], "Y": dimension[2], "X": dimension[3],
                                 "R": dimension[4], "S": dimension[5], "T": dimension[6]}

        dst_path = "../../cost_model/maestro"
        maestro = dst_path
        self._executable = "{}".format(maestro)
        self.out_repr = set(["K", "C", "R", "S"])
        self.num_pe = num_pe
        self.fitness = fitness
        self.cluster_space = ["K", "C", "Y", "X", "R", "S"] if par_RS else ["K", "C", "Y", "X"]
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.NocBW = NocBW

    def encode(self, proposal, sp2_sz=None):
        answer = []
        sp = ST_dicts[proposal[0]]
        sp_sz = 1
        order = order_table[proposal[1]]
        tile_dict = {}
        space = ["K","C", "Y","X","R","S"]
        for i in range(6):
            tile_dict[space[i]] = proposal[i+2]
        df = []
        for t in order:
            df.append([t, tile_dict[t]])
        answer.append([sp, sp_sz])
        answer.extend(copy.deepcopy(df))
        if len(proposal)> 8:
            sp = ST_dicts[proposal[8]]
            last_sp_size = copy.deepcopy(tile_dict[sp])
            if sp2_sz:
                sp_sz = sp2_sz
            else:
                sp_sz = last_sp_size
            order = order_table[proposal[9]]
            tile_dict = {}
            space = ["K", "C", "Y", "X", "R", "S"]
            for i in range(6):
                tile_dict[space[i]] = proposal[i + 2 + 8]
            df = []
            for t in order:
                df.append([t, tile_dict[t]])
            answer.append([sp, sp_sz])
            answer.extend(copy.deepcopy(df))
        if len(proposal)>16:
            sp = ST_dicts[proposal[16]]
            last_sp_size = copy.deepcopy(tile_dict[sp])
            sp_sz = last_sp_size
            order = order_table[proposal[17]]
            tile_dict = {}
            space = ["K", "C", "Y", "X", "R", "S"]
            for i in range(6):
                tile_dict[space[i]] = proposal[i + 2 + 16]
            df = []
            for t in order:
                df.append([t, tile_dict[t]])
            answer.append([sp, sp_sz])
            answer.extend(copy.deepcopy(df))
        return answer


    def write_maestro(self, indv, layer_id=0, m_file=None):
        m_type = m_type_dicts[int(self.dimension[-1])]
        with open("{}.m".format(m_file), "w") as fo:
            fo.write("Network {} {{\n".format(layer_id))
            fo.write("Layer {} {{\n".format(m_type))
            fo.write("Type: {}\n".format(m_type))
            fo.write("Dimensions {{ K: {:.0f}, C: {:.0f}, Y: {:.0f}, X: {:.0f}, R: {:.0f}, S: {:.0f} }}\n".format(*self.dimension))
            fo.write("Dataflow {\n")
            for k in range(0, len(indv), 7):
                for i in range(k, k+7):
                    d, d_sz = indv[i]
                    if i%7==0:
                        if k != 0:
                            fo.write("Cluster({},P);\n".format(d_sz))
                    else:
                        sp = "SpatialMap" if d == indv[k][0] else "TemporalMap"
                        fo.write("{}({},{}) {};\n".format(sp, d_sz, d_sz, self.get_out_repr(d)))
            fo.write("}\n")
            fo.write("}\n")
            fo.write("}")

    def oberserve_maestro(self, indv, sp2_sz=None):
        m_file = "{}".format(random.randint(0, 2 ** 32))
        en_indv = self.encode(indv,sp2_sz)
        self.write_maestro(en_indv,m_file=m_file)

        # print(num_pe, bw, l1_size)
        os.remove("./{}.csv".format(m_file)) if os.path.exists("./{}.csv".format(m_file)) else None
        command = [self._executable,
                   "--Mapping_file={}.m".format(m_file),
                   "--full_buffer=false", "--noc_bw={}".format(self.NocBW),
                   "--noc_hops=1", "--noc_hop_latency=1",
                   "--noc_mc_support=true", "--num_pes={}".format(self.num_pe),
                   "--num_simd_lanes=1", "--l1_size={}".format(self.l1_size),
                   "--l2_size={}".format(self.l2_size), "--print_res=false", "--print_res_csv_file=true", "--print_log_file=false", "--print_design_space=false", "--msg_print_lv=0"]


        process = Popen(command, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        process.wait()
        try:
            df = pd.read_csv("./{}.csv".format(m_file))
            layer_name = df[" Layer Number"]
            runtime = np.array(df[" Runtime (Cycles)"]).reshape(-1, 1)
            throughput = np.array(df[" Throughput (MACs/Cycle)"]).reshape(-1, 1)
            energy = np.array(df[" Activity count-based Energy (nJ)"]).reshape(-1, 1)
            area = np.array(df[" Area"]).reshape(-1, 1)
            power = np.array(df[" Power"]).reshape(-1, 1)
            l1_size = np.array(df[" L1 SRAM Size (Bytes)"]).reshape(-1, 1)
            l2_size = np.array(df["  L2 SRAM Size (Bytes)"]).reshape(-1, 1)
            mac = np.array(df[" Num MACs"]).reshape(-1, 1)
            os.remove("./{}.csv".format(m_file))  if os.path.exists("./{}.csv".format(m_file)) else None
            os.remove("./log.txt") if os.path.exists("./log.txt") else None
            self.observation = [np.mean(x) for x in [runtime, throughput, energy, area, l1_size, l2_size, mac, power]]
            # penalty = 1 / num_pe + 1 / bw
            # penalty= 1/penalty
            if len(str(stdout))>3:
                return None
            return self.judge()
        except:
            return None

    def judge(self):
        runtime, throughput, energy, area, l1_size, l2_size, mac, power = self.observation
        values = []
        for term in self.fitness:
            if term == "energy":
                reward = -energy
            elif term == "thrpt_ave":
                reward = throughput
            elif term == "EDP":
                reward = -energy * runtime
            elif term == "LAP":
                reward = -area * runtime
            elif term == "EAP":
                reward = -area * energy
            elif term == "thrpt" or term == "thrpt_naive":
                reward = throughput
            elif term == "thrpt_btnk":
                reward = throughput
            elif term == "latency":
                reward = -runtime
            elif term == "area":
                reward = -area
            elif term == "l1_size":
                reward = - l1_size
            elif term == "l2_size":
                reward = -l2_size
            elif term == "power":
                reward = -power
            else:
                raise NameError('Undefined fitness type')
            values.append(reward)
        return values
    def print_indv(self, indv,fd=False):
        for k in range(0, len(indv), 7):
            if fd:
                fd.write("\n{}".format(indv[k:k+7]))
            else:
                print(indv[k:k+7])
    def get_out_repr(self, x):
        if x in self.out_repr:
            return x
        else:
            return x + "'"