import numpy as np
import copy, random
import os
import sys
from subprocess import Popen, PIPE
import pandas as pd
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from functools import reduce
from collections import defaultdict
from math import ceil
m_type_dicts = {0:"CONV", 1:"CONV", 2:"DSCONV", 3:"CONV"}
CONVtype_dicts = {0:"FC", 1:"CONV",2:"DSCONV", 3:"GEMM"}
MAC_AREA_MAESTRO=4470
MAC_AREA_INT8=282
DEVELOP_MODE=False

class GAMMA(object):
    def __init__(self,dimension, map_cstr=None, num_pe=64, pe_limit=1024, 
                fitness="latency", constraints=dict(), par_RS=False, l1_size=512,
                l2_size=108000, NocBW=81920000, offchipBW=81920000, slevel_min=2,
                slevel_max=2, fixedCluster=0, log_level=2,constraint_class=None,
                external_mem_cstr=None, use_factor=False,uni_base=True,
                use_reorder=True, use_growing=True, use_aging=True,
                reorder_alpha=0.5, growing_alpha=0.5, aging_alpha=0.5
                ):
        super(GAMMA,self).__init__()
        self.dimension = dimension
        self.dimension_dict = {"K":dimension[0], "C":dimension[1], "Y":dimension[2], "X":dimension[3], "R":dimension[4],"S":dimension[5], "T":dimension[6]}
        self.lastcluster_dict = {"K":dimension[0], "C":dimension[1], "Y":dimension[2], "X":dimension[3], "R":dimension[4],"S":dimension[5], "T":dimension[6]}
        if DEVELOP_MODE:
            path = "/usr/scratch/felix/my_code/HW_optimizer_rnn_result_history/maestro_his/maestro21/"
            if os.path.exists(path) is False:
                path = "/Users/chuchu/Documents/gt_local/HW_optimizer_rnn_result_history/maestro_his/maestro21/"
            # maestro = path + "maestro21"
            # maestro = path + "maestro21_noRScstr"
            maestro = path + "maestro22_noRScstr"
        else:
            maestro = "../../cost_model/maestro"

        self._executable = "{}".format(maestro)
        self.out_repr = set(["K", "C", "R", "S"])
        self.num_pe = num_pe
        self.pe_limit = pe_limit
        self.fitness_objective = fitness
        self.cluster_space = ["K", "C", "Y","X","R","S"] if par_RS else ["K", "C", "Y","X"]
        self.l1_size = l1_size if l1_size > 0 else 2**30
        self.l2_size = l2_size if l2_size > 0 else 2**30
        self.NocBW = NocBW if NocBW>0 else 2**30
        self.offchipBW = offchipBW if offchipBW > 0 else 2**30
        self.slevel_min = slevel_min
        self.slevel_max = slevel_max
        self.fixedCluster = fixedCluster
        self.log_level = log_level
        self.map_cstr = map_cstr
        self.num_free_order = None
        self.num_free_par = None
        self.cstr_list = None
        self.stat = None
        self.dimension_factors = self.get_dimension_factors(self.dimension_dict)
        self.use_ranking = True if self.fitness_objective[0] == "ranking" else False
        self.constraints=constraints
        self.constraint_class=constraint_class
        self.external_mem_cstr = external_mem_cstr
        self.use_factor = use_factor
        self.uni_base = uni_base
        self.L1_bias_template = None
        self.area_pebuf_only=False
        self.external_area_model = False

        # add flags to select or not select the type of operations
        self.use_reorder = use_reorder
        self.use_growing = use_growing
        self.use_aging = use_aging
        self.reorder_alpha = reorder_alpha
        self.growing_alpha = growing_alpha
        self.aging_alpha = aging_alpha

    def reset_hw_parm(self, l1_size=None, l2_size=None, num_pe=None, NocBW=None, map_cstr=None, pe_limit=None,area_pebuf_only=None, external_area_model=None, offchipBW=None):
        if l1_size:
            self.l1_size=l1_size if l1_size > 0 else 2**30
        if l2_size:
            self.l2_size=l2_size if l2_size > 0 else 2**30
        if num_pe:
            self.num_pe=num_pe
        if NocBW:
            self.NocBW=NocBW if NocBW > 0 else 2**30
        if offchipBW:
            self.offchipBW=offchipBW if offchipBW > 0 else 2**30
        if map_cstr:
            self.map_cstr = map_cstr
        if pe_limit:
            self.pe_limit = pe_limit
        if area_pebuf_only:
            self.area_pebuf_only = area_pebuf_only
        if external_area_model:
            self.external_area_model = external_area_model

    def get_dimension_factors(self, dimension_dict):
        dimension_factors = dict()
        for key, value in dimension_dict.items():
            if key != "T":
                factors = self.get_factors(value)
                dimension_factors[key] = {"set":factors, "array":np.array(list(factors))}
        return dimension_factors

    def reset_dimension(self, dimension=None, fitness=None, constraints=None, constraint_class=None, external_mem_cstr=None):
        if dimension is not None:
            self.dimension = dimension
        if fitness is not None:
            self.fitness_objective =  fitness
        if constraints is not None:
            self.constraints = constraints
        if constraint_class is not None:
            self.constraint_class = constraint_class
        if external_mem_cstr is not None:
            self.external_mem_cstr = external_mem_cstr
        self.use_ranking = True if self.fitness_objective[0] == "ranking" else False
        self.dimension_dict = {"K": self.dimension[0], "C": self.dimension[1], "Y": self.dimension[2], "X": self.dimension[3], "R": self.dimension[4],"S": self.dimension[5], "T": self.dimension[6]}
        self.dimension_factors = self.get_dimension_factors(self.dimension_dict)

    def create_genome_with_cstr(self):
        indv = self.create_genome()
        print('[Debug][Create Genome with Cstr][slevel_min]', self.slevel_min)
        for _ in range(self.slevel_min - 1):
            indv = self.born_cluster_ind(indv)
        self.map_cstr.create_from_constraint(indv, self.fixedCluster, self.dimension_dict)
        print("[Debug][Create Genome with Cstr][indv]", indv)
        return indv

    def create_genome(self, uni_base=False,last_cluster_dict=None, l1_bias_template=None):
        print('[Debug][Create Genome][uni_base]', uni_base)
        print('[Debug][Create Genome][last_cluster_dict]', last_cluster_dict)
        print('[Debug][Create Genome][l1_bias_template]', l1_bias_template)
        print('[Debug][Create Genome][use_factor]]', self.use_factor)
        print('[Debug][Create Genome][self.dimension_dict]', self.dimension_dict)
        print('[Debug][Create Genome][self.dimension]', self.dimension)
        print('[Debug][Create Genome][self.dimension_factors]', self.dimension_factors)
        print('[Debug][Create Genome][self.cluster_space]', self.cluster_space)
        print('[Debug][Create Genome][pe_limit]', self.pe_limit)
        
        
        if uni_base:
            if l1_bias_template:
                K, C, Y, X, R, S = l1_bias_template
            else:
                K,C,Y,X,R,S,T = [1]*len(self.dimension)
                
        else:
            K,C,Y,X,R,S,T = self.dimension
            print('[Debug][Create Genome][self.dimension]', self.dimension)
            print('[Debug][Create Genome][K,C,Y,X,R,S,T]', K,C,Y,X,R,S,T)

        if uni_base is False and last_cluster_dict:
            K = last_cluster_dict["K"]
            C = last_cluster_dict["C"]
            Y = last_cluster_dict["Y"]
            X = last_cluster_dict["X"]
            R = last_cluster_dict["R"]
            S = last_cluster_dict["S"]
        sp = random.choice(self.cluster_space)
        print('[Debug][Create Genome][sp]', sp)
        
        lastcluster_sz = last_cluster_dict[sp] if last_cluster_dict else self.dimension_dict[sp]
        if uni_base == True:
            if self.fixedCluster>0:
                sp_sz = self.fixedCluster
            else:
                if self.num_pe > 0:
                    sp_sz = random.randint(1, min(lastcluster_sz, self.num_pe))
                else:
                    sp_sz = random.randint(1, lastcluster_sz)
        else:
            sp_sz = random.randint(1, self.num_pe if self.num_pe > 0 else self.pe_limit)
            print("[DEBUG][Create Genome][sp_sz]", sp_sz)
            
        if self.use_factor and not uni_base:
            df = [["K", np.random.choice(self.dimension_factors["K"]["array"])],
                ["C",np.random.choice(self.dimension_factors["C"]["array"])],
                ["Y", np.random.choice(self.dimension_factors["Y"]["array"])],
                ["X", np.random.choice(self.dimension_factors["X"]["array"])],
                ["R",np.random.choice(self.dimension_factors["R"]["array"])], 
                ["S",np.random.choice(self.dimension_factors["S"]["array"])]]
        else:
            if uni_base:
                df = [["K", K], ["C", C], ["Y", Y],["X", X], ["R", R], ["S", S]]  
            else:
                df = [["K", random.randint(1, K)],
                ["C", random.randint(1, C)],
                ["Y", random.randint(1, Y)],
                ["X", random.randint(1, X)], 
                ["R", random.randint(1, R)], 
                ["S", random.randint(1, S)]]
        
        idx = np.random.permutation(len(df))
        print("[DEBUG][Create Genome][idx]", idx)
        indv = [[sp, sp_sz]] + [df[i] for i in idx]
        print("[DEBUG][Create Genome][indv]", indv)
        return indv

    def search_loc(self, segment_of_indv, dim):
        for i in range(len(segment_of_indv)):
            if segment_of_indv[i][0]==dim:
                return i

    def validTo_external_mem_cstr(self, indv,num_pe=1024):
        if not self.external_mem_cstr:
            return True
        mem_used = self.compute_l1_l2_mem_size(indv,num_pe=num_pe)
        for key, value in self.external_mem_cstr.items():
            if mem_used[key]> value:
                return False
        return True

    def compute_l1_l2_mem_size(self, indv, num_pe=1024):
        mem = {}
        def get_w_i_o_size(picks, level=1, num_pe=1024):
            if level==2:
                sp_dim_L2 = indv[0][0]
                sp_dim_size_L2 = picks[sp_dim_L2]
                dim = self.dimension_dict[sp_dim_L2]
                sp_sz = indv[7][1]
                num_cluster = num_pe//sp_sz
                needed_iters = ceil(dim/sp_dim_size_L2)
                actual_sp_tile_size = min(dim, sp_dim_size_L2 * min(needed_iters, num_cluster))
                picks[sp_dim_L2] = actual_sp_tile_size

            weight = picks["K"] * picks["C"] * picks["R"] * picks["S"]
            input = picks["C"] * picks["Y"] * picks["X"]
            output = picks["K"] * picks["Y"] * picks["X"]
            return weight, input, output
        weight, input, output = get_w_i_o_size(picks=self.scan_indv(indv[0:7]), level=2,num_pe=num_pe)
        mem[f"L2-W"] = weight
        mem[f"L2-I"] = input
        mem[f"L2-O"] = output
        mem[f"L2-soft"] = output + input + weight
        weight, input, output = get_w_i_o_size(picks=self.scan_indv(indv[7:14]), level=1)
        mem[f"L1-W"] = weight
        mem[f"L1-I"] = input
        mem[f"L1-O"] = output
        mem[f"L1-soft"] = output + input + weight
        return mem

    def biased_init(self, indv, bias = None):
        if bias is None:
            return indv
        if "par" in bias:
            for key, value in bias["par"].items():
                pointer = (key-1) * 7
                indv[pointer][0] = value
        if "order" in bias:
            for key, value in bias["order"].items():
                st, end = (key-1)*7+1, (key)*7
                temp_indv = copy.deepcopy(indv[st: end])
                for di in value[::-1]:
                    idx = self.search_loc(temp_indv, di)
                    item = temp_indv.pop(idx)
                    temp_indv.insert(0, item)
                indv[st: end] = temp_indv
        if "tiles" in bias:
            for key, value in bias["tiles"].items():
                st, end = (key-1)*7+1, (key)*7
                temp_indv = copy.deepcopy(indv[st: end])
                if key == 1:
                    last_cluster_dict = self.dimension_dict
                else:
                    last_cluster_dict = self.scan_indv(indv[:7])
                for i in range(len(temp_indv)):
                    dim = temp_indv[i][0]
                    if dim in value:
                        new_tile = max(1, int(last_cluster_dict[dim]* value[dim]))
                        temp_indv[i][1] = new_tile
                indv[st: end] = temp_indv
        return indv

    def create_genome_fixedSL(self,  bias = None):
        if self.map_cstr:
            return self.create_genome_with_cstr()
        ind = self.create_genome()
        for _ in range(self.slevel_min-1):
            ind = self.born_cluster_ind(ind)
        if bias:
            ind = self.biased_init(ind, bias=bias)
        return ind

    def select_parents(self, pop, fitness, num_parents, num_population):
        #=====sel unique======================
        pop_set = set()
        to_saved_idx = []
        for i in range(len(pop)):
            cur_cand = tuple([tt for i, t in enumerate(pop[i]) for j, tt in enumerate(t)  if (i, j) != (0, 1)])
            if cur_cand not in pop_set:
                pop_set.add(cur_cand)
                to_saved_idx.append(i)
        fitness = fitness[to_saved_idx]
        pop = [pop[i] for i in range(len(pop)) if i in set(to_saved_idx)]
        # print(f'Unique pop: {len(to_saved_idx)}')
        #=====================================

        if self.normalize:
            norm_fitness = fitness/np.abs(np.nanmean(np.ma.masked_equal(fitness, value=float("-Inf")), axis=0))
            fitness_list = [tuple([-np.prod(ar[1:]), -i]) for i, ar in enumerate(norm_fitness)]
        else:
            fitness_list = [tuple(list(ar)+[-i]) for i, ar in enumerate(fitness)]
        fitness_list = sorted(fitness_list, reverse=True)
        idx = [int(-ar[-1]) for ar in fitness_list]
        new_pop = [pop[i] for i in idx][:num_population]
        new_fitness = fitness[idx][:num_population]
        parents = copy.deepcopy(new_pop[:num_parents])
        if self.use_pleteau:
            num_pletau = self.build_pleteau(fitness, pop)
            # print(f"Num pleteau: {num_pletau}")
            fitness_list = [tuple([*ar[:len(self.fitness_objective)], *ar]) for i, ar in enumerate(self.pleteau_sol.keys())]
            fitness_list = sorted(fitness_list, reverse=True)
            idx = [tuple(ar[-len(self.fitness_objective):]) for ar in fitness_list]

            new_pop[num_pletau:] = new_pop[:-num_pletau]
            new_pop[:num_pletau] =[self.pleteau_sol[i] for i in idx]
            new_fitness[num_pletau:] = new_fitness[:-num_pletau]
            new_fitness[:num_pletau] =[i for i in idx]
            parents = copy.deepcopy(new_pop[:num_parents+num_pletau])
            self.best_reward_pleteau = copy.deepcopy(new_fitness[:num_pletau])
            self.best_sol_pleteau = copy.deepcopy(new_pop[:num_pletau])
        return new_pop, new_fitness, parents

    def mutate_par(self, pop,alpha=0.5):
        if self.map_cstr is not None:
            return
        for idx in range(len(pop)):
            if random.random() < alpha:
                # if self.map_cstr is not None:
                #     avail_val = self.num_free_par + self.num_free_order - 1
                # else:
                #     avail_val = len(indv) - 1
                    # ##===ad hoc trial=========
                pop[idx][7][0], pop[idx][0][0] = pop[idx][0][0], pop[idx][7][0]
                continue
                    # #=========================
                pick = random.randint(0, avail_val)
                pick_level = pick//7
                pick = int(pick_level *7)
                if  self.map_cstr  and "sp" in self.cstr_list[pick_level]:
                    choices = self.cstr_list[pick_level]["sp"]
                else:
                    choices = self.cluster_space
                sp = random.choice(choices)
                if self.map_cstr  and "sp_sz" in self.cstr_list[pick_level]:
                    sp_sz = self.self.cstr_list[pick_level]["sp_sz"]
                else:
                    if self.fixedCluster < 1:
                        last_cluster_dict = self.scan_indv(indv[:-7]) if pick != 0 else None
                        lastcluster_sz = last_cluster_dict[sp] if last_cluster_dict else self.dimension_dict[sp]
                        sp_sz = random.randint(1, min(lastcluster_sz, self.num_pe))
                    else:
                        sp_sz = self.fixedCluster
                pop[idx][pick] = [sp, sp_sz]

    def get_factors(self, n):
        return set(reduce(list.__add__,
                          ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))

    def mutate_tile(self, pop, is_finetune=False, num_mu_loc=1, alpha=0.5, range_alpha=0.5, cluster_only=False):
        for idx in range(len(pop)):
            indv = pop[idx]
            for _ in range(num_mu_loc):
                if random.random() < alpha:
                    if self.map_cstr:
                        num_free_tile = self.cstr_list[1]["num_free_tile"]
                        if num_free_tile==0:
                            pick = random.randint(0, len(indv) - 6 - 1)
                        else:
                            pick = random.randint(0, len(indv) - 1)
                    else:
                        pick = random.randint(0, len(indv) - 1)
                    if cluster_only:
                        pick = 7
                    if pick % 7 == 0:
                        if  self.map_cstr  and "sp" in self.cstr_list[pick // 7]:
                            choices = self.cstr_list[pick // 7]["sp"]
                        else:
                            choices = self.cluster_space
                        sp = random.choice(choices)
                        if pick>0:
                            if self.map_cstr  and "sp_sz" in self.cstr_list[pick // 7]:
                                sp_sz = self.cstr_list[pick // 7]["sp_sz"]
                            else:
                                if self.fixedCluster < 1:
                                    last_cluster_dict = self.scan_indv(indv[:-7]) if pick != 0 else None
                                    lastcluster_sz = last_cluster_dict[sp] if last_cluster_dict else self.dimension_dict[sp]
                                    if self.num_pe > 0:
                                        # sp_sz = max(1, random.randint(0, min(lastcluster_sz, self.num_pe)))
                                        sp_sz = max(1, random.choice(list(self.get_factors(min(lastcluster_sz, self.num_pe)))))
                                    else:
                                        # sp_sz = max(1, random.randint(0, min(lastcluster_sz, indv[0][1])))
                                        sp_sz = max(1, random.choice(list(self.get_factors(min(lastcluster_sz, indv[0][1])))))
                                else:
                                    sp_sz = self.fixedCluster
                        else:
                            sp_sz =pop[idx][pick][1]
                        pop[idx][pick] = [sp, sp_sz]
                    else:
                        d, d_sz = indv[pick]
                        if pick > 7:
                            last_cluster_dict = self.scan_indv(indv[:-7])
                            thr = last_cluster_dict[d]
                            if self.use_factor is False:
                                new_d_sz = random.randint(1, thr)
                            else:
                                choices = self.get_factors(thr)
                                new_d_sz = np.random.choice(list(choices))

                        else:
                            if self.use_factor is False:
                                thr = self.dimension_dict[d]
                                new_d_sz = random.randint(1, thr)
                            else:
                                new_d_sz = np.random.choice(self.dimension_factors[d]["array"])
                        if is_finetune:
                            sampling = np.random.uniform(-range_alpha, range_alpha, 1)
                            sampling = int(sampling * thr)
                            new_d_sz = d_sz + sampling
                            new_d_sz = max(1, min(new_d_sz, self.dimension_dict[d]))
                        pop[idx][pick][1] = new_d_sz

    def mutate_pe(self, pop, alpha=0.5, mutate_range_ratio=0.5):
        for idx in range(len(pop)):
            if len(pop[idx])<=7:
                if random.random() < alpha:
                    pop[idx][0][1] = random.randint(1, self.pe_limit)
            else:
                sp , sp_sz, *a = pop[idx][7]
                cur_multiplier = pop[idx][0][1]//sp_sz
                if random.random()< alpha:
                    if self.use_factor is False:
                        #==method 1
                        last_cluster_dict = self.scan_indv(pop[idx][:7])
                        last_cluster_dict_sz = last_cluster_dict[sp]
                        max_multiplier = max(1, self.pe_limit // sp_sz)
                        cur_multiplier = random.randint(1, min(max_multiplier, ceil(self.dimension_dict[sp]/last_cluster_dict_sz)))
                        # ====constrained to smaller search space====
                        max_value = min(max_multiplier, ceil(self.dimension_dict[sp]/last_cluster_dict_sz))
                        cur_multiplier = random.randint(max(1, int(max_value*mutate_range_ratio)), max_value)
                        #============================================
                    else:
                        #method 2
                        factors = self.dimension_factors[sp]["array"]
                        max_multiplier = max(1, self.pe_limit // sp_sz)
                        factors = factors[(factors<= max_multiplier)]
                        cur_multiplier = random.choice(factors)
                        # ====constrained to smaller search space====
                        cur_multiplier =  random.choice(factors[int(len(factors)*mutate_range_ratio):])
                        #============================================
                cur_pe = min(self.pe_limit, cur_multiplier * sp_sz)
                pop[idx][0][1] = cur_pe
                # pop[idx][7][1] = sp_sz

    def swap_order(self, pop, alpha=0.5):
        max_count = len(pop)
        if self.num_free_order == 0:
            return
        while max_count > 0:
            max_count -= 1
            if random.random()< alpha:
                idx = random.randint(0, len(pop) - 1)
                if self.map_cstr is None:
                    sel_cluster = random.randint(0, (len(pop[idx])-1)//7)
                    swap_id = np.random.randint(1, 6+1, (2,)) + sel_cluster * 7
                else:
                    sel_cluster = random.randint(0, (self.num_free_order-1)//6)
                    num_free_order = (self.num_free_order - sel_cluster*6 -1)%6
                    swap_id = np.random.randint(1, 1+num_free_order+1, (2,)) + sel_cluster * 7
                pop[idx][swap_id[0]], pop[idx][swap_id[1]] = pop[idx][swap_id[1]], pop[idx][swap_id[0]]

    def crossover_tile(self, parents, pop, alpha=0.5):
        if len(parents) ==1:
            for idx in range(len(pop)):
                pop[idx] = copy.deepcopy(parents[0])
        else:
            for idx in range(0,len(pop),2):
                pick_range = np.random.permutation(np.arange(0, len(parents)))
                dad, mom = parents[pick_range[0]], parents[pick_range[1]]
                # dad, mom = parents[random.randint(0, len(parents)-1)], parents[random.randint(0, len(parents)-1)]
                dad = copy.deepcopy(dad)
                mom = copy.deepcopy(mom)
                length = min(len(dad), len(mom))
                if random.random() < alpha:
                    cross_point = random.choice(["K", "C", "Y", "X", "R", "S"])
                    for k in range(0, length, 7):
                        for i in range(k+1, k+7):
                            d, d_sz = dad[i]
                            if d== cross_point:
                                dad_sz = d_sz
                                dad_idx = i
                            d, d_sz = mom[i]
                            if d == cross_point:
                                mom_sz = d_sz
                                mom_idx = i
                        dad[dad_idx][1] = mom_sz
                        mom[mom_idx][1] = dad_sz
                pop[idx] = dad
                if idx + 1 < len(pop):
                    pop[idx+1] = mom

    def check_tile_dependency(self,  pop):
        for idx in range(0, len(pop)):
            cur_pop = pop[idx]
            last_cluster =self.scan_indv(cur_pop)
            first_cluster = self.scan_indv(cur_pop[:7])
            for key in ["K", "C", "Y", "X", "R", "S"]:
                if last_cluster[key] > first_cluster[key]:
                    print("Error", cur_pop)

    def correctify_tile_dependency(self, pop):
        for i in range(0, len(pop)):
            ind = pop[i]
            cur_cluster = None
            levels = len(ind)//7
            for i in range(levels):
                last_cluster = copy.deepcopy(cur_cluster)
                cur_cluster = self.scan_indv(ind[7*i:7*(i+1)])
                if i == 0:
                    continue
                else:
                    for idx in range(7*i+1, 7*(i+1)):
                        d, d_sz = ind[idx]
                        d_sz = min(last_cluster[d], d_sz)
                        ind[idx][1] = d_sz

    def correctify_tile_dependency_thread(self, indv):
        ind = copy.deepcopy(indv)
        cur_cluster = None
        levels = len(ind)//7
        for i in range(levels):
            last_cluster = copy.deepcopy(cur_cluster)
            cur_cluster = self.scan_indv(ind[7*i:7*(i+1)])
            if i == 0:
                continue
            else:
                for idx in range(7*i+1, 7*(i+1)):
                    d, d_sz = ind[idx]
                    d_sz = min(last_cluster[d], d_sz)
                    ind[idx][1] = d_sz
        return ind

    def born_cluster_ind(self, ind):
        print("[DEBUG][born_cluster_ind][slevel_max]", self.slevel_max)
        if (len(ind)) // 7 < self.slevel_max:
            last_cluster_dict = self.scan_indv(ind)
            new_ind = ind + self.create_genome(uni_base=self.uni_base, l1_bias_template=self.L1_bias_template, last_cluster_dict=last_cluster_dict)
            print("[DEBUG][born_cluster_ind][len(ind)]",new_ind)
            
            ind = new_ind
        return ind

    def born_cluster(self, pop, alpha=0.1):
        max_count = len(pop)
        while max_count > 0:
            max_count -= 1
            if random.random() < alpha:
                idx = random.randint(0, len(pop) - 1)
                ind = self.born_cluster_ind(pop[idx])
                pop[idx] = ind

    def kill_cluster(self, pop, alpha=0.5):
        max_count = len(pop)
        while max_count > 0:
            max_count -= 1
            if random.random() < alpha:
                idx = random.randint(0, len(pop) - 1)
                if (len(pop[idx]))//7>self.slevel_min:
                    pop[idx] = pop[idx][:-7]

    def scan_indv(self,indv):
        last_cluster_dict=defaultdict(str)
        for i in range(len(indv)-6,len(indv), 1):
            d, d_sz = indv[i]
            last_cluster_dict[d] = d_sz
        return  last_cluster_dict

    def get_out_repr(self, x):
        if x in self.out_repr:
            return x
        else:
            return x + "'"

    def comform_to_cstr(self, pop):
        if self.map_cstr is None:
            return
        for indv in pop:
            self.map_cstr.impose_constraint(indv, fixed_sp_sz=self.fixedCluster)

    def get_fitness(self):
        return self.fitness

    def create_unit_base_pops(self, population, num_all_unit=None):
        if num_all_unit is None:
            num_all_unit = len(population)
        for idx in range(num_all_unit):
            for level in range(len(population[0]) // 7):
                for i in range(1, 7):
                    population[idx][i + level * 7][1] = 1

    def reinit_pop(self,pool, num_population,  stage_idx, best_sol_1st, init_pop, cur_gen=-1, bias= None, num_all_unit=2):
        population = [self.create_genome_fixedSL(bias=bias) for _ in range(num_population)]
        #====always create a base unit pop=======
        self.create_unit_base_pops(population, num_all_unit=num_all_unit)

        #========================================
        if init_pop is not None:
            # population = [self.create_genome_fixedSL() for _ in range(num_population)] if best_sol_1st is None else [best_sol_1st for _ in range(num_population)]
            population[:10] = init_pop[:10]
            # population = init_pop
        else:
            # population = [self.create_genome_fixedSL(bias=bias) for _ in range(num_population)] if best_sol_1st is None else [best_sol_1st for _ in range(num_population)]
            if best_sol_1st is not None:
                population[0] = best_sol_1st
        self.num_parents = num_population
        self.comform_to_cstr(population)
        self.fitness = np.ones((max(num_population, len(population)), len(self.fitness_objective)), float)
        self.evaluate(pool=pool, population=population,cur_gen=cur_gen)
        return population



    def cal_statstics(self):
        fitness = np.array(self.fitness)
        reward = fitness[:,0]
        sel_valid = reward>float("-Inf")
        latency_ave =  np.mean(-fitness[sel_valid, 0])
        area_ave =  np.mean(-fitness[sel_valid, 1])
        l1_size_ave = np.mean(-fitness[sel_valid, 2])
        l2_size_ave = np.mean(-fitness[sel_valid, 3])
        # l1_size_pops = -np.array(self.l1_size_pop)
        # l2_size_pops = np.array(self.l2_size_pop)
        # l1_size_pops = -l1_size_pops[l1_size_pops>float("-Inf")]
        # l2_size_pops = -l2_size_pops[l2_size_pops>float("-Inf")]
        statstics = {
            "latency_ave":latency_ave,
            "area_ave":area_ave,
            "l1_size_ave":l1_size_ave,
            "l2_size_ave":l2_size_ave
        }
        self.stat = statstics
        return statstics

    def cal_pletau_stat(self):
        fitness = np.array(list(self.pleteau_sol.keys()))
        fitness = np.mean(fitness, axis=0)
        stats = {
            "fitness":fitness,
            "Reward": fitness[0],
            "latency": fitness[1],
            "area": fitness[2],
            "l1_size": fitness[3],
            "l2_size": fitness[4]
        }
        return stats

    def build_pleteau(self,fitness, population):
        self.pleteau_sol = dict()
        for cand_fit, cand_sol in zip(fitness, population):
            self.insert_into_pleteau(cand_fit, cand_sol)
        return len(self.pleteau_sol)

    def insert_into_pleteau(self, cand_fit, cand_sol):
        reject = False
        if np.prod(cand_fit>float("-inf"))!=1:
            return
        cand_fit = tuple(list(cand_fit))
        for pl in set(self.pleteau_sol.keys()):
            if all([cand_fit[i]< pl[i] for i in range(len(cand_fit))]):
                del self.pleteau_sol[pl]
                self.pleteau_sol[cand_fit] = cand_sol
            elif all([cand_fit[i]> pl[i] for i in range(len(cand_fit))]):
                reject = True
        if not reject:
            self.pleteau_sol[cand_fit] = cand_sol

    def adjust_fitness(self, fitness):
        fitness_list = [(ar[1], -i) for i, ar in enumerate(fitness)]
        fitness_list = sorted(fitness_list, reverse=True)
        idx = np.array([int(-ar[-1]) for ar in fitness_list])
        rank1 = np.zeros((len(idx),))
        rank1[idx] = -np.arange(len(idx))
        fitness_list = [(ar[2], -i) for i, ar in enumerate(fitness)]
        fitness_list = sorted(fitness_list, reverse=True)
        idx = np.array([int(-ar[-1]) for ar in fitness_list])
        rank2 = np.zeros((len(idx),))
        rank2[idx] = -np.arange(len(idx))
        rank = rank1 + rank2
        fitness[:,0] = rank
        gen_best_idx = np.argmax(fitness[:,0])
        return fitness, gen_best_idx

    def evaluate(self, pool, population, cur_gen=-1):
        gen_best = -float("Inf")
        gen_best_activity = None
        gen_best_idx = 0
        count_non_valid = 0
        # populations = pool.map(self.thread_fun_correctify_tile_dependency, population)
        # population[:] = populations
        reward_activ_list = pool.map(self.thread_fun, population)

        for i in range(len(population)):
            reward, activity_count = reward_activ_list[i]
            if reward is None or any(np.array(reward) >= 0):
                reward = [float("-Inf") for _ in range(len(self.best_reward))]
                count_non_valid += 1
            # elif stage_idx > 0:
            #     if any([reward[kk] < prev_stage_value[kk] for kk in range(len(prev_stage_value))]):
            #         reward = [float("-Inf") for _ in range(len(best_reward))]
            #         count_non_valid += 1
            judging_reward = reward[self.stage_idx]
            self.fitness[i] = reward
            if gen_best < judging_reward:
                gen_best = judging_reward
                gen_best_activity = activity_count
                gen_best_idx = i
        if self.use_ranking:
            self.fitness, gen_best_idx =  self.adjust_fitness(self.fitness)
            gen_best = - np.prod(self.fitness[gen_best_idx][1:])
            judging_best_reward = - np.prod(self.best_reward[1:])
        else:
            judging_best_reward = self.best_reward[self.stage_idx]
        # self.cal_statstics()

        if judging_best_reward < gen_best:
            if self.use_ranking:
                self.best_reward = copy.deepcopy(np.concatenate((np.array([gen_best]), self.fitness[gen_best_idx][1:])))
            else:
                self.best_reward = copy.deepcopy(self.fitness[gen_best_idx])
            self.best_activity = copy.deepcopy(gen_best_activity)
            self.best_sol = copy.deepcopy(population[gen_best_idx])

        self.best_reward_list.append(self.best_reward)

        chkpt = {
            "best_activity": self.best_activity,
            "best_reward": self.best_reward,
            "best_reward_list": self.best_reward_list,
            "best_sol": self.best_sol,
            "num_population": self.num_population,
            "num_generations": self.num_generations,
            "fitness_use": self.fitness_objective,
            "num_pe": self.num_pe,
            "pe_limit":self.pe_limit,
            "l1_size": self.l1_size,
            "l2_size": self.l2_size,
            "NocBW": self.NocBW,
            "dimension": self.dimension,
            "best_reward_pleteau":self.best_reward_pleteau ,
            "best_sol_pleteau":self.best_sol_pleteau ,
            # "stat":stat,
            # "stat_list":self.stat_list
        }
        # parent_ratio = max(self.parents_ratio, 0.1)
        self.num_parents = int(self.num_population * self.parents_ratio)
        self.num_parents = min(self.num_parents, len(population) - count_non_valid)
        self.parents_ratio *= self.ratio_decay
        # print("Gen {}: Best reward: {}".format( (cur_gen + 1), np.abs(self.best_reward)[0]))
        # if self.stage_idx == 0:
        #     print("[Stage {}]Gen {}: Best reward: {}".format(self.stage_idx + 1, (g + 1), np.abs(self.best_reward)[0]))
        # else:
        #     print("[Stage {}]Gen {}:  1st stage Reward: {}, Best reward: {}".format(self.stage_idx + 1, (g + 1),
        #                                                                             np.abs(prev_stage_value),
        #                                                                             np.abs(best_reward)))
        return chkpt

    def injection(self, inject_ratio=1.0):
        num_inject = int(self.num_population * inject_ratio)
        pop_inj = [self.create_genome_fixedSL() for _ in range(num_inject)]
        inj_fitness = np.ones((num_inject, len(self.fitness_objective)), float)
        return pop_inj, inj_fitness

    def run(self, dimension, stage_idx=0, prev_stage_value=0, num_population=100, num_generations=100, elite_ratio=0.05,
                       parents_ratio=0.4, ratio_decay=1, num_finetune=1, best_sol_1st=None, init_pop=None, bias=None, uni_base=True, use_factor=False, use_pleteau=False, L1_bias_template=None
):
        self.init_arguement(dimension=dimension, stage_idx=stage_idx, prev_stage_value=prev_stage_value, num_population=num_population, num_generations=num_generations, elite_ratio=elite_ratio,
                       parents_ratio=parents_ratio, ratio_decay=ratio_decay, num_finetune=num_finetune, best_sol_1st=best_sol_1st, init_pop=init_pop,uni_base=uni_base, use_factor=use_factor, use_pleteau=use_pleteau,L1_bias_template=L1_bias_template)
        pool = Pool(min(self.num_population + self.num_elite, cpu_count()))
        population = self.reinit_pop(pool,self.num_population,  self.stage_idx, self.best_sol_1st, self.init_pop, bias=bias)
        if self.map_cstr:
            self.cstr_list, self.num_free_order, self.num_free_par = self.map_cstr.get_cstr_list(copy.deepcopy(population[0]), fixed_sp_sz=self.fixedCluster)
        for g in range(num_generations):

            while self.num_parents < 1:  # restart
                population = self.reinit_pop(pool, self.num_population, self.stage_idx, self.best_sol_1st, self.init_pop, cur_gen=g)
                print("Reinitialize population")

            population, self.fitness, self.parents = self.select_parents(population, self.fitness, self.num_parents, self.num_population,)
            elite = copy.deepcopy(self.parents[:self.num_elite])
            self.elite_fitness = copy.deepcopy(self.fitness[:(len(elite))])

            # print(list(zip(self.elite_fitness, elite)))
            
            # Cross over operation is performed here
            self.crossover_tile(self.parents, population, alpha=0.57)
            if self.constraint_class == "1000":
                self.mutate_tile(population, num_mu_loc=3, range_alpha=0.53, alpha=1, is_finetune=False)
            elif self.constraint_class == "0100":
                self.swap_order(population, alpha=1)
            elif self.constraint_class == "0010":
                self.mutate_tile(population, num_mu_loc=1, range_alpha=0.53, alpha=1, is_finetune=False,
                                 cluster_only=True)
            elif self.constraint_class == "0001":
                self.mutate_par(population, alpha=1)
            else:
                # Perform reorder operation
                if self.use_reorder:
                    self.swap_order(population, alpha=self.reorder_alpha)

                # Perform mutation operation
                self.mutate_tile(population, num_mu_loc=3, range_alpha=0.53, alpha=0.53, is_finetune=False)
                self.mutate_pe(population, alpha=1 if g==0 else 0.5) if self.num_pe<1 else None
                self.mutate_par(population, alpha=0.1)


            if self.map_cstr is None:
                # perform growing operation here
                if self.use_growing:
                    self.born_cluster(population, alpha=self.growing_alpha)

                # perform aging operation here
                if self.use_aging:
                    
                    self.kill_cluster(population, alpha=self.aging_alpha)


            # pop_inj, inj_fitness = self.injection()
            self.correctify_tile_dependency(population)
            # self.calculate_equivalent_num_pe(population)
            self.comform_to_cstr(population)
            population = elite + population
            # population = elite + population + pop_inj
            self.fitness = np.concatenate((self.elite_fitness, self.fitness))
            # self.fitness = np.concatenate((self.elite_fitness, self.fitness, inj_fitness))
            chkpt = self.evaluate(pool=pool, population=population, cur_gen=g)
            # self.check_tile_dependency(population)

            if self.log_level>1:
                if chkpt["best_sol"] is not None and self.log_level>1:
                    best_runtime, best_throughput, best_energy, best_area, best_l1_size, best_l2_size, best_mac, best_power, best_num_pe = self.get_indiv_info( chkpt["best_sol"])
                    # best_num_pe = chkpt["best_sol"][0][1] if self.num_pe<1 else self.num_pe
                    # print(f"Runtime: {best_runtime}, L1: {best_l1_size}, L2: {best_l2_size}, L1_usage:{best_l1_size/self.l1_size:}, L2_usage:{best_l2_size/self.l2_size:.4f}, PE: {best_num_pe}")
                    print(f"Gen {g+1}: Reward: {chkpt['best_reward'][0]:.3e}, Runtime: {best_runtime}, Area: {best_area/1e6:.3f}mm2,  PE Area_ratio: {best_num_pe*MAC_AREA_INT8/best_area*100:.1f}%, L1: {best_l1_size}, L2: {best_l2_size},  PE: {best_num_pe}")
                else:
                    print(f"Gen {g+1}: Reward: {chkpt['best_reward'][0]:.3e}")

        population = self.sort_population(population)
        pool.close()
        return chkpt, population[:self.num_population]

    def calculate_equivalent_num_pe(self, population):
        for idx in range(len(population)):
            indv = population[idx]
            num_pe, sp_sz = indv[0][1], indv[7][1]
            num_cluster = num_pe//sp_sz
            sp_dim_L2_loc = [i for i, item in enumerate(indv) if item[0]==indv[0][0] and i%7!=0]
            sp_real_tile_sizeL2 = indv[sp_dim_L2_loc[0]][1]
            sp_real_tile_sizeL1 = indv[sp_dim_L2_loc[1]][1]
            if sp_real_tile_sizeL2 > num_cluster:
                sp_dim_sp_sizeL2 = ceil(sp_real_tile_sizeL2/num_cluster)
                using_num_cluster = num_cluster
            else:
                using_num_cluster = sp_real_tile_sizeL2
                sp_dim_sp_sizeL2 = 1
            if sp_dim_sp_sizeL2 < sp_real_tile_sizeL1:
                sp_dim_sp_sizeL2 = sp_real_tile_sizeL1
                using_num_cluster = ceil(sp_real_tile_sizeL2/sp_dim_sp_sizeL2)
            indv[0][1] = using_num_cluster * sp_sz
            indv[sp_dim_L2_loc[0]][1] = sp_dim_sp_sizeL2

            if indv[0][1]>self.pe_limit:
                print("error1")
            if indv[sp_dim_L2_loc[0]][1] * using_num_cluster < sp_real_tile_sizeL2:
                print("error2")
        return population

    def sort_population(self, population):
        population, self.fitness, self.parents = self.select_parents(population, self.fitness, self.num_parents,
                                                                     self.num_population,)
        return population


    def thread_fun_correctify_tile_dependency(self, indv):
        return self.correctify_tile_dependency_thread(indv)

    def thread_fun(self, individual):
        reward, activity_count = self.oberserve_maestro(individual)
        return [reward, activity_count]

    def get_indiv_info(self, individual, num_pe=None, l1_size=None, l2_size=None, NocBW=None):
        self.oberserve_maestro(individual,num_pe=num_pe, l1_size=l1_size, l2_size=l2_size, NocBW=NocBW)
        return self.observation

    def get_CONVtypeShape(self, dimensions, CONVtype=1):
        CONVtype = CONVtype_dicts[CONVtype]
        if CONVtype == "CONV"or CONVtype=="DSCONV":
            pass
        elif CONVtype == "GEMM" or CONVtype=="SGEMM":
            SzM, SzN, SzK,*a = dimensions
            dimensions = [SzN, SzK, SzM, 1, 1, 1]
        elif CONVtype == "FC":
            SzOut, SzIn, *a = dimensions
            dimensions = [SzOut, SzIn, 1, 1, 1, 1]
        else:
            print("Not supported layer.")
        return dimensions

    def write_maestro(self, indv, layer_id=0, m_file=None):
        print("[DEBUG][write_maestro][m_file: {}]", m_file)
        dimensions = [self.dimension]
        with open("{}.m".format(m_file), "w") as fo:
            fo.write("Network {} {{\n".format(layer_id))
            for i in range(len(dimensions)):
                dimension = dimensions[i]
                m_type = m_type_dicts[int(dimension[-1])]
                dimension = self.get_CONVtypeShape(dimension, int(dimension[-1]))
                fo.write("Layer {} {{\n".format(m_type))
                fo.write("Type: {}\n".format(m_type))
                fo.write(
                    "Dimensions {{ K: {:.0f}, C: {:.0f}, Y: {:.0f}, X: {:.0f}, R: {:.0f}, S: {:.0f} }}\n".format(
                        *dimension))
                fo.write("Dataflow {\n")
                for k in range(0, len(indv), 7):
                    for i in range(k, k + 7):
                        if len(indv[i]) == 2:
                            d, d_sz = indv[i]
                        else:
                            d, d_sz, _ = indv[i]
                        if i % 7 == 0:
                            if k != 0:
                                fo.write("Cluster({},P);\n".format(d_sz))
                        else:
                            sp = "SpatialMap" if d == indv[k][0] or (
                                        len(indv[k]) > 2 and d == indv[k][2]) else "TemporalMap"
                            # MAESTRO cannot take K dimension as dataflow file
                            if not (m_type == "DSCONV"):
                                fo.write("{}({},{}) {};\n".format(sp, d_sz, d_sz, self.get_out_repr(d)))
                            else:
                                if self.get_out_repr(d) == "C" and self.get_out_repr(indv[k][0]) == "K":
                                    fo.write("{}({},{}) {};\n".format("SpatialMap", d_sz, d_sz, "C"))
                                else:
                                    if not (self.get_out_repr(d) == "K"):
                                        fo.write("{}({},{}) {};\n".format(sp, d_sz, d_sz, self.get_out_repr(d)))

                fo.write("}\n")
                fo.write("}\n")
            fo.write("}")
            
    def oberserve_maestro(self, indv, num_pe=None, l1_size=None, l2_size=None, NocBW=None, offchipBW=None):
        print("[DEBUG][indv] {}".format(indv))
        m_file = "{}".format(random.randint(0, 2**32))
        self.write_maestro(indv,m_file=m_file)
        print("[Debug][]m-file",m_file)

        if num_pe:
            to_use_num_pe = num_pe
        elif self.num_pe <1:
            to_use_num_pe = indv[0][1]
        else:
            to_use_num_pe = self.num_pe
        # print(num_pe, bw, l1_size)
        os.remove("./{}.csv".format(m_file)) if os.path.exists("./{}.csv".format(m_file)) else None
        command = [self._executable,
                   "--Mapping_file={}.m".format(m_file),
                   "--full_buffer=false", "--noc_bw_cstr={}".format(self.NocBW if not NocBW else NocBW),
                   "--noc_hops=1", "--noc_hop_latency=1",
                   "--offchip_bw_cstr={}".format(self.offchipBW if not offchipBW else offchipBW),
                   "--noc_mc_support=true", "--num_pes={}".format(int(to_use_num_pe)),
                   "--num_simd_lanes=1", "--l1_size_cstr={}".format(self.l1_size if not l1_size else l1_size),
                   "--l2_size_cstr={}".format(self.l2_size if not l2_size else l2_size), "--print_res=false", "--print_res_csv_file=true", "--print_log_file=false", "--print_design_space=false", "--msg_print_lv=0"]

        print("[DEBUG][Command]", command)
        sys.exit()
        process = Popen(command, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        process.wait()
        #os.remove("./{}.m".format(m_file)) if os.path.exists("./{}.m".format(m_file)) else None
        try:
            df = pd.read_csv("./{}.csv".format(m_file))
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
            os.remove("./{}.csv".format(m_file))  if os.path.exists("./{}.csv".format(m_file)) else None
            os.remove("./log.txt") if os.path.exists("./log.txt") else None
            if self.external_area_model:
                area = self.compute_area_external(to_use_num_pe, l1_size, l2_size)
            elif self.area_pebuf_only:
                area = self.compute_area_maestro(to_use_num_pe, l1_size, l2_size)

            self.observation = [np.mean(x) for x in [runtime, throughput, energy, area, l1_size, l2_size, mac, power, to_use_num_pe]]
            def catch_exception():
                if l1_size>self.l1_size or l2_size>self.l2_size or any(runtime_series<1) or any(l1_size_series<1) or any(l2_size_series<1):
                    return True
                else:
                    return False
            stdout_as_str = stdout.decode("utf-8")
            stdout_as_str = "".join(stdout_as_str.split())
            # if (len(str(stdout))>3 and stdout_as_str[:len("Numpartialsumsislessthan0!")]!="Numpartialsumsislessthan0!") or catch_exception() or not self.validTo_external_mem_cstr(indv, num_pe=to_use_num_pe):
            # if len(str(stdout))>3  or catch_exception() or not self.validTo_external_mem_cstr(indv, num_pe=to_use_num_pe):
            if  catch_exception() or not self.validTo_external_mem_cstr(indv, num_pe=to_use_num_pe):
            # if  catch_exception():
                return None, None
            return self.judge(), activity_count
        except:
            return None, None

    def impose_halloffame(self, observe_value, target="latency_ave" ):
        is_violated = False
        if self.stat is not None:
            target_value = self.stat[target]
            if observe_value > target_value:
                is_violated = True
        return is_violated

    def compute_area_maestro(self, num_pe, l1_size, l2_size):
        MAC_AREA_MAESTRO=4470
        L2BUF_AREA_MAESTRO = 4161.536
        L1BUF_AREA_MAESTRO = 4505.1889
        L2BUF_UNIT = 32768
        L1BUF_UNIT = 64
        area = num_pe * MAC_AREA_MAESTRO + ceil(int(l2_size)/L2BUF_UNIT)*L2BUF_AREA_MAESTRO + ceil(int(l1_size)/L1BUF_UNIT)*L1BUF_AREA_MAESTRO * num_pe
        return area

    def compute_area_external(self, num_pe, l1_size, l2_size):
        MAC_AREA_INT8=282
        MAC_AREA_INT32=3495
        BUF_AREA_perbit = 0.086
        buf_size = l1_size * num_pe + l2_size
        area = num_pe * MAC_AREA_INT8 + buf_size * BUF_AREA_perbit * 8
        return area

    def judge(self):
        runtime, throughput, energy, area, l1_size, l2_size, mac, power, num_pe = self.observation

        def get_objective(objective):
            values = []
            for term in objective:
                if term[:1] == "n":
                    term = term[1:]
                if term == "energy":
                    reward = -energy
                elif term == "thrpt_ave":
                    reward = throughput
                elif term == "EDP":
                    reward = -energy * runtime
                elif term == "LAP":
                    reward = -area * runtime
                elif term == "LALP":
                    reward = -area * runtime * runtime
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
                elif term =="ranking":
                    reward = -1
                elif term =="L-PE-L2":
                    reward = -runtime * num_pe * l2_size
                elif term =="L-PE":
                    reward = -runtime * num_pe
                elif term == "PE":
                    reward = -num_pe
                else:
                    raise NameError('Undefined fitness type')
                if term in self.constraints:
                    if reward < -self.constraints[term]:
                        return [float("-Inf")] * len(self.fitness_objective)
                values.append(reward)
            return values
        values = get_objective(self.fitness_objective)
        return values

    def print_indv(self, indv,fd=False):
        for k in range(0, len(indv), 7):
            if fd:
                fd.write("\n{}".format(indv[k:k+7]))
            else:
                print(indv[k:k+7])

    def init_arguement(self, dimension=None, stage_idx=0, prev_stage_value=0, num_population=100, num_generations=100,
                       elite_ratio=0.05,
                       parents_ratio=0.15, ratio_decay=1, num_finetune=1, best_sol_1st=None, init_pop=None, uni_base=False, use_factor=False, use_pleteau=False,L1_bias_template=None):
        self.stage_idx = stage_idx
        self.num_generations = num_generations
        self.num_population = num_population
        self.prev_stage_value = prev_stage_value
        self.ratio_decay = ratio_decay
        self.best_sol_1st = best_sol_1st
        self.init_pop = init_pop
        self.parents_ratio = parents_ratio
        self.num_elite = int(num_population * elite_ratio)
        self.best_reward_list = []
        self.best_reward = [-float("Inf") for _ in range(len(self.fitness_objective))]
        self.best_activity = None
        self.best_sol = None
        self.stat_list = []
        self.uni_base =uni_base
        self.stat = None
        self.pleteau_sol = dict()
        self.use_factor = use_factor
        self.use_pleteau = use_pleteau
        self.best_reward_pleteau = None
        self.best_sol_pleteau = None
        self.normalize=True   if self.fitness_objective[0][:1] == "n" else False
        self.L1_bias_template =L1_bias_template

