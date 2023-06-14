import random
import numpy as np
from collections import OrderedDict
dimension_set = {"K","C","R","S","Y","X"}
class Constraint():
    def __init__(self,num_pe=256):
        self.lastcluster_dict = {}
        self.cstr_list = [None, None, None]
        self.num_free_order = 21
        self.num_pe = num_pe
    def set_constraint(self, level, cstr):
        if level == "L3":
            L = 2
        if level == "L2":
            L = 1
        if level == "L1":
            L  = 0
        if self.cstr_list[L] is None:
            self.cstr_list[L] = OrderedDict(cstr)
        else:
            for key, value in cstr.items():
                if key in self.cstr_list[L]:
                    self.cstr_list[L].pop(key, None)
                self.cstr_list[L][key] = value

    def impose_constraint(self, indv, fixed_sp_sz=0):
        num_levels = len(indv)//7
        index_offset = 0
        self.lastcluster_dict = {}
        for nl in range(num_levels-1,-1,-1):
            self.set_valid_value_v2(self.cstr_list[nl], index_offset, indv, fixed_sp_sz=fixed_sp_sz)
            index_offset += 7

    def create_from_constraint(self, indv, fixed_sp_sz=0, dimension_dict=None):
        num_levels = len(indv)//7
        index_offset = 0
        self.dimension_dict = dimension_dict
        self.lastcluster_dict = {}
        ret_num_free_order = 0
        ret_num_free_par = 0
        for nl in range(num_levels-1,-1,-1):
            num_free_order, num_free_par, free_orders,num_free_tile = self.set_valid_value_v2(self.cstr_list[nl], index_offset, indv, fixed_sp_sz=fixed_sp_sz)
            index_offset += 7
            ret_num_free_order += num_free_order
            ret_num_free_par += num_free_par
            self.cstr_list[nl]["free_order"] = free_orders
            self.cstr_list[nl]["num_free_tile"] = num_free_tile
        return ret_num_free_order, ret_num_free_par

    def reverse_cstr_list(self, cstr_list):
        ret = []
        for i in range(2, -1, -1):
            if cstr_list[i] is not None:
                ret.append(cstr_list[i])
        return ret

    def get_cstr_list(self, indv, fixed_sp_sz=0):
        ret_num_free_order, ret_num_free_par = self.create_from_constraint(indv, fixed_sp_sz, self.dimension_dict)
        return self.reverse_cstr_list(self.cstr_list), ret_num_free_order, ret_num_free_par

    def set_valid_value(self, lever_cstr, index_offset, indv,fixed_sp_sz=0):
        for key, value in lever_cstr.items():
            if key == "sp":
                if  indv[index_offset][0] not in value:
                    sp = np.random.choice(value, 1)[0]
                    indv[index_offset][0] = sp
                    if len(self.lastcluster_dict) > 0:
                        if fixed_sp_sz > 0:
                            sp_sz = fixed_sp_sz
                        else:
                            sp_sz = self.lastcluster_dict[sp]
                        indv[index_offset][1] = sp_sz
            for i in range(index_offset + 1, index_offset + 7):
                if indv[i][0] == key:
                    change_idx = i
                if indv[i][0] == value:
                    valid_value = random.randint(1, indv[i][1])
                self.lastcluster_dict[indv[i][0]] = indv[i][1]
            if key in dimension_set:
                if valid_value >  indv[change_idx][1]:
                    indv[change_idx][1] = valid_value





    def set_valid_value_v2(self, lever_cstr, index_offset, indv, fixed_sp_sz=0):
        num_free_order = 6
        num_free_par  = 1
        num_free_tile = 6
        free_orders = {"X", "Y", "K","C","R","S"}
        for key, value in lever_cstr.items():
            if key == "sp":
                sp = np.random.choice(value, 1)[0]
                sp_sz = indv[index_offset][1]
                if len(self.lastcluster_dict) > 0:
                    if fixed_sp_sz > 0:
                        sp_sz = fixed_sp_sz
                    else:
                        if sp != indv[index_offset][0]:
                            sp_sz = random.randint(1,min(self.num_pe if self.num_pe>0 else float('Inf'), self.lastcluster_dict[sp])) if len(self.lastcluster_dict) >0 else self.dimension_dict[sp]
                indv[index_offset] = [sp, sp_sz]
                num_free_par = 0
            elif key== "sp2":
                sp2 = np.random.choice(value, 1)[0]
                if len(indv[index_offset])>2:
                    indv[index_offset][2] = sp2
                else:
                    indv[index_offset].append(sp2)
            elif key == "sp_sz":
                if type(value) is int:
                    indv[index_offset][1] = value
                else:
                    indv[index_offset][1] =  random.randint(1,min(self.num_pe if self.num_pe>0 else float('Inf'), self.lastcluster_dict[value])) if len(self.lastcluster_dict) >0 else self.dimension_dict[value]
            elif key == "order":
                free_orders -= set(value)
                tile_dict = OrderedDict()
                for i in range(index_offset + 1, index_offset + 7):
                    tile_dict[indv[i][0]] = indv[i][1]
                num_fixed_order = len(value)
                num_free_order -= num_fixed_order
                for value_idx, i in enumerate(range(index_offset + 1 + num_free_order, index_offset + 7)):
                    indv[i][0] = value[value_idx]
                    indv[i][1] =  tile_dict[indv[i][0]]
                    tile_dict.pop(indv[i][0], None)
                for i in  range(index_offset + 1, index_offset + 1 + num_free_order):
                    indv[i][0], indv[i][1] = tile_dict.popitem(last=False)
            elif key in dimension_set:
                num_free_tile -= 1
                for i in range(index_offset + 1, index_offset + 7):
                    if indv[i][0] == key:
                        change_idx = i
                        if type(value) is int:
                            indv[change_idx][1] = value
                        else:
                            if type(value) is list:
                                left, right = value
                                right_num = self.lastcluster_dict[right] if len(self.lastcluster_dict) >0 else self.dimension_dict[right]
                                # indv[change_idx][1] =  random.randint(left, right_num)
                                indv[change_idx][1] =  right_num
                            else:
                                indv[change_idx][1] = self.lastcluster_dict[value] if len(self.lastcluster_dict) >0 else self.dimension_dict[value]
                        break



        for i in range(index_offset + 1, index_offset + 7):
            self.lastcluster_dict[indv[i][0]] = indv[i][1]
        return num_free_order, num_free_par, free_orders,num_free_tile

