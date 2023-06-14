import matplotlib
import os,sys
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '../'))
if module_path not in sys.path:
    sys.path.insert(0,module_path)
import os, pickle
import GAMMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_hwconfig_file(file):
    dicts = {}
    with open(file, "r") as fd:
       for line in fd:
           try:
                key, values = line.split(":")
                dicts[key.strip()] = values.strip()
           except:
               continue
    return dicts


def set_hw_config(opt):
    if opt.hwconfig is None:
        return opt
    hw_config  = os.path.join("../../data/HWconfigs", opt.hwconfig)
    hw_dicts = get_hwconfig_file(hw_config)
    opt.l1_size = int(hw_dicts["L1Size"]) if "L1Size" in hw_dicts else opt.l1_size
    opt.l2_size = int(hw_dicts["L2Size"]) if "L2Size" in hw_dicts else opt.l2_size
    opt.num_pe = int(hw_dicts["NumPEs"]) if "NumPEs" in hw_dicts else opt.num_pe
    opt.NocBW = int(hw_dicts["NoC_BW"]) if "NoC_BW" in hw_dicts else opt.NocBW
    return opt

def get_method(opt):
    method = opt.method
    if method == "TBPSA":
        method = "NaiveTBPSA"
    elif method == "pureGA":
        method = "cGA"
    elif method == "Random":
        method = "RandomSearch"
    return method



def print_indv(indv, fd=False):
    for k in range(0, len(indv), 7):
        if fd:
            fd.write("\n{}".format(indv[k:k + 7]))
        else:
            print(indv[k:k + 7])


def print_result(file):
    csv = file[:-4] + ".csv"
    log_dir = file[:-4] + "_log"
    img_dir = file[:-4] + "_img"
    m_dir = file[:-4] + "_m"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(m_dir, exist_ok=True)
    with open(file, "rb") as fd:
        chkpt_all = pickle.load(fd)
    dim_info = chkpt_all["dim_info"]

    layerid_to_dim = chkpt_all["layerid_to_dim"]
    num_stage = len(dim_info)
    any_value = next(iter(dim_info["Stage1"].values()))
    fitness_use = any_value["fitness_use"]
    column = ["Layer", "Dimension"]
    if isinstance(fitness_use, list):
        for n_s in range(num_stage):
            column.extend(["{}:{}".format(n_s+1,fit) for fit in fitness_use])
        df = pd.DataFrame(columns=column)
        for idx, dim in layerid_to_dim.items():
            value_pack = []
            for s in range(len(dim_info)):
                value = dim_info["Stage{}".format(s+1)][tuple(dim)]
                best_reward = [abs(v) for v in value["best_reward"][:2]]
                value_pack.extend(best_reward)
            df2 = pd.DataFrame(np.array([idx+1, dim, *value_pack]).reshape(1,-1),index=[idx],columns=column)
            df = df.append(df2)
        df.to_csv(csv)
    for idx, dim in layerid_to_dim.items():
        img_file = os.path.join(img_dir, "Layer_{}".format(idx + 1))
        log_file = os.path.join(log_dir, "Layer_{}".format(idx + 1))
        cur_m_file = os.path.join(m_dir, "Layer_{}".format(idx + 1))
        value = dim_info["Stage1"][tuple(dim)]
        num_pe = value["num_pe"]
        best_reward = value["best_reward"][0] if isinstance(fitness_use, list) else value["best_reward"]
        best_sol = value["best_sol"]
        num_population = value["num_population"]
        num_generations = value["num_generations"]
        print("Best  fitness :{:9e}".format(best_reward))
        print("Best Sol:")
        print_indv(best_sol)
        dimension = np.array(dim)
        with open(log_file, "w") as fd:
            fd.write("Layer id: {}\n".format(idx + 1))
            fd.write("Layer: {}\n".format(dimension))
            fd.write("\nNum generation: {}, Num population: {}".format(num_generations, num_population))
            fd.write("\nBest  fitness :{:9e}".format(abs(best_reward)))
            fd.write("Best Sol:")
            print_indv(best_sol, fd=fd)
        env = GAMMA.GAMMA(dimension=dimension, fitness=[fitness_use[0]])
        env.write_maestro(best_sol, m_file=cur_m_file)
        if isinstance(fitness_use, list):
            os.makedirs(img_dir, exist_ok=True)
            best_reward_list = np.array(value["best_reward_list"])[:, 0]
            font = {
                'weight': 'bold',
                'size': 12}
            matplotlib.rc('font', **font)
            fig = plt.figure(0)
            plt.plot(np.arange(len(best_reward_list)), np.abs(np.array(best_reward_list)), label="GAMMA", linewidth=5)
            plt.figtext(0.5, 0.65, "GAMMA fitness: {:9e}".format(abs(best_reward)))
            plt.figtext(0.5, 0.7, "Layer: {}".format(dimension))
            plt.figtext(0.5, 0.75, "Layer Id: {}".format(idx))
            plt.figtext(0.5, 0.8, "Num PE: {}".format(num_pe))
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            plt.ylabel(fitness_use[0], fontdict=font)
            plt.xlabel('Generation #', fontdict=font)
            plt.legend()
            plt.savefig(img_file + ".png", dpi=300)
            plt.show()
            plt.close(fig)

