import nevergrad as ng
import copy
import argparse
import other_method_env as Env
from datetime import datetime
from multiprocessing import Pool
from multiprocessing import cpu_count

from functools import partial
import glob
import os, sys
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '../'))
if module_path not in sys.path:
    sys.path.insert(0,module_path)
from utils import *
fitness_list = None
fitness = None
stage_idx = 0
prev_stage_value = []
tune_iter = 1
choose_optimizer=None
search_level = 1






def get_reward(env, indv):
    if opt.fixedCluster>0:
        reward = env.oberserve_maestro(indv, sp2_sz=opt.fixedCluster)
    else:
        reward = env.oberserve_maestro(indv)
    if reward is None:
        reward = [-2**63]
    reward = reward[0]
    return -reward

def eval_function(env, indv0, indv1, indv2, indv3, indv4, indv5, indv6, indv7):
    indv = [indv0, indv1, indv2, indv3, indv4, indv5, indv6, indv7]
    return get_reward(env, indv)

def eval_function_2_level(env, indv0, indv1, indv2, indv3, indv4, indv5, indv6, indv7, indv8, indv9, indv10, indv11, indv12, indv13, indv14, indv15):
    indv = [indv0, indv1, indv2, indv3, indv4, indv5, indv6, indv7, indv8, indv9, indv10, indv11, indv12, indv13, indv14, indv15]
    return get_reward(env, indv)



def ng_search_2_level(env,num_generations=100,num_population=100,dimension=None,choose_optimizer=None):
    parametrization = ng.p.Instrumentation(
    indv0=ng.p.Scalar(lower=0, upper=3).set_integer_casting(),
    indv1=ng.p.Scalar(lower=0, upper=720-1).set_integer_casting(),
    indv2=ng.p.Scalar(lower=1, upper=dimension[0]).set_integer_casting()   if dimension[0]>1 else 1,
    indv3=ng.p.Scalar(lower=1, upper=dimension[1]).set_integer_casting() if dimension[1]>1 else 1,
    indv4=ng.p.Scalar(lower=1, upper=dimension[2]).set_integer_casting() if dimension[2]>1 else 1,
    indv5=ng.p.Scalar(lower=1, upper=dimension[3]).set_integer_casting() if dimension[3]>1 else 1,
    indv6=ng.p.Choice(np.arange(1, dimension[4], 2)) if dimension[4]>1 else 1,
    indv7=ng.p.Choice(np.arange(1, dimension[5], 2)) if dimension[5]>1 else 1,
    indv8=ng.p.Scalar(lower=0, upper=3).set_integer_casting(),
    indv9=ng.p.Scalar(lower=0, upper=720-1).set_integer_casting(),
    indv10=ng.p.Scalar(lower=1, upper=dimension[0]).set_integer_casting()   if dimension[0]>1 else 1,
    indv11=ng.p.Scalar(lower=1, upper=dimension[1]).set_integer_casting() if dimension[1]>1 else 1,
    indv12=ng.p.Scalar(lower=1, upper=dimension[2]).set_integer_casting() if dimension[2]>1 else 1,
    indv13=ng.p.Scalar(lower=1, upper=dimension[3]).set_integer_casting() if dimension[3]>1 else 1,
    indv14=ng.p.Choice(np.arange(1, dimension[4], 2)) if dimension[4]>1 else 1,
    indv15=ng.p.Choice(np.arange(1, dimension[5], 2)) if dimension[5]>1 else 1 )
    optimizer = ng.optimizers.registry[choose_optimizer](parametrization=parametrization, budget=num_generations* num_population, num_workers=1)
    partial_func = partial(eval_function_2_level,env)
    recommendation = optimizer.minimize(partial_func)
    answer = recommendation.kwargs
    indv = [answer["indv{}".format(i)] for i in range(len(answer))]
    reward = get_reward(env, indv)
    indv_enc = env.encode(indv, sp2_sz = opt.fixedCluster)
    print("Cases: {}, Best fitness: {}".format(num_generations* num_population, reward))
    print("Best sol:\n")
    print(indv_enc)
    chkpt = {
        "best_reward": reward,
        "best_sol": indv_enc,
        "num_population": num_population,
        "num_generations": num_generations,
        "fitness_use": opt.fitness1,
        "num_pe": opt.num_pe,
        "l1_size": opt.l1_size,
        "l2_size": opt.l2_size,
        "dimension": dimension
    }
    return chkpt




def ng_search(env,num_generations=100,num_population=100,dimension=None,choose_optimizer=None):
    parametrization = ng.p.Instrumentation(
    indv0=ng.p.Scalar(lower=0, upper=3).set_integer_casting(),
    indv1=ng.p.Scalar(lower=0, upper=720-1).set_integer_casting(),
    indv2=ng.p.Scalar(lower=1, upper=dimension[0]).set_integer_casting()   if dimension[0]>1 else 1,
    indv3=ng.p.Scalar(lower=1, upper=dimension[1]).set_integer_casting() if dimension[1]>1 else 1,
    indv4=ng.p.Scalar(lower=1, upper=dimension[2]).set_integer_casting() if dimension[2]>1 else 1,
    indv5=ng.p.Scalar(lower=1, upper=dimension[3]).set_integer_casting() if dimension[3]>1 else 1,
    indv6=ng.p.Choice(np.arange(1, dimension[4], 2)) if dimension[4]>1 else 1,
    indv7=ng.p.Choice(np.arange(1, dimension[5], 2)) if dimension[5]>1 else 1 )
    optimizer = ng.optimizers.registry[choose_optimizer](parametrization=parametrization, budget=num_generations* num_population, num_workers=1)
    partial_func = partial(eval_function,env)
    recommendation = optimizer.minimize(partial_func)
    answer = recommendation.kwargs
    indv = [answer["indv{}".format(i)] for i in range(len(answer))]
    reward = get_reward(env, indv)
    indv_enc = env.encode(indv)
    print("Cases: {}, Best fitness: {}".format(num_generations* num_population, reward))
    print("Best sol:\n")
    print(indv_enc)
    chkpt = {
        "best_reward": reward,
        "best_sol": indv_enc,
        "num_population": num_population,
        "num_generations": num_generations,
        "fitness_use": opt.fitness1,
        "num_pe": opt.num_pe,
        "l1_size": opt.l1_size,
        "l2_size": opt.l2_size,
        "dimension": dimension
    }
    return chkpt

def combine_chkpt(chkpt1, chkpt2):
    rew1 = chkpt1["best_reward"]
    rew2 = chkpt2["best_reward"]
    if rew2> rew1:
        chkpt1["best_reward"] = chkpt2["best_reward"]
        chkpt1["best_sol"] = chkpt2["best_sol"]
    chkpt1["num_generations"] = chkpt1["num_generations"] + chkpt2["num_generations"]
    return chkpt1
def save_chkpt(layerid_to_dim,dim_info,dim_set,first_stage_value=None,choose_optimizer=None):
    chkpt = {
        "layerid_to_dim": layerid_to_dim,
        "dim_info": dim_info,
        "dim_set": dim_set,
        "first_stage_value":first_stage_value,
        "choose_optimizer": choose_optimizer

    }
    with open(chkpt_file, "wb") as fd:
        pickle.dump(chkpt, fd)



def thread_fun(dimension):
    dimension = np.array(dimension, dtype=int)
    env = Env.MaestroEnvironment(dimension=dimension, num_pe=opt.num_pe, fitness=[opt.fitness1], par_RS=opt.parRS,
                                 l1_size=opt.l1_size,
                                 l2_size=opt.l2_size)
    if search_level == 1:
        chkpt = ng_search(env, num_generations=opt.epochs, num_population=opt.num_pop, dimension=dimension,
                                  choose_optimizer=choose_optimizer)
    else:
        chkpt = ng_search_2_level(env, num_generations=opt.epochs, num_population=opt.num_pop, dimension=dimension,
                      choose_optimizer=choose_optimizer)
    return chkpt

def train_model(model_defs):
    layerid_to_dim = {}
    dim_infos = {}
    stages=1
    dim_set = set((tuple(m) for m in model_defs))
    threadcount = len(dim_set)
    pool = Pool(min(threadcount, cpu_count()))
    for i, dim in enumerate(model_defs):
        layerid_to_dim[i] = dim
    for s in range(stages):
        dim_stage = {}
        for dimension in dim_set:
            dim_stage[dimension] = {"best_reward": float("-Inf") }
        dim_infos["Stage{}".format(s+1)] = copy.deepcopy(dim_stage)
    dim_list = list(dim_set)
    chkpt_list = pool.map(thread_fun, dim_list)
    for i, chkpt in enumerate(chkpt_list):
        best_reward = chkpt["best_reward"]
        cur_best_reward =  dim_infos["Stage{}".format(s+1)][dim_list[i]]["best_reward"]
        if cur_best_reward <= best_reward:
            dim_infos["Stage{}".format(s+1)][dim_list[i]] = chkpt
    save_chkpt(layerid_to_dim, dim_infos, dim_set, choose_optimizer=choose_optimizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fitness1', type=str, default="latency", help='first stage fitness')
    parser.add_argument('--stages', type=int, default=2, help='number of stages', choices=[1,2])
    parser.add_argument('--num_pop', type=int, default=100, help='number of populations')
    parser.add_argument('--parRS', default=False, action='store_true', help='Parallize across R S dimension')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--outdir', type=str, default="outdir", help='output directiory')
    parser.add_argument('--num_pe', type=int, default=168, help='number of PEs')
    parser.add_argument('--l1_size', type=int, default=512, help='L1 size')
    parser.add_argument('--l2_size', type=int, default=108000, help='L2 size')
    parser.add_argument('--NocBW', type=int, default=8192000, help='NoC BW')
    parser.add_argument('--hwconfig', type=str, default="hw_config.m", help='HW configuration file')
    parser.add_argument('--model', type=str, default="vgg16", help='Model to run')
    parser.add_argument('--num_layer', type=int, default=0, help='number of layers to optimize')
    parser.add_argument('--singlelayer', type=int, default=1, help='The layer index to optimize')
    parser.add_argument('--slevel', type=int, default=2, help='parallelization level')
    parser.add_argument('--fixedCluster', type=int, default=0, help='Rigid cluster size')
    parser.add_argument('--log_level', type=int, default=1, help='Detail: 2, runtimeinfo: 1')
    parser.add_argument('--method', type=str, default="DE", help='["PSO", "Portfolio", "OnePlusOne","CMA", "DE","TBPSA","pureGA","Random"]',
                        choices=["PSO", "Portfolio", "OnePlusOne","CMA", "DE","TBPSA","pureGA","Random"])

    opt = parser.parse_args()
    opt = set_hw_config(opt)
    m_file_path = "../../data/model/"
    method = get_method(opt)
    m_file = os.path.join(m_file_path, opt.model + ".csv")
    df = pd.read_csv(m_file)
    model_defs = df.to_numpy()
    if opt.singlelayer:
        model_defs = model_defs[opt.singlelayer - 1:opt.singlelayer]
    else:
        if opt.num_layer:
            model_defs = model_defs[:opt.num_layer]
    _, dim_size = model_defs.shape
    now = datetime.now()
    now_date = "{}".format(now.date())
    now_time = "{}".format(now.time())
    outdir = opt.outdir
    outdir = os.path.join("../../", outdir)
    choose_optimizer = method
    if opt.fixedCluster>0:
        exp_name = "{}_{}_SL-{}-{}_FixCl-{}_F1-{}_PE-{}_GEN-{}_POP-{}_L1-{}_L2-{}".format(method, opt.model,opt.slevel,opt.slevel,opt.fixedCluster, opt.fitness1, opt.num_pe, opt.epochs, opt.num_pop, opt.l1_size, opt.l2_size)
    else:
        exp_name = "{}_{}_SL-{}-{}_F1-{}_PE-{}_GEN-{}_POP-{}_L1-{}_L2-{}".format(method, opt.model,opt.slevel,opt.slevel,opt.fitness1, opt.num_pe, opt.epochs, opt.num_pop, opt.l1_size, opt.l2_size)

    search_level = opt.slevel

    outdir_exp = os.path.join(outdir, exp_name)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir_exp, exist_ok=True)
    chkpt_file_t = "{}".format("result")
    chkpt_file = os.path.join(outdir_exp, chkpt_file_t + "_c.plt")




    try:
        train_model(model_defs)
        print_result(chkpt_file)

    finally:
        for f in glob.glob("*.m"):
            os.remove(f)
        for f in glob.glob("*.csv"):
            os.remove(f)
