#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import numpy as np
from settings import config
import matplotlib.pyplot as plt
from copy import deepcopy
import copy
import matplotlib.cbook as cbook
import _pickle as cPickle

if config.simulation_method == "power_knobs":
    from specs.database_input_powerKnobs import *
elif config.simulation_method == "performance":
    from specs.database_input import *
else:
    raise NameError("Simulation method unavailable")


# ------------------------------
# Functionality:
#       plot moves stats
# ------------------------------
def move_profile_plot(move_lists_):
    move_lists = [move_ for move_ in move_lists_ if not(move_.get_metric() == "cost")] # for now filtered cost
    move_on_metric_freq = {}
    for metric in config.budgetted_metrics:
        move_on_metric_freq[metric] = [0]

    for move in move_lists:
        metric = move.get_metric()
        move_on_metric_freq[metric] = [move_on_metric_freq[metric][0] + 1]

    labels = ['Metric']
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 1 * (width), move_on_metric_freq["latency"], width, label='perf moves', color="orange")
    rects2 = ax.bar(x, move_on_metric_freq["power"], width, label='power moves', color="mediumpurple")
    rects3 = ax.bar(x + 1 * width, move_on_metric_freq["area"], width, label='area moves', color="brown")
    ax.set_ylabel('frequency', fontsize=15)
    ax.set_title('Move frequency', fontsize=15)
    # ax.set_ylabel('Sim time (s)', fontsize=25)
    # ax.set_title('Sim time across system comoplexity.', fontsize=24)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=4)
    ax.legend(prop={'size': 15})
    fig.savefig(os.path.join(config.latest_visualization,"move_freq_breakdown.pdf"))


# ------------------------------
# Functionality:
#        visualize the sequence of moves made
# Variables:
#      des_trail_list: design trail, i.e., the list of designs made in the chronological order
#      move_profiles: list of moves made
#      des_per_iteration: number of designs tried per iteration (in each iteration, we can be looking at a host of designs
#                         depending on the depth and breadth of search at the point)
# ------------------------------
def des_trail_plot(des_trail_list, move_profile, des_per_iteration):
    metric_bounds = {}
    for metric in config.budgetted_metrics:
        metric_bounds[metric] = (+10000, -10000)
    metric_ref_des_dict = {}
    metric_trans_des_dict = {}
    for metric in config.budgetted_metrics:
       metric_ref_des_dict[metric]  = []
       metric_trans_des_dict[metric] = []

    # contains all the results
    res_list = []
    for ref_des, transformed_des in des_trail_list:
        ref_des_metrics = []
        transformed_des_metrics = []
        # get the metrics
        for metric in config.budgetted_metrics:
            #ref_des_metric_value = 100*(1 - ref_des.get_dp_stats().get_system_complex_metric(metric)/ref_des.database.get_budget(metric, "glass"))
            if isinstance(ref_des.get_dp_stats().get_system_complex_metric(metric), dict): # must be latency
                system_complex_metric = max(list(ref_des.get_dp_stats().get_system_complex_metric(metric).values()))
                system_complex_budget = max(list(ref_des.database.get_budget(metric, "glass").values()))
                ref_des_metric_value = 100 * (1 - system_complex_metric/system_complex_budget)
                trans_des_metric_value =  100 * (1 - system_complex_metric/system_complex_budget) - ref_des_metric_value# need to subtract since the second one needs to be magnitude
            else:
                ref_des_metric_value = 100*(1 - ref_des.get_dp_stats().get_system_complex_metric(metric)/ref_des.database.get_budget(metric, "glass"))
                trans_des_metric_value = \
                    100*(1 - transformed_des.get_dp_stats().get_system_complex_metric(metric)/
                         ref_des.database.get_budget(metric, "glass")) - ref_des_metric_value  # need to subtract since the second one needs to be magnitude

            ref_des_metrics.append(ref_des_metric_value)
            transformed_des_metrics.append(trans_des_metric_value)
            metric_bounds[metric] = (min(metric_bounds[metric][0], ref_des_metric_value, trans_des_metric_value),
                                  max(metric_bounds[metric][1], ref_des_metric_value, trans_des_metric_value))
            metric_ref_des_dict[metric].append(ref_des_metric_value)
            metric_trans_des_dict[metric].append((ref_des_metric_value + trans_des_metric_value))

        #res_list.append(copy.deepcopy(ref_des_metrics + transformed_des_metrics))
        res_list.append(cPickle.loads(cPickle.dumps(ref_des_metrics + transformed_des_metrics, -1)))

    # soa = np.array([[0, 0, 0, 1, 3, 1], [1,1,1, 3,3,3]])
    soa = np.array(res_list)

    des_per_iteration.append(len(res_list))
    des_iteration_unflattened = [(des_per_iteration[itr+1]-des_per_iteration[itr])*[itr+1] for itr,el in enumerate(des_per_iteration[:-1])]
    des_iteration = [j for sub in des_iteration_unflattened for j in sub]

    X, Y, Z, U, V, W = zip(*soa)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    c_ = list(range(0, len(X)))
    for id in range(0, len(c_)):
        c_[id] = str((1 - c_[id]/(len(X)))/2)
    #c_ = (list(range(0, len(X)))* float(1)/len(X))
    ax.quiver(X, Y, Z, U, V, W, arrow_length_ratio=.04, color=c_)
    ax.scatter(X[-1], Y[-1], Z[-1], c="red")
    ax.scatter(0, 0, 0, c="green")
    #ax.quiver(X, Y, Z, U, V, W)

    ax.set_xlim([-1 + metric_bounds[config.budgetted_metrics[0]][0], 1 + max(metric_bounds[config.budgetted_metrics[0]][1], 1)])
    ax.set_ylim([-1 + metric_bounds[config.budgetted_metrics[1]][0], 1 + max(1.1*metric_bounds[config.budgetted_metrics[1]][1], 1)])
    ax.set_zlim([-1 + metric_bounds[config.budgetted_metrics[2]][0], 1 + max(1.1*metric_bounds[config.budgetted_metrics[2]][1], 1)])
    #ax.set_xlim([0*metric_bounds[config.budgetted_metrics[0]][0], 5*metric_bounds[config.budgetted_metrics[0]][1]])
    #ax.set_ylim([0*metric_bounds[config.budgetted_metrics[1]][0], 5*metric_bounds[config.budgetted_metrics[1]][1]])
    #ax.set_zlim([0*metric_bounds[config.budgetted_metrics[2]][0], 5*metric_bounds[config.budgetted_metrics[2]][1]])
    ax.set_title("normalized distance to budget")
    ax.set_xlabel(config.budgetted_metrics[0])
    ax.set_ylabel(config.budgetted_metrics[1])
    ax.set_zlabel(config.budgetted_metrics[2])
    fig.savefig(os.path.join(config.latest_visualization,"DSE_trail.pdf"))

    des_iteration_move_markers = {}
    des_iteration_move_markers["latency"] = []
    des_iteration_move_markers["power"] = []
    des_iteration_move_markers["area"] = []
    des_iteration_move_markers["energy"] = []
    for move in move_profile:
        for metric in metric_ref_des_dict.keys() :
            if move.get_metric() == metric:
                des_iteration_move_markers[metric].append(1)
            else:
                des_iteration_move_markers[metric].append(2)
        if metric == "cost":
            print("ok")

    # proression per metric
    for metric in metric_ref_des_dict.keys():
        fig, ax = plt.subplots()
        ax.set_title("normalize distance to budget VS iteration")
        #blah = des_iteration[des_iteration_move_markers[metric]==1]
        #blah3 = metric_ref_des_dict[metric]
        #blah2 = blah3[des_iteration_move_markers[metric]==1]
        #ax.scatter(des_iteration, metric_ref_des_dict[metric][des_iteration_move_markers[metric]==1], color="red", label="orig des", marker="*")
        #ax.scatter(des_iteration[des_iteration_move_markers[metric]==2], metric_ref_des_dict[metric][des_iteration_move_markers[metric] == 2], color="red", label="orig des", marker=".")
        #ax.scatter(des_iteration[des_iteration_move_markers[metric] == 1], metric_trans_des_dict[metric][des_iteration_move_markers[metric]==1], color ="green", label="trans des", alpha=.05, marker="*")
        #ax.scatter(des_iteration[des_iteration_move_markers[metric] == 2], metric_trans_des_dict[metric][des_iteration_move_markers[metric]==2], color ="green", label="trans des", alpha=.05, marker=".")
        blah_ = [np.array(des_iteration_move_markers[metric])==1]
        blah = np.array(des_iteration)[np.array(des_iteration_move_markers[metric])==1]
        blah2 = np.array(metric_ref_des_dict[metric])[np.array(des_iteration_move_markers[metric])==1]
        ax.scatter(np.array(des_iteration)[np.array(des_iteration_move_markers[metric])==1], np.array(metric_ref_des_dict[metric])[np.array(des_iteration_move_markers[metric])==1], color="red", label="orig des", marker="*")
        ax.scatter(np.array(des_iteration)[np.array(des_iteration_move_markers[metric])==2], np.array(metric_ref_des_dict[metric])[np.array(des_iteration_move_markers[metric])==2], color="red", label="orig des", marker=".")
        ax.scatter(np.array(des_iteration)[np.array(des_iteration_move_markers[metric])==1], np.array(metric_trans_des_dict[metric])[np.array(des_iteration_move_markers[metric])==1], color="green", label="trans_des", marker="*", alpha=.05)
        ax.scatter(np.array(des_iteration)[np.array(des_iteration_move_markers[metric])==2], np.array(metric_trans_des_dict[metric])[np.array(des_iteration_move_markers[metric])==2], color="green", label="trans_des", marker=".", alpha=0.05)
        ax.legend()
        ax.set_xlabel("iteration count")
        ax.set_ylabel(metric + " norm dist to budget (%)")
        fig.savefig(os.path.join(config.latest_visualization,metric + "_distance_to_buddget_itr.png"))


    trans_des_dist_to_goal = []   # contains list of designs over all distance to goal
    for _, transformed_des in des_trail_list:
        trans_des_dist_to_goal.append(
            100*transformed_des.get_dp_stats().dist_to_goal(["latency", "power", "area"],"simple")/len(metric_ref_des_dict.keys()))

    fig, ax = plt.subplots()
    ax.set_title("normalize distance to all budgets VS iteration")
    ax.scatter(des_iteration, trans_des_dist_to_goal)
    #ax.legend()
    ax.set_xlabel("iteration count")
    ax.set_ylabel(metric + " norm dist to all budgets (%)")
    fig.savefig(os.path.join(config.latest_visualization,"avg_budgets_distance_itr.pdf"))


    plt.close('all')
    barplot_moves(move_profile)

# ------------------------------
# Functionality:
#       plot move stats
# ------------------------------
def barplot_moves(move_profile):
    # move_to_plot (only show the depth/breadth == 0) to simplicity purposes
    move_to_plot = []
    for move in move_profile:
        if config.regulate_move_tracking:
            if move.get_breadth() == 0 and move.get_depth() == 0 and move.get_mini_breadht() == 0 and move.is_valid():
                move_to_plot.append(move)
        elif move.is_valid():
            move_to_plot.append(move)

    #if (len(move_to_plot)+1) > 15:
    #    print("what")
    # draw the metrics
    metric_dict = {}
    metric_names = ["latency", "power", "area"]
    for metric_name in metric_names:
        metric_dict[metric_name] = []
    metric_dict["cost"] = []
    height_list = []
    for move_ in move_to_plot:
        # get metric values
        metrics = move_.get_logs("metrics")
        for metric_name in metric_names:
            metric_dict[metric_name].append(metrics[metric_name])
        selected_metric = move_.get_metric()

        # find the height that you'd like to mark to specify the metric of interest
        height = 0
        for metric in metric_names:
            if metric == selected_metric:
                height += metric_dict[metric][-1]/2
                height_list.append(height)
                break
            height += metric_dict[metric][-1]
        if selected_metric =="cost":
            height_list.append(height)
            metric_dict["cost"].append(1)
            for metric in metric_names:
                metric_dict[metric][-1] = 0
        else:
            metric_dict["cost"].append(0)
    labels = [str(i) for i in list(range(1, len(metric_dict["latency"])+1))]
    power_plus_area =  [metric_dict["latency"][i]+ metric_dict["power"][i] for i in range(len(labels))]
    power_plus_latency=  [metric_dict["latency"][i]+ metric_dict["power"][i] for i in range(len(labels))]
    power_plus_latency_plus_area=  [metric_dict["latency"][i]+ metric_dict["power"][i] for i in range(len(labels))]

    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars


    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    rects1 = ax.bar(x - .5*(width), metric_dict["latency"], width, label='perf', color="gold")
    rects2 = ax.bar(x - .5*(width), metric_dict["power"], width, bottom=metric_dict["latency"],
            label='power', color="orange")
    rects3 = ax.bar(x - .5*(width), metric_dict["area"], width, bottom=power_plus_latency,
            label='area', color="red")
    rects4 = ax.bar(x - .5*(width), metric_dict["cost"], width, bottom=power_plus_latency_plus_area,
            label='cost', color="purple")

    plt.plot(x, height_list, marker='o', linewidth=.3, color="red", ms=1)
    ax.set_ylabel('Metrics contribution (%)', fontsize=15)
    ax.set_xlabel('Iteration ', fontsize=15)
    ax.set_title('Metric Selection.', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=4)
    ax.legend(prop={'size': 15})
    plt.savefig(os.path.join(config.latest_visualization,"Metric_Selection"))
    plt.close('all')

    # draw the metrics with distance
    metric_dict = {}
    metric_names = ["latency", "power", "area"]
    for metric_name in metric_names:
        metric_dict[metric_name] = []
    metric_dict["cost"] = []
    height_list = []
    for move_ in move_to_plot:
        # get metric values
        dist_to_goal = move_.get_logs("ref_des_dist_to_goal_non_cost")
        metrics = move_.get_logs("metrics")
        for metric_name in metric_names:
            metric_dict[metric_name].append(metrics[metric_name]*dist_to_goal*100)
        selected_metric = move_.get_metric()

        # find the height that you'd like to mark to specify the metric of interest
        height = 0
        for metric in metric_names:
            if metric == selected_metric:
                height += metric_dict[metric][-1] / 2
                height_list.append(height)
                break
            height += metric_dict[metric][-1]
        if selected_metric == "cost":
            height_list.append(height)
            metric_dict["cost"].append(1*dist_to_goal*100)
            for metric in metric_names:
                metric_dict[metric][-1] = 0
        else:
            metric_dict["cost"].append(0)
    labels = [str(i) for i in list(range(1, len(metric_dict["latency"]) + 1))]
    power_plus_area = [metric_dict["latency"][i] + metric_dict["power"][i] for i in range(len(labels))]
    power_plus_latency = [metric_dict["latency"][i] + metric_dict["power"][i] for i in range(len(labels))]
    power_plus_latency_plus_area = [metric_dict["latency"][i] + metric_dict["power"][i] for i in range(len(labels))]

    x = np.arange(len(labels))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    rects1 = ax.bar(x - .5 * (width), metric_dict["latency"], width, label='perf', color="gold")
    rects2 = ax.bar(x - .5 * (width), metric_dict["power"], width, bottom=metric_dict["latency"],
                    label='power', color="orange")
    rects3 = ax.bar(x - .5 * (width), metric_dict["area"], width, bottom=power_plus_latency,
                    label='area', color="red")
    rects4 = ax.bar(x - .5 * (width), metric_dict["cost"], width, bottom=power_plus_latency_plus_area,
                    label='cost', color="purple")

    plt.yscale("log")
    plt.plot(x, height_list, marker='>', linewidth=.6, color="green", label="targeted metric", ms= 1)
    ax.set_ylabel('Distance to Goal(Budget) (%)', fontsize=15)
    ax.set_xlabel('Iteration ', fontsize=15)
    ax.set_title('Metric Selection', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=4)
    ax.legend(prop={'size': 13}, ncol = 4, bbox_to_anchor = (-0.3, 1.4), loc = 'upper left')
    fig.tight_layout()
    plt.savefig(os.path.join(config.latest_visualization,"Metric_Selection_with_distance"))
    plt.close('all')

    # plot kernels
    task_dict = {}
    task_names = [krnl.get_task().name for krnl in move_profile[0].get_logs("kernels")]

    for task_name in task_names:
        task_dict[task_name] = []
    #metric_dict["cost"] = []
    height_list = []
    for move_ in move_to_plot:
        # get metric values
        krnl_prob_dict = move_.get_logs("kernels")
        for krnl, value in krnl_prob_dict.items():
            task_dict[krnl.get_task().name].append(100*value)
        selected_krnl = move_.get_kernel_ref()

        # find the height that you'd like to mark to specify the metric of interest
        height = 0
        for task_name in task_names:
            if task_name == selected_krnl.get_task().name:
                height += task_dict[task_name][-1]/2
                height_list.append(height)
                break
            height += task_dict[task_name][-1]

        selected_metric = move_.get_metric()
        if selected_metric =="cost":
            for task in task_names:
                task_dict[task][-1] = 0
            height_list[-1] = 1
            task_dict[(move_.get_tasks()[0]).name][-1] = 100
        #if move_.dist_to_goal < .05:
        #    print("what done now")
        #else:
        #    task_dict["cost"].append(0)
    labels = [str(i) for i in list(range(1, len(task_dict[task_names[0]])+1))]
    """ 
    sum = 0
    for task in task_names:
        sum += task_dict[task][0]
    """
    try:
        x = np.arange(len(labels))  # the label locations
    except:
        print("what")
    width = 0.4  # the width of the bars


    #my_cmap = plt.get_cmap("Set3")
    my_cmap = ["bisque", "darkorange", "tan", "gold", "olive", "greenyellow", "darkgreen", "turquoise", "crimson",
               "lightblue", "yellow",
               "chocolate", "hotpink", "darkorchid"]
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    rects1 = ax.bar(x - .5*(width), task_dict[task_names[0]], width, label=task_names[0], color = my_cmap[0])
    prev_offset = len(x)*[0]
    rects = []
    prev_task = task_names[0]
    for idx, task_name in enumerate(task_names[1:]):
        for idx_, y in enumerate(task_dict[prev_task]):
            prev_offset[idx_] += y
        prev_task = task_name
        rects.append(ax.bar(x - .5*(width), task_dict[task_name], width, label=task_name, bottom=prev_offset, color = my_cmap[(idx+1)%len(my_cmap)]))

    plt.plot(x, height_list, marker='>', linewidth=1.5, color="green", label="Targeted Kernel", ms=1)

    ax.set_ylabel('Kernel Contribution (%)', fontsize=15)
    ax.set_xlabel('Iteration ', fontsize=15)
    ax.set_title('Kernel Selection', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=2)
    ax.legend(prop={'size': 9}, ncol=1, bbox_to_anchor = (1.01, 1), loc = 'upper left')
    fig.tight_layout()
    plt.savefig(os.path.join(config.latest_visualization,"Kernel_Selection"))
    plt.close('all')



    # plot blocks
    block_dict = {}
    block_names = ["pe", "mem", "ic"]

    for block_name in block_names:
        block_dict[block_name] = []
    #metric_dict["cost"] = []
    height_list = []
    for move_ in move_to_plot:
        selected_metric = move_.get_metric()
        # get metric values
        block_prob_dict = move_.get_logs("blocks")
        seen_blocks = []
        for block, value in block_prob_dict:
            if block.type in seen_blocks:  # can have multiple memory
                if selected_metric == "latency":
                    block_dict[block.type][-1] = max(value, block_dict[block.type][-1])
                else:
                    block_dict[block.type][-1] += value
            else:
                block_dict[block.type].append(value)
            seen_blocks.append(block.type)
        selected_block = move_.get_block_ref().type

        # find the height that you'd like to mark to specify the metric of interest
        height = 0
        seen_blocks = []
        for block in block_names:
            if block in seen_blocks:
                continue
            if block == selected_block:
                height += block_dict[block][-1]/2
                height_list.append(min(height, 100))
                break
            height += block_dict[block][-1]
            seen_blocks.append(block)

        if selected_metric =="cost":
            for block in block_names:
                block_dict[block][-1] = 0
            height_list[-1] = 1
            block_dict[move_.get_block_ref().type][-1] = 100

    #if selected_metric == "latency" and not(block_dict["pe"][-1] == 100):
    #    print("what")

    labels = [str(i) for i in list(range(1, len(block_dict[block_names[0]])+1))]

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    my_cmap = ["orange", "blue", "red"]

    block_name
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    rects1 = ax.bar(x - .5*(width), block_dict[block_names[0]], width, label=block_names[0].upper(), color = my_cmap[0])
    prev_offset = len(x)*[0]
    rects = []
    prev_block_name = block_names[0]
    for idx, block_name in enumerate(block_names[1:]):
        for idx_, y in enumerate(block_dict[prev_block_name]):
            prev_offset[idx_] += y
        if not (len(x) == len(block_dict[block_name])):
            ok = len(x)
            print("what")
        rects.append(ax.bar(x - .5 * (width), block_dict[block_name], width, label=block_name.upper(), bottom=prev_offset,
                        color=my_cmap[idx + 1]))
        prev_block_name = block_name
    plt.plot(x, height_list, marker='>', linewidth=.6, color="green", ms=1, label="Targeted Block")

    ax.set_ylabel('Block contribution (%)', fontsize=15)
    ax.set_xlabel('Iteration ', fontsize=15)
    ax.set_title('Block Selection', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=4)
    ax.legend(prop={'size': 10}, ncol=1, bbox_to_anchor = (1.01, 1), loc = 'upper left')

    fig.tight_layout()
    #plt.savefig("system_sim_error_diff_workload")
    plt.savefig(os.path.join(config.latest_visualization,"Block_Selection"))
    plt.close('all')


    # transformations
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    transformation_decoding = ["split", "migrate", "swap", "split_swap", "cleanup", "identity", "dram_fix", "transfer","routing"]
    y = []

    for move_ in move_to_plot:
        y.append(transformation_decoding.index(move_.get_transformation_name())+1)
    x = np.arange(len(y))  # the label locations
    plt.yticks(list(range(1,len(transformation_decoding)+1)), transformation_decoding, fontsize=15)
    ax.set_ylabel('Transformation ', fontsize=15)
    ax.set_xlabel('Iteration ', fontsize=15)
    ax.set_title('Transformation Selection', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=4)
    fig.tight_layout()
    plt.plot(list(range(0, len(y))), y, marker='o', linewidth=.6, color="green", label="Transformation", ms=1)
    ax.legend(prop={'size': 15})
    plt.savefig(os.path.join(config.latest_visualization,"Transformation_Selection"))
    plt.close('all')

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    y = []

    for move_ in move_to_plot:
        y.append(move_.get_logs("kernel_rnk_to_consider"))
    x = np.arange(len(y))  # the label locations
    ax.set_ylabel('Task Rank To Consider', fontsize=15)
    ax.set_xlabel('Iteration ', fontsize=15)
    ax.set_title('Kernel Rank to Consider', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=4)
    plt.plot(list(range(0, len(y))), y, marker='o', linewidth=.6, color="green", label="Kernel Rank", ms=1)
    ax.legend(prop={'size': 10})
    plt.savefig(os.path.join(config.latest_visualization,"Kernel_rank_selection"))
    plt.close('all')



    # distnace to goal
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    y = []
    for move_ in move_to_plot:
        y.append(100*move_.get_logs("ref_des_dist_to_goal_non_cost"))
    x = np.arange(len(y))  # the label locations
    ax.set_ylabel('(Normalized) Distance to Goal (%)', fontsize=15)
    ax.set_xlabel('Iteration ', fontsize=15)
    ax.set_title('Convergence to Goal', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=4)
    #plt.rcParams["figure.figsize"] = (10, 5)
    plt.plot(list(range(0, len(y))), y, marker='>', linewidth=.6, color="green", label="Distance", ms=1)
    ax.legend(prop={'size': 10})
    plt.savefig(os.path.join(config.latest_visualization,"distance_to_goal"))
    plt.close('all')


    # distnace to goal
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    y = []
    for move_ in move_to_plot:
        y.append(move_.get_logs("cost")/move_to_plot[0].get_logs("cost"))
    x = range(1, len(y)+1)  # the label locations
    ax.set_ylabel('Cost (Development and Silicon)', fontsize=15)
    ax.set_xlabel('Iteration ', fontsize=15)
    ax.set_title('Cost', fontsize=15)
    ax.set_xticks(x)
    #ax.set_xticklabels(labels, fontsize=4)
    #plt.rcParams["figure.figsize"] = (10, 5)
    plt.plot(list(range(1, len(y)+1)), y, marker='>', linewidth=.6, color="green", label="cost", ms=1)
    ax.legend(prop={'size': 10})
    plt.savefig(os.path.join(config.latest_visualization,"cost_"))
    plt.close('all')


def scatter_plot(x, y, axis_name, database):
    fig, ax = plt.subplots()
    ax.scatter(x, y, marker="x")
    ax.set_xlabel(axis_name[0] + " count")
    if (config.DATA_DELIVEYRY== "obfuscate"):
        axis_name_ =  "normalized " + axis_name[1]+ " (normalized to simple single core design)"
    else:
        axis_name_ =  axis_name[1] + " ("+config.axis_unit[axis_name[1]] +")"
    ax.set_ylabel(axis_name_)
    title = ""
    title = " with constraints"

    title += " \nand " + config.migrant_clustering_policy + " migration policy"
    ax.set_title(axis_name[0] + " V.S. " +  axis_name[1])
    fig.savefig(os.path.join(config.latest_visualization,axis_name[0]+"_"+axis_name[1]+".png"))