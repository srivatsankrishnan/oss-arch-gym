#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
import math
import itertools
import copy
import csv
import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import shutil
from settings import config_plotting
import time
import re
from collections import OrderedDict
from pygmo import *
def get_column_name_number(dir_addr, mode):
    column_name_number_dic = {}
    try:
        if mode == "all":
            file_name = "result_summary/FARSI_simple_run_0_1_all_reults.csv"
        elif mode =="aggregate":
            file_name = "result_summary/aggregate_all_results.csv"
        else:
            file_name = "result_summary/FARSI_simple_run_0_1.csv"

        file_full_addr = os.path.join(dir_addr, file_name)
        with open(file_full_addr) as f:
            resultReader = csv.reader(f, delimiter=',', quotechar='|')
            for row in resultReader:
                for idx, el_name in enumerate(row):
                    column_name_number_dic[el_name] = idx
                break
        return column_name_number_dic
    except Exception as e:
        raise e



#


# the function to get the column information of the given category
def columnNum(dirName, fileName, cate, result):
    if result == "all":
        with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
            resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

            for i, row in enumerate(resultReader):
                if i == 0:
                    for j in range(0, len(row)):
                        if row[j] == cate:
                            return j
                    raise Exception("No such category in the list! Check the name: " + cate)
                break
    elif result == "simple":
        with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1.csv", newline='') as csvfile:
            resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

            for i, row in enumerate(resultReader):
                if i == 0:
                    for j in range(0, len(row)):
                        if row[j] == cate:
                            return j
                    raise Exception("No such category in the list! Check the name: " + cate)
                break
    else:
        raise Exception("No such result file! Check the result type! It should be either \"all\" or \"simple\"")

# the function to plot the frequency of all comm_comp in the pie chart
def plotCommCompAll(dirName, fileName, all_res_column_name_number):
    colNum = all_res_column_name_number["comm_comp"]
    truNum = all_res_column_name_number["move validity"]

    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        commNum = 0
        compNum = 0

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i > 1:
                if row[colNum] == "comm":
                    commNum += 1
                elif row[colNum] == "comp":
                    compNum += 1
                else:
                    raise Exception("comm_comp is not giving comm or comp! The new type: " + row[colNum])

        plt.figure()
        plt.pie([commNum, compNum], labels = ["comm", "comp"])
        plt.title("comm_comp: Frequency")
        plt.savefig(dirName + fileName + "/comm-compFreq-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot the frequency of all high level optimizations in the pie chart
def plothighLevelOptAll(dirName, fileName, all_res_column_name_number):
    colNum = all_res_column_name_number["high level optimization name"]
    truNum = all_res_column_name_number["move validity"]

    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        topoNum = 0
        tunNum = 0
        mapNum = 0
        idenOptNum = 0

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i > 1:
                if row[colNum] == "topology":
                    topoNum += 1
                elif row[colNum] == "customization":
                    tunNum += 1
                elif row[colNum] == "mapping":
                    mapNum += 1
                elif row[colNum] == "identity":
                    idenOptNum += 1
                else:
                    raise Exception("high level optimization name is not giving topology or customization or mapping or identity! The new type: " + row[colNum])
        
        plt.figure()
        plt.pie([topoNum, tunNum, mapNum, idenOptNum], labels = ["topology", "customization", "mapping", "identity"])
        plt.title("High Level Optimization: Frequency")
        plt.savefig(dirName + fileName + "/highLevelOpt-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot the frequency of all architectural variables to improve in the pie chart
def plotArchVarImpAll(dirName, fileName, colNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        parazNum = 0
        custNum = 0
        localNum = 0
        idenImpNum = 0

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i > 1:
                if row[colNum] == "parallelization":
                    parazNum += 1
                elif row[colNum] == "customization":
                    custNum += 1
                elif row[colNum] == "locality":
                    localNum += 1
                elif row[colNum] == "identity":
                    idenImpNum += 1                
                else:
                    raise Exception("architectural principle is not parallelization or customization or locality or identity! The new type: " + row[colNum])

        plt.figure()
        plt.pie([parazNum, custNum, localNum, idenImpNum], labels = ["parallelization", "customization", "locality", "identity"])
        plt.title("Architectural Principle: Frequency")
        plt.savefig(dirName + fileName + "/archVarImp-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot simulation time vs. system block count
def plotSimTimeVSblk(dirName, fileName, blkColNum, simColNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        sysBlkCount = []
        simTime = []

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i > 1:
                sysBlkCount.append(int(row[blkColNum]))
                simTime.append(float(row[simColNum]))

        plt.figure()
        plt.plot(sysBlkCount, simTime)
        plt.xlabel("System Block Count")
        plt.ylabel("Simulation Time")
        plt.title("Simulation Time vs. Sytem Block Count")
        plt.savefig(dirName + fileName + "/simTimeVSblk-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot move generation time vs. system block count
def plotMoveGenTimeVSblk(dirName, fileName, blkColNum, movColNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        sysBlkCount = []
        moveGenTime = []

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i > 1:
                sysBlkCount.append(int(row[blkColNum]))
                moveGenTime.append(float(row[movColNum]))
        
        plt.figure()
        plt.plot(sysBlkCount, moveGenTime)
        plt.xlabel("System Block Count")
        plt.ylabel("Move Generation Time")
        plt.title("Move Generation Time vs. System Block Count")
        plt.savefig(dirName + fileName + "/moveGenTimeVSblk-" + fileName + ".png")
        # plt.show()
        plt.close('all')

def get_experiments_workload(all_res_column_name):
    latency_budget =  all_res_column_name_number["latency budget"][:-1]
    workload_latency = latency_budget.split(";")
    workloads = []
    for workload_latency in workload_latency:
        workloads.append(workload_latency.split("=")[0])
    return workloads

def get_experiments_name(file_full_addr, all_res_column_name_number):
    with open(file_full_addr, newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        row1 = next(resultReader)
        row2 = next(resultReader)
        latency_budget =  row2[all_res_column_name_number["latency_budget"]]
        power_budget =  row2[all_res_column_name_number["power_budget"]]
        area_budget =  row2[all_res_column_name_number["area_budget"]]
        try:
            transformation_selection_mode =  row2[all_res_column_name_number["transformation_selection_mode"]]
        except:
            transformation_selection_mode =  ""


        workload_latency = latency_budget[:-1].split(';')
        latency_budget_refined =""
        for workload_latency in workload_latency:
            latency_budget_refined +="_" + (workload_latency.split("=")[0][0]+workload_latency.split("=")[1])

        return latency_budget_refined[1:]+"_" + power_budget + "_" + area_budget+"_"+transformation_selection_mode

def get_all_col_values_of_a_file(file_full_addr, all_res_column_name_number, column_name):
    column_number = all_res_column_name_number[column_name]
    all_values = []
    with open(file_full_addr, newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        experiment_name = get_experiments_name(file_full_addr, all_res_column_name_number)
        for i, row in enumerate(resultReader):
            if i > 1:
                if not row[column_number] == '':
                    value =row[column_number]
                    values = value.split(";") # if mutiple values
                    for val in values:
                        if "=" in val:
                            val_splitted = val.split("=")
                            all_values.append(val_splitted[0])
                        else:
                            all_values.append(val)

    return all_values

def get_all_col_values_of_a_folders(input_dir_names, input_all_res_column_name_number, column_name):
    all_values = []
    for dir_name in input_dir_names:
        file_full_addr = os.path.join(dir_name, "result_summary/FARSI_simple_run_0_1_all_reults.csv")
        all_values.extend(get_all_col_values_of_a_file(file_full_addr, input_all_res_column_name_number, column_name))

    # get rid of duplicates
    all_values_rid_of_duplicates = list(set(all_values))
    return all_values_rid_of_duplicates

def extract_latency_values(values_):
    print("")


def plot_codesign_rate_efficacy_cross_workloads_updated(input_dir_names, res_column_name_number):
    #itrColNum = all_res_column_name_number["iteration cnt"]
    #distColNum = all_res_column_name_number["dist_to_goal_non_cost"]
    trueNum  =  all_res_column_name_number["move validity"]
    move_name_number =  all_res_column_name_number["move name"]

    # experiment_names
    file_full_addr_list = []
    for dir_name in input_dir_names:
        file_full_addr = os.path.join(dir_name, "result_summary/FARSI_simple_run_0_1_all_reults.csv")
        file_full_addr_list.append(file_full_addr)

    axis_font = {'fontname': 'Arial', 'size': '4'}
    x_column_name = "iteration cnt"
    #y_column_name_list = ["high level optimization name", "exact optimization name", "architectural principle", "comm_comp"]
    y_column_name_list = ["exact optimization name",  "architectural principle", "comm_comp", "workload"]

    #y_column_name_list = ["high level optimization name", "exact optimization name", "architectural principle", "comm_comp"]



    column_co_design_cnt = {}
    column_non_co_design_cnt = {}
    column_co_design_rate = {}
    column_non_co_design_rate = {}
    column_co_design_efficacy_avg = {}
    column_non_co_design_efficacy_rate = {}
    column_non_co_design_efficacy = {}
    column_co_design_dist= {}
    column_co_design_dist_avg= {}
    column_co_design_improvement = {}
    experiment_name_list = []
    last_col_val = ""
    for file_full_addr in file_full_addr_list:
        experiment_name = get_experiments_name(file_full_addr, res_column_name_number)
        experiment_name_list.append(experiment_name)
        column_co_design_dist_avg[experiment_name] = {}
        column_co_design_efficacy_avg[experiment_name] = {}

        column_co_design_cnt = {}
        for y_column_name in y_column_name_list:
            y_column_number = res_column_name_number[y_column_name]
            x_column_number = res_column_name_number[x_column_name]


            dis_to_goal_column_number = res_column_name_number["dist_to_goal_non_cost"]
            ref_des_dis_to_goal_column_number = res_column_name_number["ref_des_dist_to_goal_non_cost"]
            column_co_design_cnt[y_column_name] = []
            column_non_co_design_cnt[y_column_name] = []

            column_non_co_design_efficacy[y_column_name] = []
            column_co_design_dist[y_column_name] = []
            column_co_design_improvement[y_column_name] = []
            column_co_design_rate[y_column_name] = []

            all_values = get_all_col_values_of_a_folders(input_dir_names, all_res_column_name_number, y_column_name)

            last_row_change = ""
            with open(file_full_addr, newline='') as csvfile:
                resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
                rows = list(resultReader)
                for i, row in enumerate(rows):
                    if i > 1:
                        last_row = rows[i - 1]
                        if row[y_column_number] not in all_values or row[move_name_number]=="identity":
                            continue

                        col_value = row[y_column_number]
                        col_values = col_value.split(";")
                        for idx, col_val in enumerate(col_values):


                            # only for improvement
                            if float(row[ref_des_dis_to_goal_column_number]) - float(row[dis_to_goal_column_number]) < 0:
                                continue

                            try:
                                delta_x_column = (float(row[x_column_number]) - float(last_row[x_column_number]))/len(col_values)
                            except:
                                pass

                            delta_improvement = (float(last_row[dis_to_goal_column_number]) - float(row[dis_to_goal_column_number]))/(float(last_row[dis_to_goal_column_number])*len(col_values))


                            if not col_val == last_col_val and i > 1:
                                if not last_row_change == "":
                                    distance_from_last_change =  float(last_row[x_column_number]) - float(last_row_change[x_column_number]) + idx * delta_x_column
                                    column_co_design_dist[y_column_name].append(distance_from_last_change)
                                    improvement_from_last_change =  (float(last_row[dis_to_goal_column_number]) - float(row[dis_to_goal_column_number]))/float(last_row[dis_to_goal_column_number])  + idx *delta_improvement
                                    column_co_design_improvement[y_column_name].append(improvement_from_last_change)

                                last_row_change = copy.deepcopy(last_row)


                            last_col_val = col_val



            # co_des cnt
            # we ignore the first element as the first element distance is always zero
            co_design_dist_sum = 0
            co_design_efficacy_sum = 0
            avg_ctr = 1
            co_design_dist_selected = column_co_design_dist[y_column_name]
            co_design_improvement_selected = column_co_design_improvement[y_column_name]
            for idx,el in enumerate(column_co_design_dist[y_column_name]):
                if idx == len(co_design_dist_selected) - 1:
                    break
                co_design_dist_sum += 1/(column_co_design_dist[y_column_name][idx] + column_co_design_dist[y_column_name][idx+1])
                co_design_efficacy_sum += (column_co_design_improvement[y_column_name][idx] + column_co_design_improvement[y_column_name][idx+1])
                #/(column_co_design_dist[y_column_name][idx] + column_co_design_dist[y_column_name][idx+1])
                avg_ctr+=1

            column_co_design_improvement = {}
            column_co_design_dist_avg[experiment_name][y_column_name]= co_design_dist_sum/avg_ctr
            column_co_design_efficacy_avg[experiment_name][y_column_name] = co_design_efficacy_sum/avg_ctr

        #result = {"rate":{}, "efficacy":{}}
        #rate_column_co_design = {}

    plt.figure()
    plotdata = pd.DataFrame(column_co_design_dist_avg, index=y_column_name_list)
    #plotdata_2 = pd.DataFrame(column_co_design_efficacy_avg, index=y_column_name_list)
    fontSize = 10
    plotdata.plot(kind='bar', fontsize=fontSize)
    #plotdata_2.plot(kind='bar', fontsize=fontSize)
    plt.xticks(fontsize=fontSize, rotation=6)
    plt.yticks(fontsize=fontSize)
    plt.xlabel("co design parameter", fontsize=fontSize)
    plt.ylabel("co design distance", fontsize=fontSize)
    plt.title("co desgin distance of different parameters",  fontsize=fontSize)

    # dump in the top folder
    output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
    output_dir = os.path.join(output_base_dir, "cross_workloads/co_design_rate")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir,"_".join(experiment_name_list[0]) +"_"+"co_design_avg_dist"+'_'.join(y_column_name_list)+".png"))
    plt.close('all')


    plt.figure()
    plotdata = pd.DataFrame(column_co_design_efficacy_avg, index=y_column_name_list)
    fontSize = 10
    plotdata.plot(kind='bar', fontsize=fontSize)
    plt.xticks(fontsize=fontSize, rotation=6)
    plt.yticks(fontsize=fontSize)
    plt.xlabel("co design parameter", fontsize=fontSize)
    plt.ylabel("co design dis", fontsize=fontSize)
    plt.title("co desgin efficacy of different parameters",  fontsize=fontSize)

    # dump in the top folder
    output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
    output_dir = os.path.join(output_base_dir, "cross_workloads/co_design_rate")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir,"_".join(experiment_name_list) +"_"+"co_design_efficacy"+'_'.join(y_column_name_list)+".png"))
    plt.close('all')

def plot_codesign_rate_efficacy_cross_workloads_updated_for_paper(input_dir_names, res_column_name_number):
    #itrColNum = all_res_column_name_number["iteration cnt"]
    #distColNum = all_res_column_name_number["dist_to_goal_non_cost"]
    trueNum  =  all_res_column_name_number["move validity"]
    move_name_number =  all_res_column_name_number["move name"]

    # experiment_names
    file_full_addr_list = []
    for dir_name in input_dir_names:
        file_full_addr = os.path.join(dir_name, "result_summary/FARSI_simple_run_0_1_all_reults.csv")
        file_full_addr_list.append(file_full_addr)

    axis_font = {'fontname': 'Arial', 'size': '4'}
    x_column_name = "iteration cnt"
    #y_column_name_list = ["high level optimization name", "exact optimization name", "architectural principle", "comm_comp"]
    y_column_name_list = ["exact optimization name",  "architectural principle", "comm_comp", "workload"]

    #y_column_name_list = ["high level optimization name", "exact optimization name", "architectural principle", "comm_comp"]



    column_co_design_cnt = {}
    column_non_co_design_cnt = {}
    column_co_design_rate = {}
    column_non_co_design_rate = {}
    column_co_design_efficacy_avg = {}
    column_non_co_design_efficacy_rate = {}
    column_non_co_design_efficacy = {}
    column_co_design_dist= {}
    column_co_design_dist_avg= {}
    column_co_design_improvement = {}
    experiment_name_list = []
    last_col_val = ""
    ctr_ = 0
    for file_full_addr in file_full_addr_list:
        if ctr_ == 1:
            continue
        experiment_name = get_experiments_name(file_full_addr, res_column_name_number)
        experiment_name_list.append(experiment_name)
        column_co_design_dist_avg[experiment_name] = {}
        column_co_design_efficacy_avg[experiment_name] = {}

        column_co_design_cnt = {}
        for y_column_name in y_column_name_list:
            y_column_number = res_column_name_number[y_column_name]
            x_column_number = res_column_name_number[x_column_name]


            dis_to_goal_column_number = res_column_name_number["dist_to_goal_non_cost"]
            ref_des_dis_to_goal_column_number = res_column_name_number["ref_des_dist_to_goal_non_cost"]
            column_co_design_cnt[y_column_name] = []
            column_non_co_design_cnt[y_column_name] = []

            column_non_co_design_efficacy[y_column_name] = []
            column_co_design_dist[y_column_name] = []
            column_co_design_improvement[y_column_name] = []
            column_co_design_rate[y_column_name] = []

            all_values = get_all_col_values_of_a_folders(input_dir_names, all_res_column_name_number, y_column_name)

            last_row_change = ""
            with open(file_full_addr, newline='') as csvfile:
                resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
                rows = list(resultReader)
                for i, row in enumerate(rows):
                    if i > 1:
                        last_row = rows[i - 1]
                        if row[y_column_number] not in all_values or row[move_name_number]=="identity":
                            continue

                        col_value = row[y_column_number]
                        col_values = col_value.split(";")
                        for idx, col_val in enumerate(col_values):


                            # only for improvement
                            if float(row[ref_des_dis_to_goal_column_number]) - float(row[dis_to_goal_column_number]) < 0:
                                continue

                            delta_x_column = (float(row[x_column_number]) - float(last_row[x_column_number]))/len(col_values)
                            delta_improvement = (float(last_row[dis_to_goal_column_number]) - float(row[dis_to_goal_column_number]))/(float(last_row[dis_to_goal_column_number])*len(col_values))


                            if not col_val == last_col_val and i > 1:
                                if not last_row_change == "":
                                    distance_from_last_change =  float(last_row[x_column_number]) - float(last_row_change[x_column_number]) + idx * delta_x_column
                                    column_co_design_dist[y_column_name].append(distance_from_last_change)
                                    improvement_from_last_change =  (float(last_row[dis_to_goal_column_number]) - float(row[dis_to_goal_column_number]))/float(last_row[dis_to_goal_column_number])  + idx *delta_improvement
                                    column_co_design_improvement[y_column_name].append(improvement_from_last_change)

                                last_row_change = copy.deepcopy(last_row)


                            last_col_val = col_val



            # co_des cnt
            # we ignore the first element as the first element distance is always zero
            co_design_dist_sum = 0
            co_design_efficacy_sum = 0
            avg_ctr = 1
            co_design_dist_selected = column_co_design_dist[y_column_name]
            co_design_improvement_selected = column_co_design_improvement[y_column_name]
            for idx,el in enumerate(column_co_design_dist[y_column_name]):
                if idx == len(co_design_dist_selected) - 1:
                    break
                co_design_dist_sum += 1/(column_co_design_dist[y_column_name][idx] + column_co_design_dist[y_column_name][idx+1])
                co_design_efficacy_sum += (column_co_design_improvement[y_column_name][idx] + column_co_design_improvement[y_column_name][idx+1])
                #/(column_co_design_dist[y_column_name][idx] + column_co_design_dist[y_column_name][idx+1])
                avg_ctr+=1

            column_co_design_improvement = {}
            column_co_design_dist_avg[experiment_name][y_column_name]= co_design_dist_sum/avg_ctr
            column_co_design_efficacy_avg[experiment_name][y_column_name] = co_design_efficacy_sum/avg_ctr
        ctr_ +=1
        #result = {"rate":{}, "efficacy":{}}
        #rate_column_co_design = {}

    plt.figure()
    y_column_name_list_rep = ["L.Opt",  "H.Opt", "CM", "WL"]
    y_column_name_list_rep_rep = [re.sub("(.{5})", "\\1\n", label, 0, re.DOTALL) for label in y_column_name_list_rep]
    plotdata = pd.DataFrame(column_co_design_dist_avg, index=y_column_name_list)
    fontSize = 26
    # print(plotdata)
    # print(y_column_name_list)
    color_list={'a0.021_e0.034_h0.034_0.008737_1.7475e-05_arch-aware': 'green', 'a0.021_e0.034_h0.034_0.008737_1.7475e-05_random': 'red'} # Ying: uncomment for blind_study_all_dumb_versions/blind_vs_arch_aware (T.B.M), also the one below
    ax = plotdata.plot(kind='bar', fontsize=fontSize, figsize=(6.6, 6.6), color=color_list)   # Ying: uncomment for blind_study_all_dumb_versions/blind_vs_arch_aware (T.B.M), also the one below
    ax.set_xticklabels(y_column_name_list_rep_rep, rotation=0)
    # Ying: hardcode here
    ax.set_xlabel("Co-design Parameter", fontsize=fontSize, labelpad=-25)
    ax.set_ylabel("Co-design Rate", fontsize=fontSize)
    for experiment_name, value in column_co_design_dist_avg.items():
        # print(experiment_name[-6:])
        if experiment_name[-6:] == "random":
            ax.legend(['SA', 'FARSI'], bbox_to_anchor=(0.5, 1.23), loc="upper center", fontsize=fontSize - 2, ncol=2)
        else:
            ax.legend(['FARSI', 'SA'], bbox_to_anchor=(0.5, 1.23), loc="upper center", fontsize=fontSize - 2, ncol=2)
        break
    # Ying: hardcode finished
    plt.tight_layout()

    # dump in the top folder
    output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
    output_dir = os.path.join(output_base_dir, "cross_workloads/co_design_rate")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir,"co_design_avg_dist"+'_'.join(y_column_name_list)+".png"), bbox_inches='tight')
    # plt.show()
    plt.close('all')


    plt.figure()
    plotdata = pd.DataFrame(column_co_design_efficacy_avg, index=y_column_name_list)
    fontSize = 26
    ax = plotdata.plot(kind='bar', fontsize=fontSize, figsize=(6.6, 6.6), color=color_list)   # Ying: uncomment for blind_study_all_dumb_versions/blind_vs_arch_aware (T.B.M), also the one above
    ax.set_xticklabels(y_column_name_list_rep_rep, rotation=0)
    # Ying: hardcode here
    ax.set_xlabel("Co-design Parameter", fontsize=fontSize, labelpad=-25)
    ax.set_ylabel("Co-design Improvement", fontsize=fontSize)
    for experiment_name, value in column_co_design_dist_avg.items():
        # print(experiment_name[-6:])
        if experiment_name[-6:] == "random":
            ax.legend(['SA', 'FARSI'], bbox_to_anchor=(0.5, 1.23), loc="upper center", fontsize=fontSize - 2, ncol=2)
        else:
            ax.legend(['FARSI', 'SA'], bbox_to_anchor=(0.5, 1.23), loc="upper center", fontsize=fontSize - 2, ncol=2)
        break
    # Ying: hardcode finished
    plt.tight_layout()

    # dump in the top folder
    output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
    output_dir = os.path.join(output_base_dir, "cross_workloads/co_design_rate")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir,"co_design_efficacy"+'_'.join(y_column_name_list)+".png"), bbox_inches='tight')
    # plt.show()
    plt.close('all')

"""
def plot_codesign_rate_efficacy_per_workloads(input_dir_names, res_column_name_number):
    #itrColNum = all_res_column_name_number["iteration cnt"]
    #distColNum = all_res_column_name_number["dist_to_goal_non_cost"]
    trueNum  =  all_res_column_name_number["move validity"]
    move_name_number =  all_res_column_name_number["move name"]

    # experiment_names
    file_full_addr_list = []
    for dir_name in input_dir_names:
        file_full_addr = os.path.join(dir_name, "result_summary/FARSI_simple_run_0_1_all_reults.csv")
        file_full_addr_list.append(file_full_addr)

    axis_font = {'fontname': 'Arial', 'size': '4'}
    x_column_name = "iteration cnt"
    #y_column_name_list = ["high level optimization name", "exact optimization name", "architectural principle", "comm_comp"]
    y_column_name_list = ["exact optimization name",  "architectural principle", "comm_comp", "workload"]

    #y_column_name_list = ["high level optimization name", "exact optimization name", "architectural principle", "comm_comp"]



    column_co_design_cnt = {}
    column_non_co_design_cnt = {}
    column_co_design_rate = {}
    column_non_co_design_rate = {}
    column_co_design_efficacy_rate = {}
    column_non_co_design_efficacy_rate = {}
    column_non_co_design_efficacy = {}
    column_co_design_efficacy= {}
    last_col_val = ""
    for file_full_addr in file_full_addr_list:
        experiment_name = get_experiments_name(file_full_addr, res_column_name_number)
        column_co_design_cnt = {}
        for y_column_name in y_column_name_list:
            y_column_number = res_column_name_number[y_column_name]
            x_column_number = res_column_name_number[x_column_name]


            dis_to_goal_column_number = res_column_name_number["dist_to_goal_non_cost"]
            ref_des_dis_to_goal_column_number = res_column_name_number["ref_des_dist_to_goal_non_cost"]
            column_co_design_cnt[y_column_name] = []
            column_non_co_design_cnt[y_column_name] = []

            column_non_co_design_efficacy[y_column_name] = []
            column_co_design_efficacy[y_column_name] = []

            all_values = get_all_col_values_of_a_folders(input_dir_names, all_res_column_name_number, y_column_name)

            with open(file_full_addr, newline='') as csvfile:
                resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
                rows = list(resultReader)
                for i, row in enumerate(rows):
                    if i >= 1:
                        last_row = rows[i - 1]
                        if row[y_column_number] not in all_values or row[trueNum] == "False" or row[move_name_number]=="identity":
                            continue

                        col_value = row[y_column_number]
                        col_values = col_value.split(";")
                        for idx, col_val in enumerate(col_values):
                            delta_x_column = (float(row[x_column_number]) - float(last_row[x_column_number]))/len(col_values)

                            value_to_add_1 = (float(last_row[x_column_number]) + idx * delta_x_column, 1)
                            value_to_add_0 = (float(last_row[x_column_number]) + idx * delta_x_column, 0)

                            # only for improvement
                            if float(row[ref_des_dis_to_goal_column_number]) - float(row[dis_to_goal_column_number]) < 0:
                                continue

                            if not col_val == last_col_val:

                                column_co_design_cnt[y_column_name].append(value_to_add_1)
                                column_non_co_design_cnt[y_column_name].append(value_to_add_0)
                                column_co_design_efficacy[y_column_name].append((float(row[ref_des_dis_to_goal_column_number]) - float(row[dis_to_goal_column_number]))/float(row[ref_des_dis_to_goal_column_number]))
                                column_non_co_design_efficacy[y_column_name].append(0)
                            else:
                                column_co_design_cnt[y_column_name].append(value_to_add_0)
                                column_non_co_design_cnt[y_column_name].append(value_to_add_1)
                                column_co_design_efficacy[y_column_name].append(0)
                                column_non_co_design_efficacy[y_column_name].append((float(row[ref_des_dis_to_goal_column_number]) - float(row[dis_to_goal_column_number]))/float(row[ref_des_dis_to_goal_column_number]))

                            last_col_val = col_val



            # co_des cnt
            x_values_co_design_cnt = [el[0] for el in column_co_design_cnt[y_column_name]]
            y_values_co_design_cnt = [el[1] for el in column_co_design_cnt[y_column_name]]
            y_values_co_design_cnt_total =sum(y_values_co_design_cnt)
            total_iter = x_values_co_design_cnt[-1]

            # non co_des cnt
            x_values_non_co_design_cnt = [el[0] for el in column_non_co_design_cnt[y_column_name]]
            y_values_non_co_design_cnt = [el[1] for el in column_non_co_design_cnt[y_column_name]]
            y_values_non_co_design_cnt_total =sum(y_values_non_co_design_cnt)

            column_co_design_rate[y_column_name] = y_values_co_design_cnt_total/total_iter
            column_non_co_design_rate[y_column_name] = y_values_non_co_design_cnt_total/total_iter

            # co_des efficacy
            y_values_co_design_efficacy =  column_co_design_efficacy[y_column_name]
            y_values_co_design_efficacy_total =sum(y_values_co_design_efficacy)


            # non co_des efficacy
            y_values_non_co_design_efficacy = column_non_co_design_efficacy[y_column_name]
            y_values_non_co_design_efficacy_total =sum(y_values_non_co_design_efficacy)

            column_co_design_efficacy_rate[y_column_name] = y_values_co_design_efficacy_total/(y_values_non_co_design_efficacy_total + y_values_co_design_efficacy_total)
            column_non_co_design_efficacy_rate[y_column_name] = y_values_non_co_design_efficacy_total/(y_values_non_co_design_efficacy_total + y_values_co_design_efficacy_total)


        result = {"rate":{}, "efficacy":{}}
        rate_column_co_design = {}

        result["rate"] =  {"co_design":column_co_design_rate, "non_co_design": column_non_co_design_rate}
        result["efficacy_rate"] =  {"co_design":column_co_design_efficacy_rate, "non_co_design": column_non_co_design_efficacy_rate}
        # prepare for plotting and plot


        plt.figure()
        plotdata = pd.DataFrame(result["rate"], index=y_column_name_list)
        fontSize = 10
        plotdata.plot(kind='bar', fontsize=fontSize, stacked=True)
        plt.xticks(fontsize=fontSize, rotation=6)
        plt.yticks(fontsize=fontSize)
        plt.xlabel("co design parameter", fontsize=fontSize)
        plt.ylabel("co design rate", fontsize=fontSize)
        plt.title("co desgin rate of different parameters",  fontsize=fontSize)

        # dump in the top folder
        output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
        output_dir = os.path.join(output_base_dir, "single_workload/co_design_rate")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.savefig(os.path.join(output_dir,experiment_name +"_"+"co_design_rate_"+'_'.join(y_column_name_list)+".png"))
        plt.close('all')


        plt.figure()
        plotdata = pd.DataFrame(result["efficacy_rate"], index=y_column_name_list)
        fontSize = 10
        plotdata.plot(kind='bar', fontsize=fontSize, stacked=True)
        plt.xticks(fontsize=fontSize, rotation=6)
        plt.yticks(fontsize=fontSize)
        plt.xlabel("co design parameter", fontsize=fontSize)
        plt.ylabel("co design efficacy rate", fontsize=fontSize)
        plt.title("co design efficacy rate of different parameters", fontsize=fontSize)

        # dump in the top folder
        output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
        output_dir = os.path.join(output_base_dir, "single_workload/co_design_rate")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.savefig(os.path.join(output_dir,experiment_name+"_"+"co_design_efficacy_rate_"+'_'.join(y_column_name_list)+".png"))
        plt.close('all')
"""


def plot_codesign_progression_per_workloads(input_dir_names, res_column_name_number):
    #itrColNum = all_res_column_name_number["iteration cnt"]
    #distColNum = all_res_column_name_number["dist_to_goal_non_cost"]
    trueNum  =  all_res_column_name_number["move validity"]

    # experiment_names
    experiment_names = []
    file_full_addr_list = []
    for dir_name in input_dir_names:
        file_full_addr = os.path.join(dir_name, "result_summary/FARSI_simple_run_0_1_all_reults.csv")
        file_full_addr_list.append(file_full_addr)
        experiment_name = get_experiments_name(file_full_addr, res_column_name_number)
        experiment_names.append(experiment_name)

    axis_font = {'size': '20'}
    x_column_name = "iteration cnt"
    y_column_name_list = ["high level optimization name", "exact optimization name", "architectural principle", "comm_comp"]


    experiment_column_value = {}
    for file_full_addr in file_full_addr_list:
        experiment_name = get_experiments_name(file_full_addr, res_column_name_number)
        for y_column_name in y_column_name_list:
            y_column_number = res_column_name_number[y_column_name]
            x_column_number = res_column_name_number[x_column_name]
            experiment_column_value[experiment_name] = []
            all_values = get_all_col_values_of_a_folders(input_dir_names, all_res_column_name_number, y_column_name)
            all_values_encoding = {}
            for idx, val in enumerate(all_values):
                all_values_encoding[val] = idx

            with open(file_full_addr, newline='') as csvfile:
                resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
                rows = list(resultReader)
                for i, row in enumerate(rows):
                    #if row[trueNum] != "True":
                    #    continue
                    if i >= 1:
                        if row[y_column_number] not in all_values:
                            continue

                        col_value = row[y_column_number]
                        col_values = col_value.split(";")
                        for idx, col_val in enumerate(col_values):
                            last_row =  rows[i-1]
                            delta_x_column = (float(row[x_column_number]) - float(last_row[x_column_number]))/len(col_values)
                            value_to_add = (float(last_row[x_column_number])+ idx*delta_x_column, col_val)
                            experiment_column_value[experiment_name].append(value_to_add)



            # prepare for plotting and plot
            axis_font = {'size': '20'}
            fontSize = 20

            fig = plt.figure(figsize=(12, 8))
            plt.rc('font', **axis_font)
            ax = fig.add_subplot(111)
            x_values = [el[0] for el in experiment_column_value[experiment_name]]
            #y_values = [all_values_encoding[el[1]] for el in experiment_column_value[experiment_name]]
            y_values = [el[1] for el in experiment_column_value[experiment_name]]

            #ax.set_title("experiment vs system implicaction")
            ax.tick_params(axis='both', which='major', labelsize=fontSize, rotation=60)
            ax.set_xlabel(x_column_name, fontsize=20)
            ax.set_ylabel(y_column_name, fontsize=20)
            ax.plot(x_values, y_values, label=y_column_name, linewidth=2)
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=fontSize)

            # dump in the top folder
            output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
            output_dir = os.path.join(output_base_dir, "single_workload/progression")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir,experiment_name+"_progression_"+'_'.join(y_column_name_list)+".png"))
            # plt.show()
            plt.close('all')

            fig = plt.figure(figsize=(12, 8))
            plt.rc('font', **axis_font)
            ax = fig.add_subplot(111)
            x_values = [el[0] for el in experiment_column_value[experiment_name]]
            # y_values = [all_values_encoding[el[1]] for el in experiment_column_value[experiment_name]]
            y_values = [el[1] for el in experiment_column_value[experiment_name]]

            # ax.set_title("experiment vs system implicaction")
            ax.tick_params(axis='both', which='major', labelsize=fontSize, rotation=60)
            ax.set_xlabel(x_column_name, fontsize=20)
            ax.set_ylabel(y_column_name, fontsize=20)
            ax.plot(x_values, y_values, label=y_column_name, linewidth=2)
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=fontSize)

            # dump in the top folder
            output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
            output_dir = os.path.join(output_base_dir, "single_workload/progression")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, experiment_name + "_progression_" + y_column_name + ".png"))
            # plt.show()
            plt.close('all')


def plot_3d(input_dir_names, res_column_name_number):
    # experiment_names
    experiment_names = []
    file_full_addr_list = []
    for dir_name in input_dir_names:
        file_full_addr = os.path.join(dir_name, "result_summary/FARSI_simple_run_0_1.csv")
        file_full_addr_list.append(file_full_addr)
        experiment_name = get_experiments_name(file_full_addr, res_column_name_number)
        experiment_names.append(experiment_name)

    axis_font = {'size': '10'}
    fontSize = 10
    column_value = {}
    # initialize the dictionary
    column_name_list = ["budget_scaling_power", "budget_scaling_area","budget_scaling_latency"]

    under_study_vars =["iteration cnt",
                       "local_bus_avg_theoretical_bandwidth", "local_bus_max_actual_bandwidth",
                       "local_bus_avg_actual_bandwidth",
                       "system_bus_avg_theoretical_bandwidth", "system_bus_max_actual_bandwidth",
                       "system_bus_avg_actual_bandwidth", "global_total_traffic", "local_total_traffic",
                       "global_memory_total_area", "local_memory_total_area", "ips_total_area",
                       "gpps_total_area","ip_cnt", "max_accel_parallelism", "avg_accel_parallelism",
                       "gpp_cnt", "max_gpp_parallelism", "avg_gpp_parallelism"]




    # get all the data
    for file_full_addr in file_full_addr_list:
        with open(file_full_addr, newline='') as csvfile:
            resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
            experiment_name = get_experiments_name( file_full_addr, res_column_name_number)

            for i, row in enumerate(resultReader):
                #if row[trueNum] != "True":
                #    continue
                if i >= 1:
                    for column_name in column_name_list + under_study_vars:
                        if column_name not in column_value.keys() :
                            column_value[column_name] = []
                        column_number = res_column_name_number[column_name]
                        col_value = row[column_number]
                        col_values = col_value.split(";")
                        if "=" in col_values[0]:
                            column_value[column_name].append(float((col_values[0]).split("=")[1]))
                        else:
                            column_value[column_name].append(float(col_values[0]))


    for idx,under_study_var in enumerate(under_study_vars):
        fig_budget_blkcnt = plt.figure(figsize=(12, 12))
        plt.rc('font', **axis_font)
        ax_blkcnt = fig_budget_blkcnt.add_subplot(projection='3d')
        img = ax_blkcnt.scatter3D(column_value["budget_scaling_power"], column_value["budget_scaling_area"], column_value["budget_scaling_latency"],
                                  c=column_value[under_study_var], cmap="bwr", s=80, label="System Block Count")
        for idx,_ in enumerate(column_value[under_study_var]):
            coordinate = column_value[under_study_var][idx]
            coord_in_scientific_notatio = "{:.2e}".format(coordinate)

            ax_blkcnt.text(column_value["budget_scaling_power"][idx], column_value["budget_scaling_area"][idx], column_value["budget_scaling_latency"][idx], '%s' % coord_in_scientific_notatio, size=fontSize)

        ax_blkcnt.set_xlabel("Power Budget", fontsize=fontSize)
        ax_blkcnt.set_ylabel("Area Budget", fontsize=fontSize)
        ax_blkcnt.set_zlabel("Latency Budget", fontsize=fontSize)
        ax_blkcnt.legend()
        cbar = fig_budget_blkcnt.colorbar(img, aspect=40)
        cbar.set_label("System Block Count", rotation=270)
        #plt.title("{Power Budget, Area Budget, Latency Budget} VS System Block Count: " + subDirName)
        plt.tight_layout()

        output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
        output_dir = os.path.join(output_base_dir, "3D/case_studies")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, under_study_var+ ".png"))
        # plt.show()
        plt.close('all')


def plot_convergence_per_workloads_for_paper(input_dir_names, res_column_name_number):
    budget_alpha = 1
    non_optimization_alpha = .1
    budget_marker = "_"
    regu_marker = "."
    budget_marker_size = 4
    regu_marker_size = 1


    #itrColNum = all_res_column_name_number["iteration cnt"]
    #distColNum = all_res_column_name_number["dist_to_goal_non_cost"]
    trueNum  =  all_res_column_name_number["move validity"]
    move_name_number =  all_res_column_name_number["move name"]


    # experiment_names
    experiment_names = []
    file_full_addr_list = []
    for dir_name in input_dir_names:
        file_full_addr = os.path.join(dir_name, "result_summary/FARSI_simple_run_0_1_all_reults.csv")
        file_full_addr_list.append(file_full_addr)
        experiment_name = get_experiments_name(file_full_addr, res_column_name_number)
        experiment_names.append(experiment_name)

    #color_values = ["r","b","y","black","brown","purple"]
    color_values = {}
    color_values["latency_edge_detection"] = color_values["best_des_so_far_latency_edge_detection"] =color_values["latency_budget_edge_detection"] = matplotlib.colors.to_rgba("red")
    color_values["latency_hpvm_cava"] =color_values["best_des_so_far_latency_hpvm_cava"] = color_values["latency_budget_hpvm_cava"] = matplotlib.colors.to_rgba("magenta")
    color_values["latency_audio_decoder"] = color_values["best_des_so_far_latency_audio_decoder"] =color_values["latency_budget_audio_decoder"] = matplotlib.colors.to_rgba("orange")
    color_values["area_non_dram"] =color_values["best_des_so_far_area_non_dram"] = color_values["area_budget"] =  matplotlib.colors.to_rgba("forestgreen")
    color_values["brown"] = (1,0,0,1)
    color_values["power"] = color_values["best_des_so_far_power"] =color_values["power_budget"] = matplotlib.colors.to_rgba("mediumblue")

    color_values["latency_budget_edge_detection"] = matplotlib.colors.to_rgba("white")
    color_values["latency_budget_hpvm_cava"] = matplotlib.colors.to_rgba("white")
    color_values["latency_budget_audio_decoder"] = matplotlib.colors.to_rgba("white")
    color_values["area_budget"] =  matplotlib.colors.to_rgba("white")
    color_values["power_budget"] = matplotlib.colors.to_rgba("white")



    column_name_color_val_dict = {"best_des_so_far_power":"purple", "power_budget":"purple","best_des_so_far_area_non_dram":"blue", "area_budget":"blue",
                                  "latency_budget_hpvm_cava":"orange", "latency_budget_audio_decoder":"yellow", "latency_budget_edge_detection":"red",
                                  "best_des_so_far_latency_hpvm_cava":"orange", "best_des_so_far_latency_audio_decoder": "yellow","best_des_so_far_latency_edge_detection": "red",
                                  "latency_budget":"white"
                                  }



    """
    column_name_color_val_dict = {"power":"purple", "power_budget":"purple","area_non_dram":"blue", "area_budget":"blue",
                                  "latency_budget_hpvm_cava":"orange", "latency_budget_audio_decoder":"yellow", "latency_budget_edge_detection":"red",
                                  "latency_hpvm_cava":"orange", "latency_audio_decoder": "yellow","latency_edge_detection": "red",
                                  "latency_budget":"white"
                                  }
    """

    axis_font = {'size': '20'}
    fontSize = 20
    x_column_name = "iteration cnt"
    #y_column_name_list = ["power", "area_non_dram", "latency", "latency_budget", "power_budget","area_budget"]
    y_column_name_list = ["power", "area_non_dram", "latency"]

    experiment_column_value = {}
    for file_full_addr in file_full_addr_list:
        experiment_name = get_experiments_name(file_full_addr, res_column_name_number)
        experiment_column_value[experiment_name] = {}
        for y_column_name in y_column_name_list:
            if "budget"  in y_column_name :
                prefix = ""
            else:
                prefix = "best_des_so_far_"
            y_column_name = prefix+y_column_name
            y_column_number = res_column_name_number[y_column_name]
            x_column_number = res_column_name_number[x_column_name]
            #dis_to_goal_column_number = res_column_name_number["dist_to_goal_non_cost"]
            #ref_des_dis_to_goal_column_number = res_column_name_number["ref_des_dist_to_goal_non_cost"]

            if not y_column_name == prefix+"latency":
                experiment_column_value[experiment_name][y_column_name] = []


            with open(file_full_addr, newline='') as csvfile:
                resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
                for i, row in enumerate(resultReader):
                    if i > 1:
                        if row[trueNum] == "FALSE" or row[move_name_number]=="identity":
                            continue
                        metric_chosen  = row[res_column_name_number["transformation_metric"]]
                        workload_chosen  = row[res_column_name_number["workload"]]
                        if "budget" in y_column_name:
                            alpha = budget_alpha
                        elif metric_chosen in y_column_name :
                            alpha = 1
                        else:
                            alpha = non_optimization_alpha

                        col_value = row[y_column_number]
                        if ";" in col_value:
                            col_value = col_value[:-1]
                        col_values = col_value.split(";")
                        for col_val in col_values:
                            if "=" in col_val:
                                if "budget" in y_column_name:
                                    alpha = budget_alpha
                                elif workload_chosen in col_val:
                                    alpha = 1
                                else:
                                    alpha = non_optimization_alpha
                                val_splitted = col_val.split("=")
                                value_to_add = (float(row[x_column_number]), (val_splitted[0], val_splitted[1]), alpha)
                            else:
                                value_to_add = (float(row[x_column_number]), col_val, alpha)

                            if y_column_name in [prefix+"latency", prefix+"latency_budget"] :
                                new_tuple = (value_to_add[0], 1000*float(value_to_add[1][1]), value_to_add[2])
                                if y_column_name+"_"+value_to_add[1][0] not in experiment_column_value[experiment_name].keys():
                                    experiment_column_value[experiment_name][y_column_name + "_" + value_to_add[1][0]] = []
                                experiment_column_value[experiment_name][y_column_name+"_"+value_to_add[1][0]].append(new_tuple)
                            if y_column_name in [prefix+"power", prefix+"power_budget"]:
                               new_tuple = (value_to_add[0], float(value_to_add[1])*1000, value_to_add[2])
                               experiment_column_value[experiment_name][y_column_name].append(new_tuple)
                            elif y_column_name in [prefix+"area_non_dram", prefix+"area_budget"]:
                                new_tuple = (value_to_add[0], float(value_to_add[1]) * 1000000, value_to_add[2])
                                experiment_column_value[experiment_name][y_column_name].append(new_tuple)

            # prepare for plotting and plot
            fig = plt.figure(figsize=(15, 8))
            ax = fig.add_subplot(111)
            for column, values in experiment_column_value[experiment_name].items():
                print(column)
                if "budget" in column:
                    marker = budget_marker
                    marker_size = budget_marker_size
                else:
                    marker = regu_marker
                    marker_size =regu_marker_size

                x_values = [el[0] for el in values]
                y_values = [el[1] for el in values]
                colors = []
                for el in values:
                    color_ = list(color_values[column])
                    color_[-1] = el[2]
                    colors.append(tuple(color_))
                #alphas = [el[2] for el in values]
                ax.set_yscale('log')

                if "budget" in column:
                    marker = '_'
                    #alpha_ = .3
                else:
                    marker = "x"
                    #alpha_ = 1
                for i,x in enumerate(x_values):

                    ax.plot(x_values[i], y_values[i], label=column, color=colors[i], marker=marker, markersize=marker_size)


            #ax.set_title("experiment vs system implicaction")
            ax.set_xlabel(x_column_name, fontsize=fontSize)
            y_axis_name = "_".join(list(experiment_column_value[experiment_name].keys()))
            ax.set_ylabel(y_axis_name, fontsize=fontSize)
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=fontSize)
            plt.tight_layout()

            # dump in the top folder
            output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
            output_dir = os.path.join(output_base_dir, "single_workload/convergence")
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            fig.savefig(os.path.join(output_dir,experiment_name+"_convergence.png"))
            # plt.show()
            plt.close('all')


def get_budget_values(input_dir_names, res_column_name_number):
    experiment_names = []
    file_full_addr_list = []
    for dir_name in input_dir_names:
        file_full_addr = os.path.join(dir_name, "result_summary/FARSI_simple_run_0_1_all_reults.csv")
        file_full_addr_list.append(file_full_addr)
        experiment_name = get_experiments_name(file_full_addr, res_column_name_number)
        experiment_names.append(experiment_name)

    x_column_name = "iteration cnt"
    trueNum = all_res_column_name_number["move validity"]
    move_name_number = all_res_column_name_number["move name"]

    y_column_name_list = ["latency_budget", "power_budget","area_budget"]
    experiment_column_value = {}
    budgets = {}
    for file_full_addr in file_full_addr_list:
        experiment_name = get_experiments_name(file_full_addr, res_column_name_number)
        experiment_column_value[experiment_name] = {}
        for y_column_name in y_column_name_list:
            if "budget"  in y_column_name :
                prefix = ""
            else:
                prefix = "best_des_so_far_"
            y_column_name = prefix+y_column_name
            y_column_number = res_column_name_number[y_column_name]
            x_column_number = res_column_name_number[x_column_name]
            #dis_to_goal_column_number = res_column_name_number["dist_to_goal_non_cost"]
            #ref_des_dis_to_goal_column_number = res_column_name_number["ref_des_dist_to_goal_non_cost"]

            if not y_column_name == prefix+"latency":
                experiment_column_value[experiment_name][y_column_name] = []


            with open(file_full_addr, newline='') as csvfile:
                resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
                for i, row in enumerate(resultReader):
                    if i > 1:
                        if row[trueNum] == "FALSE" or row[move_name_number]=="identity":
                            continue
                        metric_chosen  = row[res_column_name_number["transformation_metric"]]
                        workload_chosen  = row[res_column_name_number["workload"]]

                        col_value = row[y_column_number]
                        if ";" in col_value:
                            col_value = col_value[:-1]
                        col_values = col_value.split(";")
                        for col_val in col_values:
                            alpha = 1
                            if "=" in col_val:
                                val_splitted = col_val.split("=")
                                value_to_add = (float(row[x_column_number]), (val_splitted[0], val_splitted[1]), alpha)
                            else:
                                value_to_add = (float(row[x_column_number]), col_val, alpha)

                            if y_column_name in [prefix+"latency", prefix+"latency_budget"] :
                                new_tuple = (value_to_add[0], 1000*float(value_to_add[1][1]), value_to_add[2])
                                if y_column_name+"_"+value_to_add[1][0] not in experiment_column_value[experiment_name].keys():
                                    experiment_column_value[experiment_name][y_column_name + "_" + value_to_add[1][0]] = []
                                experiment_column_value[experiment_name][y_column_name+"_"+value_to_add[1][0]].append(new_tuple)
                            if y_column_name in [prefix+"power", prefix+"power_budget"]:
                               new_tuple = (value_to_add[0], float(value_to_add[1])*1000, value_to_add[2])
                               experiment_column_value[experiment_name][y_column_name].append(new_tuple)
                            elif y_column_name in [prefix+"area_non_dram", prefix+"area_budget"]:
                                new_tuple = (value_to_add[0], float(value_to_add[1]) * 1000000, value_to_add[2])
                                experiment_column_value[experiment_name][y_column_name].append(new_tuple)

            for column, values in experiment_column_value[experiment_name].items():
                if len(values) == 0:
                    continue
                x_values = [el[0] for el in values]
                budgets[column] = values[1][1]

        return budgets


def plot_convergence_per_workloads(input_dir_names, res_column_name_number):

    budgets = get_budget_values(input_dir_names, res_column_name_number)

    budget_alpha = 1
    non_optimization_alpha = .1
    budget_marker = "_"
    regu_marker = "."
    budget_marker_size = 4
    regu_marker_size = 1


    #itrColNum = all_res_column_name_number["iteration cnt"]
    #distColNum = all_res_column_name_number["dist_to_goal_non_cost"]
    trueNum  =  all_res_column_name_number["move validity"]
    move_name_number =  all_res_column_name_number["move name"]


    # experiment_names
    experiment_names = []
    file_full_addr_list = []
    for dir_name in input_dir_names:
        file_full_addr = os.path.join(dir_name, "result_summary/FARSI_simple_run_0_1_all_reults.csv")
        file_full_addr_list.append(file_full_addr)
        experiment_name = get_experiments_name(file_full_addr, res_column_name_number)
        experiment_names.append(experiment_name)

    #color_values = ["r","b","y","black","brown","purple"]
    color_values = {}
    color_values["latency_edge_detection"] = color_values["best_des_so_far_latency_edge_detection"] =color_values["latency_budget_edge_detection"] = matplotlib.colors.to_rgba("red")
    color_values["latency_hpvm_cava"] =color_values["best_des_so_far_latency_hpvm_cava"] = color_values["latency_budget_hpvm_cava"] = matplotlib.colors.to_rgba("magenta")
    color_values["latency_audio_decoder"] = color_values["best_des_so_far_latency_audio_decoder"] =color_values["latency_budget_audio_decoder"] = matplotlib.colors.to_rgba("orange")
    color_values["area_non_dram"] =color_values["best_des_so_far_area_non_dram"] = color_values["area_budget"] =  matplotlib.colors.to_rgba("forestgreen")
    color_values["brown"] = (1,0,0,1)
    color_values["power"] = color_values["best_des_so_far_power"] =color_values["power_budget"] = matplotlib.colors.to_rgba("mediumblue")

    color_values["latency_budget_edge_detection"] = matplotlib.colors.to_rgba("white")
    color_values["latency_budget_hpvm_cava"] = matplotlib.colors.to_rgba("white")
    color_values["latency_budget_audio_decoder"] = matplotlib.colors.to_rgba("white")
    color_values["area_budget"] =  matplotlib.colors.to_rgba("white")
    color_values["power_budget"] = matplotlib.colors.to_rgba("white")



    column_name_color_val_dict = {"best_des_so_far_power":"purple", "power_budget":"purple","best_des_so_far_area_non_dram":"blue", "area_budget":"blue",
                                  "latency_budget_hpvm_cava":"orange", "latency_budget_audio_decoder":"yellow", "latency_budget_edge_detection":"red",
                                  "best_des_so_far_latency_hpvm_cava":"orange", "best_des_so_far_latency_audio_decoder": "yellow","best_des_so_far_latency_edge_detection": "red",
                                  "latency_budget":"white"
                                  }



    """
    column_name_color_val_dict = {"power":"purple", "power_budget":"purple","area_non_dram":"blue", "area_budget":"blue",
                                  "latency_budget_hpvm_cava":"orange", "latency_budget_audio_decoder":"yellow", "latency_budget_edge_detection":"red",
                                  "latency_hpvm_cava":"orange", "latency_audio_decoder": "yellow","latency_edge_detection": "red",
                                  "latency_budget":"white"
                                  }
    """

    axis_font = {'size': '20'}
    fontSize = 20
    x_column_name = "iteration cnt"
    #y_column_name_list = ["power", "area_non_dram", "latency", "latency_budget", "power_budget","area_budget"]
    y_column_name_list = ["power", "area_non_dram", "latency"]

    experiment_column_value = {}
    for file_full_addr in file_full_addr_list:
        experiment_name = get_experiments_name(file_full_addr, res_column_name_number)
        experiment_column_value[experiment_name] = {}
        for y_column_name in y_column_name_list:
            if "budget"  in y_column_name :
                prefix = ""
            else:
                prefix = "best_des_so_far_"
            y_column_name = prefix+y_column_name
            y_column_number = res_column_name_number[y_column_name]
            x_column_number = res_column_name_number[x_column_name]
            #dis_to_goal_column_number = res_column_name_number["dist_to_goal_non_cost"]
            #ref_des_dis_to_goal_column_number = res_column_name_number["ref_des_dist_to_goal_non_cost"]

            if not y_column_name == prefix+"latency":
                experiment_column_value[experiment_name][y_column_name] = []


            with open(file_full_addr, newline='') as csvfile:
                resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
                for i, row in enumerate(resultReader):
                    if i > 1:
                        if row[trueNum] == "FALSE" or row[move_name_number]=="identity":
                            continue
                        metric_chosen  = row[res_column_name_number["transformation_metric"]]
                        workload_chosen  = row[res_column_name_number["workload"]]
                        if "budget" in y_column_name:
                            alpha = budget_alpha
                        elif metric_chosen in y_column_name :
                            alpha = 1
                        else:
                            alpha = non_optimization_alpha

                        col_value = row[y_column_number]
                        if ";" in col_value:
                            col_value = col_value[:-1]
                        col_values = col_value.split(";")
                        for col_val in col_values:
                            if "=" in col_val:
                                if "budget" in y_column_name:
                                    alpha = budget_alpha
                                elif workload_chosen in col_val:
                                    alpha = 1
                                else:
                                    alpha = non_optimization_alpha
                                val_splitted = col_val.split("=")
                                value_to_add = (float(row[x_column_number]), (val_splitted[0], val_splitted[1]), alpha)
                            else:
                                value_to_add = (float(row[x_column_number]), col_val, alpha)

                            if y_column_name in [prefix+"latency", prefix+"latency_budget"] :
                                budget = budgets["latency_budget"+"_"+value_to_add[1][0]]
                                new_tuple = (value_to_add[0], (-budget + 1000*float(value_to_add[1][1]))/budget, value_to_add[2])
                                if y_column_name+"_"+value_to_add[1][0] not in experiment_column_value[experiment_name].keys():
                                    experiment_column_value[experiment_name][y_column_name + "_" + value_to_add[1][0]] = []
                                experiment_column_value[experiment_name][y_column_name+"_"+value_to_add[1][0]].append(new_tuple)
                            if y_column_name in [prefix+"power", prefix+"power_budget"]:
                                budget = budgets["power_budget"]
                                new_tuple = (value_to_add[0], (-budget + float(value_to_add[1])*1000)/budget, value_to_add[2])
                                experiment_column_value[experiment_name][y_column_name].append(new_tuple)
                            elif y_column_name in [prefix+"area_non_dram", prefix+"area_budget"]:
                                budget = budgets["area_budget"]
                                new_tuple = (value_to_add[0], (-budget + float(value_to_add[1]) * 1000000)/budget, value_to_add[2])
                                experiment_column_value[experiment_name][y_column_name].append(new_tuple)

            # prepare for plotting and plot
            fig = plt.figure(figsize=(15, 8))
            ax = fig.add_subplot(111)
            for column, values in experiment_column_value[experiment_name].items():
                if "budget" in column:
                    marker = budget_marker
                    marker_size = budget_marker_size
                else:
                    marker = regu_marker
                    marker_size =regu_marker_size

                x_values = [el[0] for el in values]
                y_values = [el[1] for el in values]
                colors = []
                for el in values:
                    color_ = list(color_values[column])
                    color_[-1] = el[2]
                    colors.append(tuple(color_))
                #alphas = [el[2] for el in values]
                ax.set_yscale('log')

                if "budget" in column:
                    marker = '_'
                    #alpha_ = .3
                else:
                    marker = "x"
                    #alpha_ = 1
                for i,x in enumerate(x_values):
                    ax.plot(x_values[i], y_values[i]+10, label=column, color=colors[i], marker=marker, markersize=marker_size)


            #ax.set_title("experiment vs system implicaction")
            ax.set_xlabel(x_column_name, fontsize=fontSize)
            y_axis_name = "_".join(list(experiment_column_value[experiment_name].keys()))
            ax.set_ylabel(y_axis_name, fontsize=fontSize)
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=fontSize)
            plt.tight_layout()

            # dump in the top folder
            output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
            output_dir = os.path.join(output_base_dir, "single_workload/convergence")
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            fig.savefig(os.path.join(output_dir,experiment_name+"_convergence.png"))
            # plt.show()
            plt.close('all')

def plot_convergence_vs_time(input_dir_names, res_column_name_number):
    PA_time_scaling_factor = 8500
    #itrColNum = all_res_column_name_number["iteration cnt"]
    #distColNum = all_res_column_name_number["dist_to_goal_non_cost"]
    trueNum  =  all_res_column_name_number["move validity"]

    # experiment_names
    experiment_names = []
    file_full_addr_list = []
    for dir_name in input_dir_names:
        file_full_addr = os.path.join(dir_name, "result_summary/FARSI_simple_run_0_1_all_reults.csv")
        file_full_addr_list.append(file_full_addr)
        experiment_name = get_experiments_name(file_full_addr, res_column_name_number)
        experiment_names.append(experiment_name)

    axis_font = {'size': '15'}
    fontSize = 20
    x_column_name = "exploration_plus_simulation_time"
    y_column_name_list = ["best_des_so_far_dist_to_goal_non_cost"]
    y_column_name_list = ["dist_to_goal_non_cost"]


    PA_column_experiment_value = {}
    FARSI_column_experiment_value = {}

    #column_name = "move name"
    for k, file_full_addr in enumerate(file_full_addr_list):
        for y_column_name in y_column_name_list:
            # get all possible the values of inteFARSI_results/blind_study_all_dumb_versionrest
            y_column_number = res_column_name_number[y_column_name]
            x_column_number = res_column_name_number[x_column_name]
            PA_column_experiment_value[y_column_name] = []
            FARSI_column_experiment_value[y_column_name] = []
            PA_last_time = 0
            FARSI_last_time = 0
            with open(file_full_addr, newline='') as csvfile:
                resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
                experiment_name = get_experiments_name( file_full_addr, res_column_name_number)
                for i, row in enumerate(resultReader):
                    #if row[trueNum] != "True":
                    #    continue
                    if i >= 1:
                        FARSI_last_time += float(row[x_column_number])
                        FARSI_value_to_add = (float(FARSI_last_time), row[y_column_number])
                        FARSI_column_experiment_value[y_column_name].append(FARSI_value_to_add)

                        PA_last_time = FARSI_last_time*PA_time_scaling_factor
                        PA_value_to_add = (float(PA_last_time), row[y_column_number])
                        PA_column_experiment_value[y_column_name].append(PA_value_to_add)

                # prepare for plotting and plot
                fig = plt.figure(figsize=(12, 12))
                plt.rc('font', **axis_font)
                ax = fig.add_subplot(111)
                fontSize = 20
                #plt.tight_layout()
                x_values = [el[0] for el in FARSI_column_experiment_value[y_column_name]]
                y_values = [(float(el[1]) * 100 // 1 / 100.0) for el in FARSI_column_experiment_value[y_column_name]]
                x_values.reverse()
                y_values.reverse()
                ax.scatter(x_values, y_values, label="FARSI time to completion", marker="*")
                # ax.set_yscale('log')

                x_values = [el[0] for el in PA_column_experiment_value[y_column_name]]
                y_values = [(float(el[1]) * 100 // 1 / 100.0) for el in PA_column_experiment_value[y_column_name]]
                x_values.reverse()
                y_values.reverse()
                ax.scatter(x_values, y_values, label="PA time to completion", marker="*")
                ax.set_xscale('log')

                #ax.set_title("experiment vs system implicaction")
                ax.legend(loc="upper right")#bbox_to_anchor=(1, 1), loc="upper left")
                ax.set_xlabel(x_column_name, fontsize=fontSize)
                ax.set_ylabel(y_column_name, fontsize=fontSize)
                plt.tight_layout()

                # dump in the top folder
                output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
                output_dir = os.path.join(output_base_dir, "single_workload/progression")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                fig.savefig(os.path.join(output_dir,str(k)+"_" + y_column_name+"_vs_"+x_column_name+"_FARSI_vs_PA.png"))
                #plt.show()
                plt.close('all')

def plot_convergence_vs_time_for_paper(input_dir_names, res_column_name_number):
    PA_time_scaling_factor = 8500
    #itrColNum = all_res_column_name_number["iteration cnt"]
    #distColNum = all_res_column_name_number["dist_to_goal_non_cost"]
    trueNum  =  all_res_column_name_number["move validity"]

    # experiment_names
    experiment_names = []
    file_full_addr_list = []
    for dir_name in input_dir_names:
        file_full_addr = os.path.join(dir_name, "result_summary/FARSI_simple_run_0_1_all_reults.csv")
        file_full_addr_list.append(file_full_addr)
        experiment_name = get_experiments_name(file_full_addr, res_column_name_number)
        experiment_names.append(experiment_name)

    axis_font = {'size': '25'}
    fontSize = 25
    x_column_name = "exploration_plus_simulation_time"
    y_column_name_list = ["best_des_so_far_dist_to_goal_non_cost"]
    y_column_name_list = ["dist_to_goal_non_cost"]


    PA_column_experiment_value = {}
    FARSI_column_experiment_value = {}

    #column_name = "move name"
    for k, file_full_addr in enumerate(file_full_addr_list):
        for y_column_name in y_column_name_list:
            # get all possible the values of interest
            y_column_number = res_column_name_number[y_column_name]
            x_column_number = res_column_name_number[x_column_name]
            PA_column_experiment_value[y_column_name] = []
            FARSI_column_experiment_value[y_column_name] = []
            PA_last_time = 0
            FARSI_last_time = 0
            with open(file_full_addr, newline='') as csvfile:
                resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
                experiment_name = get_experiments_name( file_full_addr, res_column_name_number)
                for i, row in enumerate(resultReader):
                    #if row[trueNum] != "True":
                    #    continue
                    if i >= 1:
                        FARSI_last_time += float(row[x_column_number])
                        FARSI_value_to_add = (float(FARSI_last_time), row[y_column_number])
                        FARSI_column_experiment_value[y_column_name].append(FARSI_value_to_add)

                        PA_last_time = FARSI_last_time*PA_time_scaling_factor
                        PA_value_to_add = (float(PA_last_time), row[y_column_number])
                        PA_column_experiment_value[y_column_name].append(PA_value_to_add)

                # prepare for plotting and plot
                print(k)
                fig = plt.figure(figsize=(6, 6))
                # axes = plt.gca()
                ymax = 2800.0
                plt.rc('font', **axis_font)
                ax = fig.add_subplot(111)
                #plt.tight_layout()
                x_values = [el[0] for el in FARSI_column_experiment_value[y_column_name]]
                y_values = [(float(el[1]) * 100 // 1 / 100.0) for el in FARSI_column_experiment_value[y_column_name]]
                x_values.reverse()
                y_values.reverse()
                ax.scatter(x_values, y_values, label="FARSI", marker="*", color="green", s=10)
                # ax.set_yscale('log')

                x_values = [el[0] for el in PA_column_experiment_value[y_column_name]]
                y_values = [(float(el[1]) * 100 // 1 / 100.0) for el in PA_column_experiment_value[y_column_name]]
                x_values.reverse()
                y_values.reverse()
                ax.scatter(x_values, y_values, label="PA", marker="*", color="hotpink", s=10)
                ax.set_xscale('log')
                ax.set_yscale('log')
                # ax.set_yscale('linear') # Ying: by default, the y ticks are strange
                ax.set_ylim([np.power(10.0, -2), np.power(10.0, 3.5)])

                #ax.set_title("experiment vs system implicaction")
                ax.legend(bbox_to_anchor=(0.5, 1.15), loc="upper center", fontsize=fontSize, ncol=2, borderpad=0, markerscale=6)#bbox_to_anchor=(1, 1), loc="upper left")
                ax.set_xlabel("Time to Completion (s)", fontsize=fontSize)
                ax.set_ylabel("Norm Distance to Goal", fontsize=fontSize)
                # floatY = [float(i) for i in y_values]
                # maxY = max(floatY)
                # ax.set_yticks(np.arange(0, ymax + 1, ymax / 5))
                ax.set_xticks(np.power(10.0, [0, 4, 8]))
                ax.set_yticks(np.power(10.0, [-2, -1, 0, 1, 2, 3]))
                plt.tight_layout()

                # dump in the top folder
                output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
                output_dir = os.path.join(output_base_dir, "single_workload/progression")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                fig.savefig(os.path.join(output_dir,str(k)+"_" + y_column_name+"_vs_"+x_column_name+"_FARSI_vs_PA.png"), bbox_inches='tight')
                #plt.show()
                plt.close('all')

def plot_convergence_cross_workloads(input_dir_names, res_column_name_number):
    #itrColNum = all_res_column_name_number["iteration cnt"]
    #distColNum = all_res_column_name_number["dist_to_goal_non_cost"]
    trueNum  =  all_res_column_name_number["move validity"]

    # experiment_names
    experiment_names = []
    file_full_addr_list = []
    for dir_name in input_dir_names:
        file_full_addr = os.path.join(dir_name, "result_summary/FARSI_simple_run_0_1_all_reults.csv")
        file_full_addr_list.append(file_full_addr)
        experiment_name = get_experiments_name(file_full_addr, res_column_name_number)
        experiment_names.append(experiment_name)

    axis_font = {'size': '20'}
    x_column_name = "iteration cnt"
    y_column_name_list = ["best_des_so_far_dist_to_goal_non_cost", "dist_to_goal_non_cost"]

    column_experiment_value = {}
    #column_name = "move name"
    for y_column_name in y_column_name_list:
        # get all possible the values of interest
        y_column_number = res_column_name_number[y_column_name]
        #x_column_number = res_column_name_number[x_column_name]
        ctr = 0

        column_experiment_value[y_column_name] = {}
        # initialize the dictionary
        # get all the data
        for file_full_addr in file_full_addr_list:
            with open(file_full_addr, newline='') as csvfile:
                resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
                experiment_name = get_experiments_name( file_full_addr, res_column_name_number) +str(ctr)
                column_experiment_value[y_column_name][experiment_name] = []

                for i, row in enumerate(resultReader):
                    #if row[trueNum] != "True":
                    #    continue
                    if i >= 1:
                        value_to_add = (i, max(float(row[y_column_number]),.01))
                        column_experiment_value[y_column_name][experiment_name].append(value_to_add)
            ctr +=1

        # prepare for plotting and plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #plt.tight_layout()
        for experiment_name, values in column_experiment_value[y_column_name].items():
            x_values = [el[0] for el in values[:-10]]
            y_values = [el[1] for el in values[:-10]]
            ax.scatter(x_values, y_values, label=experiment_name[1:])

        #ax.set_title("experiment vs system implicaction")
        ax.set_yscale('log')
        ax.legend(bbox_to_anchor=(1, 1), loc="best")
        ax.set_xlabel(x_column_name)
        ax.set_ylabel(y_column_name)
        plt.tight_layout()

        # dump in the top folder
        output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
        output_dir = os.path.join(output_base_dir, "cross_workloads/convergence")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(os.path.join(output_dir,x_column_name+"_"+y_column_name+".png"))
        # plt.show()
        plt.close('all')

def plot_convergence_cross_workloads_for_paper(input_dir_names, res_column_name_number):
    #itrColNum = all_res_column_name_number["iteration cnt"]
    #distColNum = all_res_column_name_number["dist_to_goal_non_cost"]
    trueNum  =  all_res_column_name_number["move validity"]

    # experiment_names
    experiment_names = []
    file_full_addr_list = []
    for dir_name in input_dir_names:
        file_full_addr = os.path.join(dir_name, "result_summary/FARSI_simple_run_0_1_all_reults.csv")
        file_full_addr_list.append(file_full_addr)
        experiment_name = get_experiments_name(file_full_addr, res_column_name_number)
        experiment_names.append(experiment_name)

    axis_font = {'size': '25'}
    fontSize = 25
    x_column_name = "Iteration"
    y_column_name_list = ["best_des_so_far_dist_to_goal_non_cost", "dist_to_goal_non_cost"]

    column_experiment_value = {}
    #column_name = "move name"
    for y_column_name in y_column_name_list:
        y_column_name_rep = y_column_name
        # get all possible the values of interest
        y_column_number = res_column_name_number[y_column_name]
        #x_column_number = res_column_name_number[x_column_name]

        column_experiment_value[y_column_name] = {}
        # initialize the dictionary
        # get all the data
        ctr = 0
        for file_full_addr in file_full_addr_list:
            with open(file_full_addr, newline='') as csvfile:
                resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
                experiment_name = get_experiments_name( file_full_addr, res_column_name_number) +str(ctr)

                column_experiment_value[y_column_name][experiment_name] = []

                for i, row in enumerate(resultReader):
                    #if row[trueNum] != "True":
                    #    continue
                    if i >= 1:
                        value_to_add = (i, max(float(row[y_column_number]),.01))
                        column_experiment_value[y_column_name][experiment_name].append(value_to_add)

            ctr +=1
        # prepare for plotting and plot
        fig = plt.figure(figsize=(8, 6.5))
        plt.rc('font', **axis_font)
        ax = fig.add_subplot(111)
        #plt.tight_layout()
        labelName = ""
        color = (1.0, 0.0, 0.0, 0.5) # ""
        for experiment_name, values in column_experiment_value[y_column_name].items():
            x_values = [el[0] for el in values[:-10]]
            y_values = [el[1] for el in values[:-10]]
            # Ying: hardcode here
            if experiment_name[-1] == "3":
                labelName = "SA"
                color = (1, 0, 0, 1) # "Red"
            if experiment_name[-1] == "2":
                labelName = "FARSI"
                color = (0, 0.6, 0, 1) # "ForestGreen"
            if experiment_name[-1] == "1":
                labelName = "Task-aware"
                color = (1, 0.5, 0.0, 1) # "DarkOrange"
            if experiment_name[-1] == "0":
                labelName = "Task&Block-aware"
                color = (0.5, 0.5, 0, 1) # "Olive"
            # Ying: hardcode finished
            ax.scatter(x_values, y_values, label=labelName, color=color, marker='*', s=5)

        #ax.set_title("experiment vs system implicaction")
        ax.set_yscale('log')
        ax.legend(bbox_to_anchor=(0.5, 1.25), loc="upper center", fontsize=fontSize-2, ncol=2, borderpad=0, markerscale=6)
        ax.set_xlabel(x_column_name, fontsize=fontSize)
        # Ying: hardcode here
        if y_column_name == "dist_to_goal_non_cost":
            y_column_name_rep = "Norm Distance to Goal"
        # Ying: hardcode finished
        ax.set_ylabel(y_column_name_rep, fontsize=fontSize)
        plt.xticks(np.arange(0, 7000, 2000.0))
        plt.yticks(np.power(10.0, [-2, -1, 0, 1, 2, 3]))
        plt.tight_layout()

        # dump in the top folder
        output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
        output_dir = os.path.join(output_base_dir, "cross_workloads/convergence")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(os.path.join(output_dir,x_column_name+"_"+y_column_name+".png"), bbox_inches='tight')
        # plt.show()
        plt.close('all')

def plot_system_implication_analysis(input_dir_names, res_column_name_number, case_study):
    # experiment_names
    experiment_names = []
    file_full_addr_list = []
    for dir_name in input_dir_names:
        file_full_addr = os.path.join(dir_name, "result_summary/FARSI_simple_run_0_1.csv")
        file_full_addr_list.append(file_full_addr)
        experiment_name = get_experiments_name(file_full_addr, res_column_name_number)
        experiment_names.append(experiment_name)

    axis_font = {'size': '10'}

    column_name_list = list(case_study.values())[0]

    column_experiment_value = {}
    #column_name = "move name"
    for column_name in column_name_list:
        # get all possible the values of interest
        column_number = res_column_name_number[column_name]

        column_experiment_value[column_name] = {}
        # initialize the dictionary
        column_experiment_number_dict = {}
        experiment_number_dict = {}

        # get all the data
        for file_full_addr in file_full_addr_list:
            with open(file_full_addr, newline='') as csvfile:
                resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
                experiment_name = get_experiments_name( file_full_addr, res_column_name_number)

                for i, row in enumerate(resultReader):
                    #if row[trueNum] != "True":
                    #    continue
                    if i >= 1:
                        col_value = row[column_number]
                        col_values = col_value.split(";")
                        for col_val in col_values:
                            column_experiment_value[column_name][experiment_name] = float(col_val)

    # prepare for plotting and plot
    # plt.figure()
    index = experiment_names
    plotdata = pd.DataFrame(column_experiment_value, index=index)
    if list(case_study.keys())[0] in ["bandwidth_analysis","traffic_analysis"]:
        plotdata.plot(kind='bar', fontsize=9, rot=5, log=True)
    else:
        plotdata.plot(kind='bar', fontsize=9, rot=5)

    plt.legend(loc="best", fontsize="9")
    plt.xlabel("experiments", fontsize="10")
    plt.ylabel("system implication")
    #plt.title("experiment vs system implicaction")
    # dump in the top folder
    output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
    output_dir = os.path.join(output_base_dir, "cross_workloads/system_implications")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #plt.tight_layout()list(case_study.keys())[0]
    if "re_use" in list(case_study.keys())[0] or "speedup" in list(case_study.keys())[0]:
        plt.yscale('log')
    plt.savefig(os.path.join(output_dir,list(case_study.keys())[0]+".png"))
    plt.close('all')



def plot_co_design_nav_breakdown_post_processing(input_dir_names, column_column_value_experiment_frequency_dict):
    column_name_list = [("exact optimization name", "neighbouring design space size", "div")]
    #column_name = "move name"
    for n, column_name_tuple in enumerate(column_name_list):
        first_column =  column_name_tuple[0]
        second_column =  column_name_tuple[1]
        operation =   column_name_tuple[2]
        new_column_name = first_column+"_"+operation+"_"+second_column

        first_column_value_experiment_frequency_dict = column_column_value_experiment_frequency_dict[first_column]
        second_column_value_experiment_frequency_dict = column_column_value_experiment_frequency_dict[second_column]
        modified_column_value_experiment_frequency_dict = {}

        experiment_names = []
        for column_val, experiment_freq  in first_column_value_experiment_frequency_dict.items():
            if column_val == "unknown":
                continue
            modified_column_value_experiment_frequency_dict[column_val] = {}
            for experiment, freq in  experiment_freq.items():
                if(second_column_value_experiment_frequency_dict[column_val][experiment]) < .000001:
                    modified_column_value_experiment_frequency_dict[column_val][experiment] = 0
                else:
                    modified_column_value_experiment_frequency_dict[column_val][experiment] = first_column_value_experiment_frequency_dict[column_val][experiment]/max(second_column_value_experiment_frequency_dict[column_val][experiment],.0000000000001)
                experiment_names.append(experiment)

        axis_font = {'size': '22'}
        fontSize = 22
        experiment_names =  list(set(experiment_names))
        # prepare for plotting and plot
        # plt.figure(n)
        plt.rc('font', **axis_font)
        index = experiment_names
        plotdata = pd.DataFrame(modified_column_value_experiment_frequency_dict, index=index)
        plotdata.plot(kind='bar', stacked=True, figsize=(13, 8))
        plt.xlabel("experiments", **axis_font)
        plt.ylabel(new_column_name, **axis_font)
        plt.xticks(fontsize=fontSize, rotation=45)
        plt.yticks(fontsize=fontSize)
        plt.title("experiment vs " + new_column_name, **axis_font)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=fontSize)
        # dump in the top folder
        output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
        output_dir = os.path.join(output_base_dir, "cross_workloads/nav_breakdown")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # plt.tight_layout()
        plt.savefig(os.path.join(output_dir,'_'.join(new_column_name.split(" "))+".png"), bbox_inches='tight')
        plt.tight_layout()
        # plt.show()
        plt.close('all')



# navigation breakdown
def plot_codesign_nav_breakdown_per_workload(input_dir_names, input_all_res_column_name_number):
    trueNum = input_all_res_column_name_number["move validity"]

    # experiment_names
    experiment_names = []
    file_full_addr_list = []
    for dir_name in input_dir_names:
        file_full_addr = os.path.join(dir_name, "result_summary/FARSI_simple_run_0_1_all_reults.csv")
        file_full_addr_list.append(file_full_addr)
        experiment_name = get_experiments_name(file_full_addr, input_all_res_column_name_number)
        experiment_names.append(experiment_name)


    axis_font = {'size': '20'}
    fontSize = 20
    column_name_list = ["transformation_metric", "comm_comp", "workload"]#, "architectural principle", "high level optimization name", "exact optimization name"]
    #column_name_list = ["architectural principle", "exact optimization name"]

    #column_name = "move name"
    # initialize the dictionary
    column_column_value_experiment_frequency_dict = {}
    for file_full_addr in file_full_addr_list:
        column_column_value_frequency_dict = {}
        for column_name in column_name_list:
            with open(file_full_addr, newline='') as csvfile:
                resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
                experiment_name = get_experiments_name(file_full_addr, input_all_res_column_name_number)
                #column_column_value_frequency_dict[column_name] = {}
                # get all possible the values of interest
                all_values = get_all_col_values_of_a_folders(input_dir_names, input_all_res_column_name_number, column_name)
                columne_number = all_res_column_name_number[column_name]
                for column in all_values:
                    column_column_value_frequency_dict[column] = {}
                    column_column_value_frequency_dict[column][column_name] = 0
                for i, row in enumerate(resultReader):
                    if row[trueNum] != "True":
                        continue
                    if i > 1:
                        col_value = row[columne_number]
                        col_values = col_value.split(";")
                        for col_val in col_values:
                            if "=" in col_val:
                                val_splitted = col_val.split("=")
                                column_column_value_frequency_dict[val_splitted[0]][column_name] += float(val_splitted[1])
                            else:
                                column_column_value_frequency_dict[col_val][column_name] += 1

        index = column_name_list
        total_cnt = 0
        for val in column_column_value_frequency_dict[column].values():
            total_cnt += val

        for col_val, column_name_val in column_column_value_frequency_dict.items():
            for column_name, val in column_name_val.items():
                column_column_value_frequency_dict[col_val][column_name] /= max(total_cnt,1)

        plotdata = pd.DataFrame(column_column_value_frequency_dict, index=index)
        plotdata.plot(kind='bar', stacked=True, figsize=(10, 10))
        plt.rc('font', ** axis_font)
        plt.xlabel("experiments", **axis_font)
        plt.ylabel(column_name, **axis_font)
        plt.xticks(fontsize=fontSize, rotation=45)
        plt.yticks(fontsize=fontSize)
        plt.title("experiment vs " + column_name, **axis_font)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=fontSize)
        # dump in the top folder
        output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
        output_dir = os.path.join(output_base_dir, "single_workload/nav_breakdown")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,"__".join(column_name_list)+".png"), bbox_inches='tight')
        # plt.show()
        plt.close('all')
        #column_column_value_experiment_frequency_dict[column_name] = copy.deepcopy(column_column_value_frequency_dict)

    return column_column_value_experiment_frequency_dict




def plot_codesign_nav_breakdown_cross_workload(input_dir_names, input_all_res_column_name_number):
    trueNum = input_all_res_column_name_number["move validity"]

    # experiment_names
    experiment_names = []
    file_full_addr_list = []
    for dir_name in input_dir_names:
        file_full_addr = os.path.join(dir_name, "result_summary/FARSI_simple_run_0_1_all_reults.csv")
        file_full_addr_list.append(file_full_addr)
        experiment_name = get_experiments_name(file_full_addr, input_all_res_column_name_number)
        experiment_names.append(experiment_name)

    axis_font = {'size': '20'}
    fontSize = 20
    column_name_list = ["transformation_metric", "transformation_block_type", "move name", "comm_comp", "architectural principle", "high level optimization name", "exact optimization name", "neighbouring design space size"]
    #column_name_list = ["transformation_metric", "move name"]#, "comm_comp", "architectural principle", "high level optimization name", "exact optimization name", "neighbouring design space size"]
    #column_name = "move name"
    # initialize the dictionary
    column_column_value_experiment_frequency_dict = {}
    for column_name in column_name_list:
        column_value_experiment_frequency_dict = {}
        # get all possible the values of interest
        all_values = get_all_col_values_of_a_folders(input_dir_names, input_all_res_column_name_number, column_name)
        columne_number = all_res_column_name_number[column_name]
        for column in all_values:
            column_value_experiment_frequency_dict[column] = {}

        # get all the data
        for file_full_addr in file_full_addr_list:
            with open(file_full_addr, newline='') as csvfile:
                resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
                experiment_name = get_experiments_name( file_full_addr, input_all_res_column_name_number)
                for column_value in all_values:
                    column_value_experiment_frequency_dict[column_value][experiment_name] = 0

                for i, row in enumerate(resultReader):
                    #if row[trueNum] != "True":
                    #    continue
                    if i > 1:
                        try:

                            # the following for workload awareness
                            #if row[all_res_column_name_number["move name"]] == "identity":
                            #    continue
                            #if row[all_res_column_name_number["architectural principle"]] == "spatial_locality":
                            #    continue


                            col_value = row[columne_number]
                            col_values = col_value.split(";")
                            for col_val in col_values:
                                if "=" in col_val:
                                    val_splitted = col_val.split("=")
                                    column_value_experiment_frequency_dict[val_splitted[0]][experiment_name] += float(val_splitted[1])
                                else:
                                    column_value_experiment_frequency_dict[col_val][experiment_name] += 1
                        except:
                            print("what")

        total_cnt = {}
        for el in column_value_experiment_frequency_dict.values():
            for exp, values in el.items():
                if exp not in total_cnt.keys():
                    total_cnt[exp] = 0
                total_cnt[exp] += values

        for col_val, exp_vals in column_value_experiment_frequency_dict.items():
            for exp, values in exp_vals.items():
                column_value_experiment_frequency_dict[col_val][exp] = column_value_experiment_frequency_dict[col_val][exp]
                column_value_experiment_frequency_dict[col_val][exp] /= total_cnt[exp] # normalize

        # prepare for plotting and plot
        # plt.figure(figsize=(10, 8))
        index = experiment_names
        plotdata = pd.DataFrame(column_value_experiment_frequency_dict, index=index)
        plotdata.plot(kind='bar', stacked=True, figsize=(12, 10))
        plt.rc('font', **axis_font)
        plt.xlabel("experiments", **axis_font)
        plt.ylabel(column_name, **axis_font)
        plt.xticks(fontsize=fontSize, rotation=45)
        plt.yticks(fontsize=fontSize)
        plt.title("experiment vs " + column_name, **axis_font)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=fontSize)
        # dump in the top folder
        output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
        output_dir = os.path.join(output_base_dir, "cross_workloads/nav_breakdown")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,'_'.join(column_name.split(" "))+".png"), bbox_inches='tight')
        # plt.show()
        plt.close('all')
        column_column_value_experiment_frequency_dict[column_name] = copy.deepcopy(column_value_experiment_frequency_dict)

    """
    # multi-stack plot here
    index = experiment_names
    plotdata = pd.DataFrame(column_column_value_experiment_frequency_dict, index=index)

    df_g = plotdata.groupby(["transformation_metric", "move name"])
    plotdata.plot(kind='bar', stacked=True, figsize=(12, 10))
    plt.rc('font', **axis_font)
    plt.xlabel("experiments", **axis_font)
    plt.ylabel(column_name, **axis_font)
    plt.xticks(fontsize=fontSize, rotation=45)
    plt.yticks(fontsize=fontSize)
    plt.title("experiment vs " + column_name, **axis_font)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=fontSize)
    # dump in the top folder
    output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
    output_dir = os.path.join(output_base_dir, "cross_workloads/nav_breakdown")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,'column____'.join(column_name.split(" "))+".png"), bbox_inches='tight')
    # plt.show()
    plt.close('all')
    """
    return column_column_value_experiment_frequency_dict

def plot_codesign_nav_breakdown_cross_workload_for_paper(input_dir_names, input_all_res_column_name_number):
    trueNum = input_all_res_column_name_number["move validity"]

    # experiment_names
    experiment_names = []
    file_full_addr_list = []
    for dir_name in input_dir_names:
        file_full_addr = os.path.join(dir_name, "result_summary/FARSI_simple_run_0_1_all_reults.csv")
        file_full_addr_list.append(file_full_addr)
        experiment_name = get_experiments_name(file_full_addr, input_all_res_column_name_number)
        """
        Ying: the following lines are added to make the names clearer in the plottings
        """
        if experiment_name[0] == 'a':
            experiment_name = "Audio"
        elif experiment_name[0] == 'h':
            experiment_name = "CAVA"
        elif experiment_name[0] == 'e':
            experiment_name = "ED"
        """
        Ying: adding finished
        """
        experiment_names.append(experiment_name)
    experiment_names.sort()

    axis_font = {'size': '25'}
    fontSize = 25
    column_name_list = ["transformation_metric", "transformation_block_type", "move name", "comm_comp", "architectural principle", "high level optimization name", "exact optimization name", "neighbouring design space size"]
    #column_name_list = ["transformation_metric", "move name"]#, "comm_comp", "architectural principle", "high level optimization name", "exact optimization name", "neighbouring design space size"]
    #column_name = "move name"
    # initialize the dictionary
    column_column_value_experiment_frequency_dict = {}
    for column_name in column_name_list:
        column_value_experiment_frequency_dict = {}
        # get all possible the values of interest
        all_values = get_all_col_values_of_a_folders(input_dir_names, input_all_res_column_name_number, column_name)
        columne_number = input_all_res_column_name_number[column_name]
        for column in all_values:
            """
            Ying: the following lines are added for "IC", "Mem", and "PE"
            """
            if column_name == "transformation_block_type":
                if column == "ic":
                    column = "NoC"
                elif column == "mem":
                    column = "Mem"
                elif column == "pe":
                    column = "PE"

            if column_name == "architectural principle":
                if column == "identity" or column == "spatial_locality":
                    continue
                elif column == "task_level_parallelism":
                    column = "TLP"
                elif column == "loop_level_parallelism":
                    column = "LLP"
                elif column == "customization":
                    column = "Customization"
            """
            Ying: adding finished
            """
            column_value_experiment_frequency_dict[column] = {}

        # get all the data
        for file_full_addr in file_full_addr_list:
            with open(file_full_addr, newline='') as csvfile:
                resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
                experiment_name = get_experiments_name( file_full_addr, input_all_res_column_name_number)
                """
                Ying: the following lines are added to make the names clearer in the plottings
                """
                if experiment_name[0] == 'a':
                    experiment_name = "Audio"
                elif experiment_name[0] == 'h':
                    experiment_name = "CAVA"
                elif experiment_name[0] == 'e':
                    experiment_name = "ED"
                """
                Ying: adding finished
                """
                for column_value in all_values:
                    """
                    Ying: the following lines are added for "IC", "Mem", and "PE"
                    """
                    if column_name == "transformation_block_type":
                        if column_value == "ic":
                            column_value = "NoC"
                        elif column_value == "mem":
                            column_value = "Mem"
                        elif column_value == "pe":
                            column_value = "PE"

                    if column_name == "architectural principle":
                        if column_value == "identity" or column_value == "spatial_locality":
                            continue
                        elif column_value == "task_level_parallelism":
                            column_value = "TLP"
                        elif column_value == "loop_level_parallelism":
                            column_value = "LLP"
                        elif column_value == "customization":
                            column_value = "Customization"
                    """
                    Ying: adding finished
                    """
                    column_value_experiment_frequency_dict[column_value][experiment_name] = 0

                for i, row in enumerate(resultReader):
                    if row[trueNum] != "True":
                        continue
                    if i > 1:
                        try:

                            # the following for workload awareness
                            if row[input_all_res_column_name_number["architectural principle"]] == "spatial_locality" or row[all_res_column_name_number["architectural principle"]] == "identity":
                                continue

                            col_value = row[columne_number]
                            col_values = col_value.split(";")
                            for col_val in col_values:
                                if "=" in col_val:
                                    val_splitted = col_val.split("=")
                                    column_value_experiment_frequency_dict[val_splitted[0]][experiment_name] += float(val_splitted[1])
                                else:
                                    """
                                    Ying: the following lines are added for "IC", "Mem", and "PE"
                                    """
                                    if column_name == "transformation_block_type":
                                        if col_val == "ic":
                                            col_val = "NoC"
                                        elif col_val == "mem":
                                            col_val = "Mem"
                                        elif col_val == "pe":
                                            col_val = "PE"

                                    if column_name == "architectural principle":
                                        if col_val == "identity" or col_val == "spatial_locality":
                                            continue
                                        elif col_val == "task_level_parallelism":
                                            col_val = "TLP"
                                        elif col_val == "loop_level_parallelism":
                                            col_val = "LLP"
                                        elif col_val == "customization":
                                            col_val = "Customization"
                                    """
                                    Ying: adding finished
                                    """
                                    column_value_experiment_frequency_dict[col_val][experiment_name] += 1
                        except:
                            print("what")

        total_cnt = {}
        for el in column_value_experiment_frequency_dict.values():
            for exp, values in el.items():
                if exp not in total_cnt.keys():
                    total_cnt[exp] = 0
                total_cnt[exp] += values

        for col_val, exp_vals in column_value_experiment_frequency_dict.items():
            for exp, values in exp_vals.items():
                column_value_experiment_frequency_dict[col_val][exp] = column_value_experiment_frequency_dict[col_val][exp]
                if column_name != "architectural principle" and column_name != "comm_comp" and total_cnt[exp] != 0: # Ying: add to get rid of normalization for the two plottings
                    column_value_experiment_frequency_dict[col_val][exp] /= total_cnt[exp] # normalize

        # prepare for plotting and plot
        # plt.figure(figsize=(6, 6))
        index = experiment_names
        column_value_experiment_frequency_dict = dict(sorted(column_value_experiment_frequency_dict.items(), key=lambda x: x[0].lower()))
        print(column_value_experiment_frequency_dict)
        plotdata = pd.DataFrame(column_value_experiment_frequency_dict, index=index)
        # tempC, tempE, tempA= plotdata.iloc[0].copy(), plotdata.iloc[1].copy(), plotdata.iloc[2].copy()
        # plotdata.iloc[0] = tempA
        # plotdata.iloc[1] = tempC
        # plotdata.iloc[2] = tempE
        print(plotdata)
        color_list = ["mediumseagreen", "gold", "tomato"]
        plotdata.plot(kind='bar', stacked=True, figsize=(6, 6.6), color=color_list) # Ying: (6, 7) for arch principle
        plt.rc('font', **axis_font)
        plt.xlabel("Workloads", **axis_font)
        # plt.ylabel(column_name, **axis_font)  # Ying: replace with the following lines
        """
        Ying: set the ylabel acordingly
        """
        if column_name == "architectural principle" or column_name == "comm_comp":
            plt.ylabel("Iteration Count", **axis_font)
        else:
            plt.ylabel("Normalized Iteration Portion", **axis_font)
        """
        Ying: adding finished
        """
        plt.xticks(fontsize=fontSize, rotation=0)   # Ying: the original one was 45)
        plt.yticks(fontsize=fontSize)
        if column_name == "architectural principle" or column_name == "comm_comp":
            plt.yticks(np.arange(0, 200, 25.0), fontsize=fontSize)  # Ying: change according to the graph ("architectural principle", "comm_comp")
        # plt.title("experiment vs " + column_name, **axis_font)    # Ying: comment it out as discussed
        # plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=fontSize)    # Ying: replaced with the following line
        plt.legend(bbox_to_anchor=(0.5, 1.3), loc='upper center', fontsize=fontSize, ncol=2, borderpad=0) # Ying: change according to the graph ("architectural principle", "comm_comp", "Normalized Iteration Portion")
        # dump in the top folder
        output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
        output_dir = os.path.join(output_base_dir, "cross_workloads/nav_breakdown")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,'_'.join(column_name.split(" "))+".png"), bbox_inches='tight')
        # plt.show()
        plt.close('all')
        column_column_value_experiment_frequency_dict[column_name] = copy.deepcopy(column_value_experiment_frequency_dict)

    """
    # multi-stack plot here
    index = experiment_names
    plotdata = pd.DataFrame(column_column_value_experiment_frequency_dict, index=index)

    df_g = plotdata.groupby(["transformation_metric", "move name"])
    plotdata.plot(kind='bar', stacked=True, figsize=(12, 10))
    plt.rc('font', **axis_font)
    plt.xlabel("experiments", **axis_font)
    plt.ylabel(column_name, **axis_font)
    plt.xticks(fontsize=fontSize, rotation=45)
    plt.yticks(fontsize=fontSize)
    plt.title("experiment vs " + column_name, **axis_font)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=fontSize)
    # dump in the top folder
    output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
    output_dir = os.path.join(output_base_dir, "cross_workloads/nav_breakdown")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,'column____'.join(column_name.split(" "))+".png"), bbox_inches='tight')
    # plt.show()
    plt.close('all')
    """
    return column_column_value_experiment_frequency_dict


# the function to plot distance to goal vs. iteration cnt
def plotDistToGoalVSitr(input_dir_names, all_res_column_name_number):
    itrColNum = all_res_column_name_number["iteration cnt"]
    distColNum = all_res_column_name_number["dist_to_goal_non_cost"]
    trueNum  =  all_res_column_name_number["move validity"]

    experiment_itr_dist_to_goal_dict = {}
    # iterate through directories, get data and store in a dictionary
    for dir_name in input_dir_names:
        itr = []
        distToGoal = []
        file_full_addr = os.path.join(dir_name, "result_summary/FARSI_simple_run_0_1_all_reults.csv")
        with open(file_full_addr, newline='') as csvfile:
            resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
            experiment_name = get_experiments_name(file_full_addr, all_res_column_name_number)
            for i, row in enumerate(resultReader):
                if row[trueNum] != "True":
                    continue
                if i > 1:
                    itr.append(int(row[itrColNum]))
                    distToGoal.append(float(row[distColNum]))

            experiment_itr_dist_to_goal_dict[experiment_name] = (itr[:], distToGoal[:])

    plt.figure()
    # iterate and plot
    for experiment_name, value in experiment_itr_dist_to_goal_dict.items():
        itr, distToGoal = value[0], value[1]
        if len(itr) == 0 or len(distToGoal) == 0: # no valid move
            continue
        plt.plot(itr, distToGoal, label=experiment_name)
        plt.xlabel("Iteration Cnt")
        plt.ylabel("Distance to Goal")
        plt.title("Distance to Goal vs. Iteration Cnt")

    # decide on the output dir
    if len(input_dir_names) == 1:
        output_dir = input_dir_names[0]
    else:
        # dump in the top folder
        output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
        output_dir  = os.path.join(output_base_dir, "cross_workloads")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir, "distToGoalVSitr.png"))
    # plt.show()
    plt.close('all')


# the function to plot distance to goal vs. iteration cnt
def plotRefDistToGoalVSitr(dirName, fileName, itrColNum, refDistColNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        itr = []
        refDistToGoal = []

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i > 1:
                itr.append(int(row[itrColNum]))
                refDistToGoal.append(float(row[refDistColNum]))

        plt.figure()
        plt.plot(itr, refDistToGoal)
        plt.xlabel("Iteration Cnt")
        plt.ylabel("Reference Design Distance to Goal")
        plt.title("Reference Design Distance to Goal vs. Iteration Cnt")
        plt.savefig(dirName + fileName + "/refDistToGoalVSitr-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to do the zonal partitioning
def zonalPartition(comparedValue, zoneNum, maxValue):
    unit = maxValue / zoneNum

    if comparedValue > maxValue:
        return zoneNum - 1
    
    if comparedValue < 0:
        return 0

    for i in range(0, zoneNum):
        if comparedValue <= unit * (i + 1):
            return i

    raise Exception("zonalPartition is fed by a strange value! maxValue: " + str(maxValue) + "; comparedValue: " + str(comparedValue))

# the function to plot simulation time vs. move name in a zonal format
def plotSimTimeVSmoveNameZoneDist(dirName, fileName, zoneNum, moveColNum, distColNum, simColNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        splitSwapSim = np.zeros(zoneNum, dtype = float)
        splitSim = np.zeros(zoneNum, dtype = float)
        migrateSim = np.zeros(zoneNum, dtype = float)
        swapSim = np.zeros(zoneNum, dtype = float)
        tranSim = np.zeros(zoneNum, dtype = float)
        routeSim = np.zeros(zoneNum, dtype = float)
        identitySim = np.zeros(zoneNum, dtype = float)

        maxDist = 0

        index = []
        for i in range(0, zoneNum):
            index.append(i)

        for i, row in enumerate(resultReader):
            # print('"' + row[trueNum] + '"\t"' + row[moveColNum] + '"\t"' + row[distColNum] + '"\t"' + row[simColNum] + '"')
            if row[trueNum] != "True":
                continue

            if i == 2:
                maxDist = float(row[distColNum])

            if i > 1:
                if row[moveColNum] == "split_swap":
                    splitSwapSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[moveColNum] == "split":
                    splitSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[moveColNum] == "migrate":
                    migrateSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[moveColNum] == "swap":
                    swapSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[moveColNum] == "transfer":
                    tranSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[moveColNum] == "routing":
                    routeSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[moveColNum] == "identity":
                    identitySim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                else:
                    raise Exception("move name is not split_swap or split or migrate or swap or transfer or routing or identity! The new type: " + row[moveColNum])
        
        # plt.figure()
        plotdata = pd.DataFrame({
            "split_swap":splitSwapSim,
            "split":splitSim,
            "migrate":migrateSim,
            "swap":swapSim,
            "transfer":tranSim,
            "routing":routeSim,
            "identity":identitySim
        }, index = index
        )
        plotdata.plot(kind = 'bar', stacked = True)
        plt.xlabel("Zone decided by the max distance to goal")
        plt.ylabel("Simulation Time")
        plt.title("Simulation Time in Each Zone based on Move Name")
        plt.savefig(dirName + fileName + "/simTimeVSmoveNameZoneDist-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot move generation time vs. move name in a zonal format
def plotMovGenTimeVSmoveNameZoneDist(dirName, fileName, zoneNum, moveColNum, distColNum, movGenColNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        splitSwapMov = np.zeros(zoneNum, dtype = float)
        splitMov = np.zeros(zoneNum, dtype = float)
        migrateMov = np.zeros(zoneNum, dtype = float)
        swapMov = np.zeros(zoneNum, dtype = float)
        tranMov = np.zeros(zoneNum, dtype = float)
        routeMov = np.zeros(zoneNum, dtype = float)
        identityMov = np.zeros(zoneNum, dtype = float)

        maxDist = 0

        index = []
        for i in range(0, zoneNum):
            index.append(i)

        for i, row in enumerate(resultReader):
            # print('"' + row[trueNum] + '"\t"' + row[moveColNum] + '"\t"' + row[distColNum] + '"\t"' + row[movGenColNum] + '"')
            if row[trueNum] != "True":
                continue

            if i == 2:
                maxDist = float(row[distColNum])

            if i > 1:
                if row[moveColNum] == "split_swap":
                    splitSwapMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[moveColNum] == "split":
                    splitMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[moveColNum] == "migrate":
                    migrateMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[moveColNum] == "swap":
                    swapMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[moveColNum] == "transfer":
                    tranMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[moveColNum] == "routing":
                    routeMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[moveColNum] == "identity":
                    identityMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                else:
                    raise Exception("move name is not split_swap or split or migrate or swap or transfer of routing or identity! The new type: " + row[moveColNum])
        
        # plt.figure()
        plotdata = pd.DataFrame({
            "split_swap":splitSwapMov,
            "split":splitMov,
            "migrate":migrateMov,
            "swap":swapMov,
            "transfer":tranMov,
            "routing":routeMov,
            "identity":identityMov
        }, index = index
        )
        plotdata.plot(kind = 'bar', stacked = True)
        plt.xlabel("Zone decided by the max distance to goal")
        plt.ylabel("Move Generation Time")
        plt.title("Move Generation Time in Each Zone based on Move Name")
        plt.savefig(dirName + fileName + "/movGenTimeVSmoveNameZoneDist-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot simulation time vs. comm_comp in a zonal format
def plotSimTimeVScommCompZoneDist(dirName, fileName, zoneNum, commcompColNum, distColNum, simColNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        commSim = np.zeros(zoneNum, dtype = float)
        compSim = np.zeros(zoneNum, dtype = float)

        maxDist = 0
        index = []
        for i in range(0, zoneNum):
            index.append(i)

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i == 2:
                maxDist = float(row[distColNum])

            if i > 1:
                if row[commcompColNum] == "comm":
                    commSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[commcompColNum] == "comp":
                    compSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                else:
                    raise Exception("comm_comp is not giving comm or comp! The new type: " + row[colNum])
        
        # plt.figure()
        plotdata = pd.DataFrame({
            "comm":commSim,
            "comp":compSim
        }, index = index
        )
        plotdata.plot(kind = 'bar', stacked = True)
        plt.xlabel("Zone decided by the max distance to goal")
        plt.ylabel("Simulation Time")
        plt.title("Simulation Time in Each Zone based on comm_comp")
        plt.savefig(dirName + fileName + "/simTimeVScommCompZoneDist-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot simulation time vs. comm_comp in a zonal format
def plotMovGenTimeVScommCompZoneDist(dirName, fileName, zoneNum, commcompColNum, distColNum, movGenColNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        commMov = np.zeros(zoneNum, dtype = float)
        compMov = np.zeros(zoneNum, dtype = float)

        maxDist = 0
        index = []
        for i in range(0, zoneNum):
            index.append(i)

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i == 2:
                maxDist = float(row[distColNum])

            if i > 1:
                if row[commcompColNum] == "comm":
                    commMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[commcompColNum] == "comp":
                    compMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                else:
                    raise Exception("comm_comp is not giving comm or comp! The new type: " + row[colNum])
        
        # plt.figure()
        plotdata = pd.DataFrame({
            "comm":commMov,
            "comp":compMov
        }, index = index
        )
        plotdata.plot(kind = 'bar', stacked = True)
        plt.xlabel("Zone decided by the max distance to goal")
        plt.ylabel("Move Generation Time")
        plt.title("Move Generation Time in Each Zone based on comm_comp")
        plt.savefig(dirName + fileName + "/movGenTimeVScommCompZoneDist-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot simulation time vs. high level optimization name in a zonal format
def plotSimTimeVShighLevelOptZoneDist(dirName, fileName, zoneNum, optColNum, distColNum, simColNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        topoSim = np.zeros(zoneNum, dtype = float)
        tunSim = np.zeros(zoneNum, dtype = float)
        mapSim = np.zeros(zoneNum, dtype = float)
        idenOptSim = np.zeros(zoneNum, dtype = float)

        maxDist = 0
        index = []
        for i in range(0, zoneNum):
            index.append(i)

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i == 2:
                maxDist = float(row[distColNum])

            if i > 1:
                if row[optColNum] == "topology":
                    topoSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[optColNum] == "customization":
                    tunSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[optColNum] == "mapping":
                    mapSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[optColNum] == "identity":
                    idenOptSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                else:
                    raise Exception("high level optimization name is not giving topology or customization or mapping or identity! The new type: " + row[optColNum])
        
        # plt.figure()
        plotdata = pd.DataFrame({
            "topology":topoSim,
            "customization":tunSim,
            "mapping":mapSim,
            "identity":idenOptSim
        }, index = index
        )
        plotdata.plot(kind = 'bar', stacked = True)
        plt.xlabel("Zone decided by the max distance to goal")
        plt.ylabel("Simulation Time")
        plt.title("Simulation Time in Each Zone based on Optimation Name")
        plt.savefig(dirName + fileName + "/simTimeVShighLevelOptZoneDist-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot simulation time vs. high level optimization name in a zonal format
def plotMovGenTimeVShighLevelOptZoneDist(dirName, fileName, zoneNum, optColNum, distColNum, movGenColNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        topoMov = np.zeros(zoneNum, dtype = float)
        tunMov = np.zeros(zoneNum, dtype = float)
        mapMov = np.zeros(zoneNum, dtype = float)
        idenOptMov = np.zeros(zoneNum, dtype = float)

        maxDist = 0
        index = []
        for i in range(0, zoneNum):
            index.append(i)

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i == 2:
                maxDist = float(row[distColNum])

            if i > 1:
                if row[optColNum] == "topology":
                    topoMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[optColNum] == "customization":
                    tunMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[optColNum] == "mapping":
                    mapMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[optColNum] == "identity":
                    idenOptMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                else:
                    raise Exception("high level optimization name is not giving topology or customization or mapping or identity! The new type: " + row[optColNum])
        
        # plt.figure()
        plotdata = pd.DataFrame({
            "topology":topoMov,
            "customization":tunMov,
            "mapping":mapMov,
            "identity":idenOptMov
        }, index = index
        )
        plotdata.plot(kind = 'bar', stacked = True)
        plt.xlabel("Zone decided by the max distance to goal")
        plt.ylabel("Transformation Generation Time")
        plt.title("Transformation Generation Time in Each Zone based on Optimization Name")
        plt.savefig(dirName + fileName + "/movGenTimeVShighLevelOptZoneDist-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot simulation time vs. architectural principle in a zonal format
def plotSimTimeVSarchVarImpZoneDist(dirName, fileName, zoneNum, archColNum, distColNum, simColNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        paraSim = np.zeros(zoneNum, dtype = float)
        custSim = np.zeros(zoneNum, dtype = float)
        localSim = np.zeros(zoneNum, dtype = float)
        idenImpSim = np.zeros(zoneNum, dtype = float)

        maxDist = 0
        index = []
        for i in range(0, zoneNum):
            index.append(i)

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i == 2:
                maxDist = float(row[distColNum])

            if i > 1:
                if row[archColNum] == "parallelization":
                    paraSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[archColNum] == "customization":
                    custSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[archColNum] == "locality":
                    localSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[archColNum] == "identity":
                    idenImpSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                else:
                    raise Exception("architectural principle is not giving parallelization or customization or locality or identity! The new type: " + row[archColNum])
        
        # plt.figure()
        plotdata = pd.DataFrame({
            "parallelization":paraSim,
            "customization":custSim,
            "locality":localSim,
            "identity":idenImpSim
        }, index = index
        )
        plotdata.plot(kind = 'bar', stacked = True)
        plt.xlabel("Zone decided by the max distance to goal")
        plt.ylabel("Simulation Time")
        plt.title("Simulation Time in Each Zone based on Architectural Principle")
        plt.savefig(dirName + fileName + "/simTimeVSarchVarImpZoneDist-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot simulation time vs. architectural principle in a zonal format
def plotMovGenTimeVSarchVarImpZoneDist(dirName, fileName, zoneNum, archColNum, distColNum, movGenColNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        paraMov = np.zeros(zoneNum, dtype = float)
        custMov = np.zeros(zoneNum, dtype = float)
        localMov = np.zeros(zoneNum, dtype = float)
        idenImpMov = np.zeros(zoneNum, dtype = float)

        maxDist = 0
        index = []
        for i in range(0, zoneNum):
            index.append(i)

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i == 2:
                maxDist = float(row[distColNum])

            if i > 1:
                if row[archColNum] == "parallelization":
                    paraMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[archColNum] == "customization":
                    custMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[archColNum] == "locality":
                    localMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[archColNum] == "identity":
                    idenImpMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                else:
                    raise Exception("architectural principle is not giving parallelization or customization or locality or identity! The new type: " + row[archColNum])
        
        # plt.figure()
        plotdata = pd.DataFrame({
            "parallelization":paraMov,
            "customization":custMov,
            "locality":localMov,
            "identity":idenImpMov
        }, index = index
        )
        plotdata.plot(kind = 'bar', stacked = True)
        plt.xlabel("Zone decided by the max distance to goal")
        plt.ylabel("Tranformation Generation Time")
        plt.title("Tranformation Generation Time in Each Zone based on Architectural Principle")
        plt.savefig(dirName + fileName + "/movGenTimeVSarchVarImpZoneZoneDist-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot convergence vs. iteration cnt, system block count, and routing complexity in 3d
def plotBudgets3d(dirName, subDirName):
    newDirName = dirName + "/"+ subDirName + "/"
    if os.path.exists(newDirName + "/figures"):
        shutil.rmtree(newDirName + "/figures")
    resultList = os.listdir(newDirName)
    latBudgets = []
    powBudgets = []
    areaBudgets = []
    itrValues = []
    cntValues = []
    routingValues = []
    workloads = []
    for j, fileName in enumerate(resultList):
        with open(newDirName + fileName + "/result_summary/FARSI_simple_run_0_1.csv", newline='') as csvfile:
            resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for i, row in enumerate(resultReader):
                if i == 1:
                    itrValues.append(int(row[columnNum(newDirName, fileName, "iteration cnt", "simple")]))
                    cntValues.append(int(row[columnNum(newDirName, fileName, "system block count", "simple")]))
                    routingValues.append(float(row[columnNum(newDirName, fileName, "routing complexity", "simple")]))
                    powBudgets.append(float(row[columnNum(newDirName, fileName, "power_budget", "simple")]))
                    areaBudgets.append(float(row[columnNum(newDirName, fileName, "area_budget", "simple")]))
                    lat = row[int(columnNum(newDirName, fileName, "latency_budget", "simple"))][:-1]
                    latDict = dict(item.split("=") for item in lat.split(";"))
                    if j == 0:
                        for k in range(0, len(latDict)):
                            latBudgets.append([])
                            workloads.append(list(latDict.keys())[k])
                    latList = list(latDict.values())
                    for k in range(0, len(latList)):
                        latBudgets[k].append(float(latList[k]))

    m = ['o', 'x', '^', 's', 'd', '+', 'v', '<', '>']
    axis_font = {'size': '10'}
    fontSize = 10
    os.mkdir(newDirName + "figures")
    fig_budget_itr = plt.figure(figsize=(12, 12))
    plt.rc('font', **axis_font)
    ax_itr = fig_budget_itr.add_subplot(projection='3d')
    for i in range(0, len(latBudgets)):
        img = ax_itr.scatter3D(powBudgets, areaBudgets, latBudgets[i], c=itrValues, cmap="bwr", marker=m[i], s=80, label='{0}'.format(workloads[i]))
        for j in range(0, len(latBudgets[i])):
            coordinate = str(itrValues[j])
            ax_itr.text(powBudgets[j], areaBudgets[j], latBudgets[i][j], '%s' % coordinate, size=fontSize)
        break
    ax_itr.set_xlabel("Power Budget")
    ax_itr.set_ylabel("Area Budget")
    ax_itr.set_zlabel("Latency Budget")
    ax_itr.legend()
    cbar_itr = fig_budget_itr.colorbar(img, aspect = 40)
    cbar_itr.set_label("Number of Iterations", rotation = 270)
    plt.title("{Power Budget, Area Budget, Latency Budget} VS Iteration Cnt: " + subDirName)
    plt.tight_layout()
    plt.savefig(newDirName + "figures/budgetVSitr-" + subDirName + ".png")
    # plt.show()
    plt.close('all')

    fig_budget_blkcnt = plt.figure(figsize=(12, 12))
    plt.rc('font', **axis_font)
    ax_blkcnt = fig_budget_blkcnt.add_subplot(projection='3d')
    for i in range(0, len(latBudgets)):
        img = ax_blkcnt.scatter3D(powBudgets, areaBudgets, latBudgets[i], c=cntValues, cmap="bwr", marker=m[i], s=80, label='{0}'.format(workloads[i]))
        for j in range(0, len(latBudgets[i])):
            coordinate = str(cntValues[j])
            ax_blkcnt.text(powBudgets[j], areaBudgets[j], latBudgets[i][j], '%s' % coordinate, size=fontSize)
        break
    ax_blkcnt.set_xlabel("Power Budget")
    ax_blkcnt.set_ylabel("Area Budget")
    ax_blkcnt.set_zlabel("Latency Budget")
    ax_blkcnt.legend()
    cbar = fig_budget_blkcnt.colorbar(img, aspect=40)
    cbar.set_label("System Block Count", rotation=270)
    plt.title("{Power Budget, Area Budget, Latency Budget} VS System Block Count: " + subDirName)
    plt.tight_layout()
    plt.savefig(newDirName + "figures/budgetVSblkcnt-" + subDirName + ".png")
    # plt.show()
    plt.close('all')

    fig_budget_routing = plt.figure(figsize=(12, 12))
    plt.rc('font', **axis_font)
    ax_routing = fig_budget_routing.add_subplot(projection='3d')
    for i in range(0, len(latBudgets)):
        img = ax_routing.scatter3D(powBudgets, areaBudgets, latBudgets[i], c=cntValues, cmap="bwr", marker=m[i], s=80, label='{0}'.format(workloads[i]))
        for j in range(0, len(latBudgets[i])):
            coordinate = str(routingValues[j])
            ax_routing.text(powBudgets[j], areaBudgets[j], latBudgets[i][j], '%s' % coordinate, size=fontSize)
        break
    ax_routing.set_xlabel("Power Budget")
    ax_routing.set_ylabel("Area Budget")
    ax_routing.set_zlabel("Latency Budget")
    ax_routing.legend()
    cbar = fig_budget_routing.colorbar(img, aspect=40)
    cbar.set_label("System Block Count", rotation=270)
    plt.title("{Power Budget, Area Budget, Latency Budget} VS System Block Count: " + subDirName)
    plt.tight_layout()
    plt.savefig(newDirName + "figures/budgetVSroutingComplexity-" + subDirName + ".png")
    # plt.show()
    plt.close('all')

def get_experiment_dir_list(run_folder_name):
    workload_set_folder_list = os.listdir(run_folder_name)

    experiment_full_addr_list = []
    #  iterate and generate plots
    for workload_set_folder in workload_set_folder_list:
        # ignore irelevant files
        if workload_set_folder in config_plotting.ignore_file_names:
            continue

        # get experiment folder
        workload_set_full_addr = os.path.join(run_folder_name,workload_set_folder)
        folder_list = os.listdir(workload_set_full_addr)
        for experiment_name_relative_addr in folder_list:
            if experiment_name_relative_addr in config_plotting.ignore_file_names:
                continue
            experiment_full_addr_list.append(os.path.join(workload_set_full_addr, experiment_name_relative_addr))

    return experiment_full_addr_list


def find_the_most_recent_directory(top_dir):
    dirs = [os.path.join(top_dir, el) for el in os.listdir(top_dir)]
    dirs = list(filter(os.path.isdir, dirs))
    dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return dirs


def get_experiment_full_file_addr_list(experiment_full_dir_list, aggregate=False):
    if aggregate:
        file_name = 'result_summary/aggregate_all_results.csv'
    else:
        file_name = "result_summary/FARSI_simple_run_0_1.csv"
    results = []
    for el in experiment_full_dir_list:
        results.append(os.path.join(el, file_name))

    return results

######### RADHIKA PANDAS PLOTS ############
def simple_stack_bar_plot(output_dir):
    labels = ['1', '2', '3']
    customization = [13, 11.74, 11.55]
    LLP = [100, 49, 27]
    TLP = [2.0, 2.3, 2.14]
    degradation = [1.07, 1.25, 1.35]
    width = 0.35  # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()

    ax.bar(labels, customization, width,  label='customization')
    ax.bar(labels, LLP, width,  bottom=customization, label='LLP')
    ax.bar(labels, TLP, width,  bottom=np.add(LLP, customization).tolist(), label='TLP')

    ax.set_ylabel('Performance Driver')
    #ax.set_title()
    ax.legend()

    output_dir_ = os.path.join(output_dir, "drivers")
    if not os.path.exists(output_dir_):
        os.makedirs(output_dir_)
    plt.savefig(os.path.join(output_dir_, "performance_driver.png"))
    plt.show()



def grouped_barplot_varying_x(df, metric, metric_ylabel, varying_x, varying_x_labels, ax):
    # [[bar heights, errs for varying_x1], [heights, errs for varying_x2]...]
    #if metric in ["ip_cnt", "local_bus_avg_bus_width", "local_memory_total_area"]:
    #    print("ok")



    grouped_stats_list = []
    for x in varying_x:
        grouped_x = df.groupby([x])
        stats = grouped_x[metric].agg([np.mean, np.std])
        grouped_stats_list.append(stats)
    """
    for el in grouped_stats_list:
       #x = el.at[2, "mean"]
       diff = (((el.at[1, "mean"] - el.at[4, "mean"])/el.at[4, "mean"])*100)
       print (metric+":"+el.index.name +" : "+ str(diff))
    """

    if metric in [# "loop_unrolling_parallelism_speed_up_full_system", "customization_speed_up_full_system",
                  "task_level_parallelism_speed_up_full_system",
                  "interference_degradation_avg"]:

        print("metric is:" + metric)
        for el in grouped_stats_list:
            print(el)
    start_loc = 0
    bar_width = 0.15
    offset = 0
    # [[bar locations for varying_x1], [bar locs for varying_x2]...]
    grouped_bar_locs_list = []
    for x in varying_x:
        n_unique_varying_x = df[x].nunique()
        bound = (n_unique_varying_x-1) * (bar_width+offset)
        end_loc = start_loc+bound
        bar_locs = np.linspace(start_loc, end_loc, n_unique_varying_x)
        grouped_bar_locs_list.append(bar_locs)
        start_loc = end_loc + 2*bar_width

    #print(grouped_bar_locs_list)

    color = ["red", "orange", "green"]
    ctr = 0
    for x_i,x in enumerate(varying_x):
        ax.bar(
            grouped_bar_locs_list[x_i],
            grouped_stats_list[x_i]["mean"],
            width=bar_width,
            yerr=grouped_stats_list[x_i]["std"],
            color = color,
            label=metric_ylabel
        )
        ctr +=1
    cat_xticks = []
    cat_xticklabels = []

    xticks = []
    xticklabels = []
    for x_i,x in enumerate(varying_x):
        xticklabels.extend(grouped_stats_list[x_i].index.astype(float))
        xticks.extend(grouped_bar_locs_list[x_i])

        xticks_cat = grouped_bar_locs_list[x_i]
        xticks_cat_start = xticks_cat[0]
        xticks_cat_end = xticks_cat[-1]
        xticks_cat_mid = xticks_cat_start + (xticks_cat_end - xticks_cat_start) / 2

        cat_xticks.append(xticks_cat_mid)
        cat_xticklabels.append("\n\n" + varying_x_labels[x_i])

    xticks.extend(cat_xticks)
    xticklabels.extend(cat_xticklabels)

    ax.set_ylabel(metric_ylabel)
    #ax.set_xlabel(xlabel)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    return ax

def pie_chart(dir_names, all_res_column_name_number, case_study):


    file_full_addr = os.path.join(dir_names[0], "result_summary/FARSI_simple_run_0_1_all_reults.csv")
    column_name_number_dic = {}

    column_name_list = case_study[1]
    column_aggregate = {}
    for column_name in column_name_list:
        column_aggregate[column_name] = 0
        with open(file_full_addr, newline='') as csvfile:
            resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for i, row in enumerate(resultReader):
                if i > 1:
                    try:
                        column_aggregate[column_name] += float(row[all_res_column_name_number[column_name]])
                    except:
                        continue

    y = np.array(list(column_aggregate.values()))
    mylabels = list(column_aggregate.keys())

    plt.pie(y, labels=mylabels)
    plt.legend()

    output_base_dir = '/'.join(dir_names[0].split("/")[:-2])
    output_dir = os.path.join(output_base_dir, "pie_chart")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, case_study[0]+".png"))
    #plt.show()


    plt.show()

def pie_chart_for_paper(dir_names, all_res_column_name_number, case_study):


    file_full_addr = os.path.join(dir_names[0], "result_summary/FARSI_simple_run_0_1_all_reults.csv")
    column_name_number_dic = {}

    column_name_list = case_study[1]
    column_aggregate = {}
    for column_name in column_name_list:
        column_aggregate[column_name] = 0
        with open(file_full_addr, newline='') as csvfile:
            resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for i, row in enumerate(resultReader):
                if i > 1:
                    try:
                        column_aggregate[column_name] += float(row[all_res_column_name_number[column_name]])
                    except:
                        continue

    y = np.array(list(column_aggregate.values()))
    mylabels = list(column_aggregate.keys())

    axis_font = {'size': '28'}
    fontSize = 24
    if case_study[0] == "Performance Breakdown":
        plt.figure(figsize=(8, 6))
    elif case_study[0] == "Transformation_Generation_Breakdown":
        plt.figure(figsize=(8, 6))
    plt.rc('font', **axis_font)
    if mylabels == ['transformation generation time', 'simulation time', 'neighbour selection time']:
        mylabels = ['System Generation', 'Simulation', 'System Selection']
    elif mylabels == ['metric selection time', 'dir selection time', 'kernel selection time', 'block selection time', 'transformation selection time', 'design duplication time']:
        mylabels = ['Metric', 'Direction', 'Task', 'Block', 'Move', 'Design Duplication']
    plt.pie(y, autopct=lambda p: '{:1.1f}%'.format(p) if p > 0.8 else '')  # Ying: original: , labels=mylabels)
    if case_study[0] == "Performance Breakdown":
        plt.legend(mylabels, bbox_to_anchor=(0.5, 1.5), loc="upper center", ncol=1, fontsize=fontSize)
    elif case_study[0] == "Transformation_Generation_Breakdown":
        plt.legend(mylabels, bbox_to_anchor=(0.5, 1.5), loc="upper center", ncol=2, fontsize=fontSize)
    plt.tight_layout()

    output_base_dir = '/'.join(dir_names[0].split("/")[:-2])
    output_dir = os.path.join(output_base_dir, "pie_chart")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(output_dir)
    plt.savefig(os.path.join(output_dir, case_study[0]+".png"), bbox_inches='tight')
    #plt.show()


    plt.show()
    plt.close('all')

def pandas_plots(input_dir_names, all_results_files, metric):
    df = pd.concat((pd.read_csv(f) for f in all_results_files))

    #df = raw_df.loc[(raw_df["move validity"] == True)]
    #df["dist_to_goal_non_cost_delta"] = df["ref_des_dist_to_goal_non_cost"] - df["dist_to_goal_non_cost"]
    #df["local_traffic_ratio"] = np.divide(df["local_total_traffic"], df["local_total_traffic"] + df["global_total_traffic"])
    #metric = "global_memory_avg_freq"
    #metric_ylabel = "Global memory avg freq"
    #metric = "local_traffic_ratio"

    metric_ylabel = metric #"Local traffic ratio"


    varying_x = [
            "budget_scaling_latency",
            "budget_scaling_power",
            "budget_scaling_area",
    ]
    varying_x_labels = [
            "latency",
            "power",
            "area",
    ]


    fig, ax = plt.subplots(1)
    grouped_barplot_varying_x(
            df,
            metric, metric_ylabel,
            varying_x, varying_x_labels,
            ax
    )
    output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
    output_dir = os.path.join(output_base_dir, "panda_study/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, metric+".png"))
    #plt.show()
    plt.close('all')



    #fig.tight_layout(rect=[0, 0, 1, 1])
    #fig.savefig("/Users/behzadboro/Project_FARSI_dir/Project_FARSI_with_channels/data_collection/data/simple_run/27_point_coverage_zad/bleh.png")
    #plt.close(fig)

def grouped_barplot_varying_x_for_paper(df, metric, metric_ylabel, varying_x, varying_x_labels, ax):
    # [[bar heights, errs for varying_x1], [heights, errs for varying_x2]...]
    grouped_stats_list = []
    for x in varying_x:
        grouped_x = df.groupby([x])
        stats = grouped_x[metric].agg([np.mean, np.std])
        grouped_stats_list.append(stats)

    start_loc = 0
    bar_width = 0.15
    offset = 0  # Ying: original: 0.03
    # [[bar locations for varying_x1], [bar locs for varying_x2]...]
    grouped_bar_locs_list = []
    for x in varying_x:
        n_unique_varying_x = df[x].nunique()
        bound = (n_unique_varying_x-1) * (bar_width+offset)
        end_loc = start_loc+bound
        bar_locs = np.linspace(start_loc, end_loc, n_unique_varying_x)
        grouped_bar_locs_list.append(bar_locs)
        start_loc = end_loc + 2*bar_width

    # print(grouped_bar_locs_list)  # Ying: comment out for WTF

    color = ["mediumseagreen", "gold", "tomato"]
    legendLabel = ["1X", "2X", "4X"]
    ctr = 0
    coloredLocList=[[], [], []]
    coloredStatsValueList = [[], [], []]
    """
    Ying: add the following lines for the legend
    """
    for x_i,x in enumerate(varying_x):
        for i in range(0, len(grouped_bar_locs_list)):
            coloredLocList[x_i].append(grouped_bar_locs_list[i][x_i])
            coloredStatsValueList[i].append(grouped_stats_list[x_i]["mean"].to_numpy()[i])
    """
    Ying: adding finished
    """
    for x_i, x, in enumerate(varying_x):
        # ax.bar(
        #     grouped_bar_locs_list[x_i],
        #     grouped_stats_list[x_i]["mean"],
        #     width=bar_width,
        #     # yerr=grouped_stats_list[x_i]["std"],
        #     color = color,
        # )
        ax.bar(
            coloredLocList[x_i],
            coloredStatsValueList[x_i],
            width=bar_width,
            # yerr=grouped_stats_list[x_i]["std"],
            color=color[x_i],
            label=legendLabel[x_i]
        )
        ctr +=1
    cat_xticks = []
    cat_xticklabels = []

    xticks = []
    xticklabels = []
    """
    Ying: add the following lines to get rid of the numbers on the x-axis
    """
    for i in range(0, 9):
        xticklabels.append(' ')
    """
    Ying: adding finished
    """
    for x_i,x in enumerate(varying_x):
        # xticklabels.extend(grouped_stats_list[x_i].index.astype(float))   # Ying: comment out and leave them for legends
        xticks.extend(grouped_bar_locs_list[x_i])

        xticks_cat = grouped_bar_locs_list[x_i]
        xticks_cat_start = xticks_cat[0]
        xticks_cat_end = xticks_cat[-1]
        xticks_cat_mid = xticks_cat_start + (xticks_cat_end - xticks_cat_start) / 2

        cat_xticks.append(xticks_cat_mid)
        cat_xticklabels.append(varying_x_labels[x_i])   # Ying: the original code was: "\n\n" + varying_x_labels[x_i])

    fontSize = 28
    axis_font = {'size': '28'}
    xticks.extend(cat_xticks)
    xticklabels.extend(cat_xticklabels)

    ax.set_ylabel(metric_ylabel, fontsize=fontSize) # Ying: add fontsize
    ax.yaxis.set_label_coords(-0.15, 0.6)
    #ax.set_xlabel(xlabel)
    ax.set_xticks(xticks)
    ax.legend(legendLabel, bbox_to_anchor=(0.5, 1.35), loc="upper center", fontsize=fontSize-4, ncol=3) # Ying: test the way to add legends  # 1.35 for the figures with 1e8, etc., 1.2 for the figures without that
    ax.set_xticklabels(xticklabels, fontsize=fontSize)  # Ying: add fontsize

    return ax



def find_average_iteration_to_distance(input_dir_names, all_results_files, intrested_distance_to_consider):
    # iterate and collect all the data
    heuristic_dist_iter_all = {}
    for result_file in all_results_files:

        df = pd.concat((pd.read_csv(f) for f in [result_file]))
        ht = df['heuristic_type']
        if list(ht)[0] not in heuristic_dist_iter_all:
            heuristic_dist_iter_all[list(ht)[0]] = []
        dist_to_goal_non_cost = df["ref_des_dist_to_goal_non_cost"]


        dist_itr = OrderedDict()
        for intrested_dist in intrested_distance_to_consider:
            for itr, dist in enumerate(dist_to_goal_non_cost) :
                if dist < intrested_dist:
                    dist_itr[intrested_dist] = itr
                    break
        #if len(list(dist_itr.values())) <= 2:
        #    continue
        heuristic_dist_iter_all[list(ht)[0]].append(dist_itr)



    # per heuristic reduce
    heuristic_dist_iter_avg = {}
    for heuristic, values in heuristic_dist_iter_all.items():
        aggregate = OrderedDict()
        for val in values:
            for dist, itr in val.items():
                if dist in aggregate:
                    aggregate[dist].append(itr)
                else:
                    aggregate[dist] = [itr]

        if heuristic not in heuristic_dist_iter_avg.keys():
            heuristic_dist_iter_avg[heuristic] = OrderedDict()
        for dist, all_itr in aggregate.items():
            heuristic_dist_iter_avg[heuristic][dist] = sum(all_itr)/len(all_itr)

    """
    # compare heuristics
    speedup = {}
    for heuristic in heuristic_dist_iter_avg.keys():
        speedup[heuristic] = {}
        for dist in intrested_distance_to_consider:
            if dist not in heuristic_dist_iter_avg[heuristic].keys():
                speedup[heuristic][dist] = float('inf')
            else:
                speedup[heuristic][dist] = heuristic_dist_iter_avg[heuristic][dist]/heuristic_dist_iter_avg["FARSI"][dist]

    speedup_sorted = {}
    for heuristic in speedup.keys():
        keys = sorted(speedup[heuristic].keys())
        sorted_ = sorted({key: speedup[heuristic][key] for key in keys}.items())
        speedup_sorted[heuristic] = sorted_
    """


    return heuristic_dist_iter_avg

def heuristic_scaling_parsing(input_dir_names, all_results_files):
    max_min_dist  = 30

    # iterate and collect data for the closest distance
    heuristic_closest_dist_iter_all = {}
    heuristic_avg_neighbourhood_size_all = {}
    power_budget_ref = .008737

    for result_file in all_results_files:
        df = pd.concat((pd.read_csv(f) for f in [result_file]))
        dist_to_goal_non_cost = df["ref_des_dist_to_goal_non_cost"]
        task_count = len((df["latency"][1]).split(";")[:-1])
        budget_scaling = float(df["power_budget"][1])/(power_budget_ref*task_count)
        neighbourhood_sizes = []
        for el in df["neighbouring design space size"]:
            blah = type(el)
            if (not el == 0) and not (math.isnan(el)):
                neighbourhood_sizes.append(el)

        # get avg neighbourhood size for the entire run
        avg_neighbourhood_size = sum(neighbourhood_sizes)/len(neighbourhood_sizes)

        dist_itr = OrderedDict()
        min_dist = 1000
        for itr, dist in enumerate(dist_to_goal_non_cost) :
            min_dist = min(min_dist, dist)
            min_itr = itr
        if min_dist >= max_min_dist:
            continue

        # collect data about the closest distance and neighbour hood in a dictionary
        if task_count not in heuristic_closest_dist_iter_all.keys():
            heuristic_closest_dist_iter_all[task_count] = {}
        if task_count not in heuristic_avg_neighbourhood_size_all.keys():
            heuristic_avg_neighbourhood_size_all[task_count] = {}

        if budget_scaling not in heuristic_closest_dist_iter_all[task_count].keys():
            heuristic_closest_dist_iter_all[task_count][budget_scaling] = []
        if budget_scaling not in heuristic_avg_neighbourhood_size_all[task_count].keys():
            heuristic_avg_neighbourhood_size_all[task_count][budget_scaling] = []


        heuristic_closest_dist_iter_all[task_count][budget_scaling].append(min_itr)
        heuristic_avg_neighbourhood_size_all[task_count][budget_scaling].append(avg_neighbourhood_size)

    # avg collected data
    heuristic_closest_dist_iter_avg = {}
    for task_count in heuristic_closest_dist_iter_all.keys():
        heuristic_closest_dist_iter_avg[task_count] = {}
        for budget_scaling in heuristic_closest_dist_iter_all[task_count].keys():
            heuristic_closest_dist_iter_avg[task_count][budget_scaling] = sum(heuristic_closest_dist_iter_all[task_count][budget_scaling])/len(heuristic_closest_dist_iter_all[task_count][budget_scaling])

    heuristic_avg_neighbourhood_size_avg = {}
    for task_count in heuristic_avg_neighbourhood_size_all.keys():
        heuristic_avg_neighbourhood_size_avg[task_count] = {}
        for budget_scaling in heuristic_avg_neighbourhood_size_all[task_count].keys():
            heuristic_avg_neighbourhood_size_avg[task_count][budget_scaling] = sum(heuristic_avg_neighbourhood_size_all[task_count][budget_scaling])/len(heuristic_avg_neighbourhood_size_all[task_count][budget_scaling])



    return heuristic_closest_dist_iter_avg, heuristic_avg_neighbourhood_size_avg


def find_closest_dist_on_average(input_dir_names, all_results_files):
    max_min_dist  = 30
    # iterate and collect data for farthest distance
    heuristic_closest_dist_iter_all = {}
    results_for_plotting = {}
    for result_file in all_results_files:
        df = pd.concat((pd.read_csv(f) for f in [result_file]))
        ht = df['heuristic_type']
        if list(ht)[0] not in heuristic_closest_dist_iter_all:
            heuristic_closest_dist_iter_all[list(ht)[0]] = []
        dist_to_goal_non_cost = df["ref_des_dist_to_goal_non_cost"]


        dist_itr = OrderedDict()
        min_dist = 1000
        for itr, dist in enumerate(dist_to_goal_non_cost) :
            min_dist = min(min_dist, dist)
        if min_dist >= max_min_dist:
            continue
        heuristic_closest_dist_iter_all[list(ht)[0]].append(min_dist)
        if int(min_dist) == 15 and list(ht)[0] == "SA":
            results_for_plotting["SA"] = result_file
        if int(min_dist) == 4 and list(ht)[0] == "moos":
            results_for_plotting["moos"] = result_file



    # per heuristic reduce
    heuristic_closest_dist_iter_avg = {}
    max_of_closest_dist_reached = {}
    for heuristic, values in heuristic_closest_dist_iter_all.items():
        #max_of_closest_dist_reached = 0
        heuristic_closest_dist_iter_avg[heuristic] = sum(values)/len(values)
        max_of_closest_dist_reached[heuristic] = max(values)

    return heuristic_closest_dist_iter_avg, max_of_closest_dist_reached


def heuristic_scaling(input_dir_names, all_results_files, summary_res_column_name_number):
    heuristic_closest_dist_iter = heuristic_scaling_parsing(input_dir_names, all_results_files)
    print("ok")

def heuristic_comparison(input_dir_names, all_results_files, summary_res_column_name_number):
    intrested_distance_to_consider = [500, 200, 100, 50, 10]
    #intrested_distance_to_consider = [1000, 500, 100, 10, 5, 1, .01]
    heuristic_dist_along_the_way_iter_avg = find_average_iteration_to_distance(input_dir_names, all_results_files, intrested_distance_to_consider)
    heuristic_closest_dist_avg, max_of_closest_dist_reached = find_closest_dist_on_average(input_dir_names, all_results_files)
    #longest_dist = max(list(heuristic_closest_dist_iter_avg.values()))
    heuristic_dist_iter_avg = find_average_iteration_to_distance(input_dir_names, all_results_files, [max_of_closest_dist_reached["SA"]])
    heuristic_phv = pareto_studies(input_dir_names, all_results_files, summary_res_column_name_number)


    # calculate quality gain
    #FARSI_closest_dist = heuristic_closest_dist_avg["FARSI"]
    quality_gain =  {}
    for heu,dist in heuristic_closest_dist_avg.items():
        quality_gain[heu] =  ((heuristic_closest_dist_avg["SA"] - dist)/heuristic_closest_dist_avg["SA"])*100


    # calculate speedup
    speedup = {}
    for heu, vals in heuristic_dist_iter_avg.items():
        for dist, iter in vals.items():
            if not dist == max_of_closest_dist_reached["SA"]:
                continue
            speedup[heu] = heuristic_dist_iter_avg["SA"][dist]/iter


    # calculate speedup
    speedup_along_the_way = {}
    for heu, vals in heuristic_dist_along_the_way_iter_avg.items():
        if heu not in speedup_along_the_way.keys():
            speedup_along_the_way[heu] = {}
        for dist, iter in vals.items():
            speedup_along_the_way[heu][dist] = iter/heuristic_dist_along_the_way_iter_avg["SA"][dist]


    # calculate hpv
    phv_normalized = {}
    for heu in heuristic_phv.keys():
        phv_normalized[heu] = heuristic_phv[heu]/heuristic_phv["SA"]
    return speedup




    print("ok")



def pandas_plots_for_paper(input_dir_names, all_results_files, metric):
    df = pd.concat((pd.read_csv(f) for f in all_results_files))

    #df = raw_df.loc[(raw_df["move validity"] == True)]
    #df["dist_to_goal_non_cost_delta"] = df["ref_des_dist_to_goal_non_cost"] - df["dist_to_goal_non_cost"]
    #df["local_traffic_ratio"] = np.divide(df["local_total_traffic"], df["local_total_traffic"] + df["global_total_traffic"])
    #metric = "global_memory_avg_freq"
    #metric_ylabel = "Global memory avg freq"
    #metric = "local_traffic_ratio"

    # metric_ylabel = metric #"Local traffic ratio" # Ying: replaced the underscores with whitespaces; the new code is the following line
    metric_ylabel = ' '.join(metric.split('_'))
    """
    Ying: add the following lines just in case we need them
    """
    if metric == "ip_cnt":
        metric_ylabel = "IP Count"
    elif metric == "local_bus_cnt":
        metric_ylabel = "NoC Count"
    elif metric == "local_bus_avg_freq":
        metric_ylabel = "NoC Avg Frequency (Hz)"
    elif metric == "local_channel_count_per_bus_coeff_var":
        metric_ylabel = "Channel Variation"
    elif metric == "local_memory_area_coeff_var":
        metric_ylabel = "Memory Area Variation"
    elif metric == "local_bus_freq_coeff_var":
        metric_ylabel = "NoC Frequency Variation"
    elif metric == "local_total_traffic_reuse_no_read_in_bytes_per_cluster_avg":
        metric_ylabel = "Memory Reuse (Bytes)"
    elif metric == "avg_accel_parallelism":
        metric_ylabel = "Avg ALP"
    elif metric == "local_bus_avg_actual_bandwidth":
        metric_ylabel = "Link Bandwidth (Bytes/s)"
    elif metric == "local_memory_total_area":
        metric_ylabel = "Local Memory Area (mm2)"
    elif metric == "local_total_traffic":
        metric_ylabel = "Local Total Traffic (Bytes)"
    """
    Ying: adding finished
    """

    varying_x = [
            "budget_scaling_latency",
            "budget_scaling_power",
            "budget_scaling_area",
    ]
    varying_x_labels = [
            "latency",
            "power",
            "area",
    ]

    axis_font = {'size': "28"}
    plt.figure(figsize=(7, 6.4))
    fig, ax = plt.subplots(1, figsize=(7, 6.4))   # Ying: add the figure size
    grouped_barplot_varying_x_for_paper(
            df,
            metric, metric_ylabel,
            varying_x, varying_x_labels,
            ax
    )
    plt.rc('font', **axis_font)
    plt.tight_layout()
    output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
    output_dir = os.path.join(output_base_dir, "panda_study/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, metric+".png"), bbox_inches='tight')
    #plt.show()
    plt.close('all')



    #fig.tight_layout(rect=[0, 0, 1, 1])
    #fig.savefig("/Users/behzadboro/Project_FARSI_dir/Project_FARSI_with_channels/data_collection/data/simple_run/27_point_coverage_zad/bleh.png")
    #plt.close(fig)


def pareto_studies(input_dir_names,all_result_files, summary_res_column_name_number):
    def points_exceed_one_of_the_budgets(point, base_budget, budget_scaling_to_consider):
        power = point[0]
        area = point[1]
        if power > base_budgets["power"] * budget_scale_to_consider and area > base_budgets[
            "area"] * budget_scale_to_consider:
            return True
        return False

    heuristic_results = {}

    #system_char_to_keep_track_of = {"memory_total_area", "local_memory_total_area","pe_total_area", "ip_cnt", "ips_total_area"}
    system_char_to_keep_track_of = {"system bus count"}

    # budget scaling to consider
    budget_scale_to_consider = .5
    # get budget first
    base_budgets = {}
    for file in all_result_files:
        with open(file, newline='') as csvfile:
            resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for i, row in enumerate(resultReader):
                if i == 1:
                    base_budgets["power"] = float(row[summary_res_column_name_number["power_budget"]])
                    base_budgets["area"] = float(row[summary_res_column_name_number["area_budget"]])
                    break


    for file in all_result_files:
        with open(file, newline='') as csvfile:
            resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for i, row in enumerate(resultReader):
                if i == 0:
                    continue
                heuristic_set_name = row[summary_res_column_name_number["heuristic_type"]]
                if heuristic_set_name not in heuristic_results.keys():
                    heuristic_results[heuristic_set_name] = []
                latencies = row[summary_res_column_name_number["latency"]].split(";")
                latency = {}
                for el in latencies:
                    if el == "":
                        continue
                    workload_name =  el.split("=")[0]
                    latency[workload_name] = float(el.split("=")[1])

                #workload_results[workload_set_name].append((float(power),float(area), float(system_complexity)))

                area= float(row[summary_res_column_name_number["area"]])
                power = float(row[summary_res_column_name_number["power"]])
                system_char = {}
                for el in system_char_to_keep_track_of:
                    system_char[el] = float(row[summary_res_column_name_number[el]])
                point_system_char = {(latency["audio_decoder"], latency["hpvm_cava"], latency["edge_detection"], power, area): system_char}
                heuristic_results[heuristic_set_name].append(point_system_char)

    heuristic_pareto_points = {}
    for heuristic, points_ in heuristic_results.items():
        points = [list(el.keys())[0] for el in points_]
        pareto_points= find_pareto_points(list(set(points)))
        #pareto_points= find_pareto_points(list(set(points)))
        heuristic_pareto_points[heuristic] = pareto_points

    max_0, max_1, max_2, max_3, max_4 = 0, 0,0,0,0
    for heuristic, pareto_points in heuristic_pareto_points.items():
        for el in pareto_points:
            max_0_, max_1_, max_2_, max_3_, max_4_ = el
            max_0 = max([max_0, max_0_])
            max_1 = max([max_1, max_1_])
            max_2 = max([max_2, max_2_])
            max_3 = max([max_3, max_3_])
            max_4 = max([max_4, max_4_])


    heuristic_hv = {}
    ref = [max_0, max_1, max_2, max_3, max_4]
    for heuristic, pareto_points in heuristic_pareto_points.items():
        hv = hypervolume(pareto_points)
        hv_value = hv.compute(ref)
        heuristic_hv[heuristic] = hv_value
    return heuristic_hv

    all_points_in_isolation = []
    all_points_cross_workloads = []

    workload_in_isolation = {}
    for workload, points in workload_results.items():
        #points = [list(el.keys())[0] for el in points_]
        if "cava" in workload and "audio" in workload and "edge_detection" in workload:
            for point in points:
                all_points_cross_workloads.append(point)
        else:
            workload_in_isolation[workload] = points


    ctr = 0
    workload_in_isolation_pareto = {}
    for workload, points_ in workload_in_isolation.items():
        workload_in_isolation_pareto[workload] = []
        points = [list(el.keys())[0] for el in points_]
        pareto_points = find_pareto_points(list(set(points)))
        for point in pareto_points:
            keys = [list(el.keys())[0] for el in workload_in_isolation[workload]]
            idx = keys.index(point)
            workload_in_isolation_pareto[workload].append({point:(workload_in_isolation[workload])[idx]})



    combined_area_power_in_isolation= []
    s = time.time()

    workload_in_isolation_pareto_only_area_power = {}
    for key, val in workload_in_isolation_pareto.items():
        workload_in_isolation_pareto_only_area_power[key] = []
        for el in val:
            for k,v in el.items():
                workload_in_isolation_pareto_only_area_power[key].append(k)


    for results_combined in itertools.product(*list(workload_in_isolation_pareto_only_area_power.values())):
        # add up all the charactersitics
        system_chars = {}
        for el in system_char_to_keep_track_of:
            system_chars[el] = 0

        # add up area,power
        combined_power_area_tuple = [0,0]
        for el in results_combined:
            combined_power_area_tuple[0] += el[0]
            combined_power_area_tuple[1] += el[1]

        for point in results_combined:
            keys = [list(point_.keys())[0] for point_ in workload_in_isolation_pareto[workload]]
            idx = keys.index(point)
            for el in system_char.keys():
                system_char[el] += workload_in_isolation

            system_chars[workload].append({point: (workload_in_isolation[workload])[idx]})


        #combined_area_power_in_isolation.append((combined_power_area_tuple[0],combined_power_area_tuple[1], combined_power_area_tuple[2]))
        combined_area_power_in_isolation.append((combined_power_area_tuple[0],combined_power_area_tuple[1]))

    combined_area_power_in_isolation_filtered = []
    for point in combined_area_power_in_isolation:
        if not points_exceed_one_of_the_budgets(point, base_budgets, budget_scale_to_consider):
            combined_area_power_in_isolation_filtered.append(point)
    combined_area_power_pareto = find_pareto_points(list(set(combined_area_power_in_isolation_filtered)))


    all_points_cross_workloads_filtered = []
    for point in all_points_cross_workloads:
        if not points_exceed_one_of_the_budgets(point, base_budgets, budget_scale_to_consider):
            all_points_cross_workloads_filtered.append(point)
    all_points_cross_workloads_area_power_pareto = find_pareto_points(list(set(all_points_cross_workloads_filtered)))


    # prepare for plotting and plot
    fig = plt.figure(figsize=(12, 12))
    #plt.rc('font', **axis_font)
    ax = fig.add_subplot(111)
    fontSize = 20

    x_values = [el[0] for el in combined_area_power_in_isolation_filtered]
    y_values = [el[1] for el in combined_area_power_in_isolation_filtered]
    x_values.reverse()
    y_values.reverse()
    ax.scatter(x_values, y_values, label="isolated design methodology",marker=".")


    # plt.tight_layout()
    x_values = [el[0] for el in combined_area_power_pareto]
    y_values = [el[1] for el in combined_area_power_pareto]
    x_values.reverse()
    y_values.reverse()
    ax.scatter(x_values, y_values, label="isolated design methodology pareto front",marker="x")


    x_values = [el[0] for el in all_points_cross_workloads_filtered]
    y_values = [el[1] for el in all_points_cross_workloads_filtered]
    x_values.reverse()
    y_values.reverse()
    ax.scatter(x_values, y_values, label="cross workload methodology",marker="8")
    ax.legend(loc="upper right")  # bbox_to_anchor=(1, 1), loc="upper left")

    x_values = [el[0] for el in all_points_cross_workloads_area_power_pareto]
    y_values = [el[1] for el in all_points_cross_workloads_area_power_pareto]
    x_values.reverse()
    y_values.reverse()
    ax.scatter(x_values, y_values, label="cross workload pareto front",marker="o")
    #for idx,_ in  enumeate(x_values):
    #    plt.text(x_values[idx], y_values[idx], s=)

    #plt.text([ for el in x)
    ax.legend(loc="upper right")  # bbox_to_anchor=(1, 1), loc="upper left")


    ax.set_xlabel("power", fontsize=fontSize)
    ax.set_ylabel("area", fontsize=fontSize)
    plt.tight_layout()

    # dump in the top folder
    output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
    output_dir = os.path.join(output_base_dir, "budget_optimality/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, "budget_optimality.png"))
    #plt.show()
    plt.close('all')




def get_budget_optimality_advanced(input_dir_names,all_result_files, summary_res_column_name_number):
    def points_exceed_one_of_the_budgets(point, base_budget, budget_scaling_to_consider):
        power = point[0]
        area = point[1]
        if power > base_budgets["power"] * budget_scale_to_consider and area > base_budgets[
            "area"] * budget_scale_to_consider:
            return True
        return False

    workload_results = {}

    #system_char_to_keep_track_of = {"memory_total_area", "local_memory_total_area","pe_total_area", "ip_cnt", "ips_total_area"}
    system_char_to_keep_track_of = {"ip_cnt"}

    # budget scaling to consider
    budget_scale_to_consider = .5
    # get budget first
    base_budgets = {}
    for file in all_result_files:
        with open(file, newline='') as csvfile:
            resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for i, row in enumerate(resultReader):
                if i == 1:
                    if float(row[summary_res_column_name_number["budget_scaling_latency"]]) == 1 and\
                            float(row[summary_res_column_name_number["budget_scaling_power"]]) == 1 and \
                            float(row[summary_res_column_name_number["budget_scaling_area"]]) == 1:
                        base_budgets["power"] = float(row[summary_res_column_name_number["power_budget"]])
                        base_budgets["area"] = float(row[summary_res_column_name_number["area_budget"]])
                        break


    for file in all_result_files:
        with open(file, newline='') as csvfile:
            resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for i, row in enumerate(resultReader):
                if i == 1:
                    workload_set_name = row[summary_res_column_name_number["workload_set"]]
                    if workload_set_name not in workload_results.keys():
                        workload_results[workload_set_name] = []
                    latency = ((row[summary_res_column_name_number["latency"]].split(";"))[0].split("="))[1]
                    latency_budget = ((row[summary_res_column_name_number["latency_budget"]].split(";"))[0].split("="))[1]
                    if float(latency)  > float(latency_budget):
                        continue

                    #workload_results[workload_set_name].append((float(power),float(area), float(system_complexity)))

                    area= float(row[summary_res_column_name_number["area"]])
                    power = float(row[summary_res_column_name_number["power"]])
                    system_char = {}
                    for el in system_char_to_keep_track_of:
                        system_char[el] = float(row[summary_res_column_name_number[el]])
                    point_system_char = {(power, area): system_char}
                    workload_results[workload_set_name].append(point_system_char)

    workload_pareto_points = {}
    for workload, points_ in workload_results.items():
        points = [list(el.keys())[0] for el in points_]
        pareto_points= find_pareto_points(list(set(points)))
        workload_pareto_points[workload] = []
        for point in pareto_points:
            keys = [list(el.keys())[0] for el in workload_results[workload]]
            idx = keys.index(point)
            workload_pareto_points[workload].append({point:(workload_results[workload])[idx]})


    """" 
    # combine the results
    combined_area_power = []
    for results_combined in itertools.product(*list(workload_pareto_points.values())):
        combined_power_area_tuple = [0,0]
        for el in results_combined:
            combined_power_area_tuple[0] += el[0]
            combined_power_area_tuple[1] += el[1]
        combined_area_power.append(combined_power_area_tuple[:])
    """


    all_points_in_isolation = []
    all_points_cross_workloads = []

    workload_in_isolation = {}
    for workload, points in workload_results.items():
        #points = [list(el.keys())[0] for el in points_]
        if "cava" in workload and "audio" in workload and "edge_detection" in workload:
            for point in points:
                all_points_cross_workloads.append(point)
        else:
            workload_in_isolation[workload] = points


    ctr = 0
    workload_in_isolation_pareto = {}
    for workload, points_ in workload_in_isolation.items():
        workload_in_isolation_pareto[workload] = []
        points = [list(el.keys())[0] for el in points_]
        pareto_points = find_pareto_points(list(set(points)))
        for point in pareto_points:
            keys = [list(el.keys())[0] for el in workload_in_isolation[workload]]
            idx = keys.index(point)
            workload_in_isolation_pareto[workload].append({point:(workload_in_isolation[workload])[idx]})



    combined_area_power_in_isolation= []
    s = time.time()

    workload_in_isolation_pareto_only_area_power = {}
    for key, val in workload_in_isolation_pareto.items():
        workload_in_isolation_pareto_only_area_power[key] = []
        for el in val:
            for k,v in el.items():
                workload_in_isolation_pareto_only_area_power[key].append(k)


    for results_combined in itertools.product(*list(workload_in_isolation_pareto_only_area_power.values())):
        # add up all the charactersitics
        system_chars = {}
        for el in system_char_to_keep_track_of:
            system_chars[el] = 0

        # add up area,power
        combined_power_area_tuple = [0,0]
        for el in results_combined:
            combined_power_area_tuple[0] += el[0]
            combined_power_area_tuple[1] += el[1]

        for point in results_combined:
            keys = [list(point_.keys())[0] for point_ in workload_in_isolation_pareto[workload]]
            idx = keys.index(point)
            for el in system_char.keys():
                system_char[el] += workload_in_isolation

            system_chars[workload].append({point: (workload_in_isolation[workload])[idx]})


        #combined_area_power_in_isolation.append((combined_power_area_tuple[0],combined_power_area_tuple[1], combined_power_area_tuple[2]))
        combined_area_power_in_isolation.append((combined_power_area_tuple[0],combined_power_area_tuple[1]))

    combined_area_power_in_isolation_filtered = []
    for point in combined_area_power_in_isolation:
        if not points_exceed_one_of_the_budgets(point, base_budgets, budget_scale_to_consider):
            combined_area_power_in_isolation_filtered.append(point)
    combined_area_power_pareto = find_pareto_points(list(set(combined_area_power_in_isolation_filtered)))


    all_points_cross_workloads_filtered = []
    for point in all_points_cross_workloads:
        if not points_exceed_one_of_the_budgets(point, base_budgets, budget_scale_to_consider):
            all_points_cross_workloads_filtered.append(point)
    all_points_cross_workloads_area_power_pareto = find_pareto_points(list(set(all_points_cross_workloads_filtered)))


    # prepare for plotting and plot
    fig = plt.figure(figsize=(12, 12))
    #plt.rc('font', **axis_font)
    ax = fig.add_subplot(111)
    fontSize = 20

    x_values = [el[0] for el in combined_area_power_in_isolation_filtered]
    y_values = [el[1] for el in combined_area_power_in_isolation_filtered]
    x_values.reverse()
    y_values.reverse()
    ax.scatter(x_values, y_values, label="isolated design methodology",marker=".")


    # plt.tight_layout()
    x_values = [el[0] for el in combined_area_power_pareto]
    y_values = [el[1] for el in combined_area_power_pareto]
    x_values.reverse()
    y_values.reverse()
    ax.scatter(x_values, y_values, label="isolated design methodology pareto front",marker="x")


    x_values = [el[0] for el in all_points_cross_workloads_filtered]
    y_values = [el[1] for el in all_points_cross_workloads_filtered]
    x_values.reverse()
    y_values.reverse()
    ax.scatter(x_values, y_values, label="cross workload methodology",marker="8")
    ax.legend(loc="upper right")  # bbox_to_anchor=(1, 1), loc="upper left")

    x_values = [el[0] for el in all_points_cross_workloads_area_power_pareto]
    y_values = [el[1] for el in all_points_cross_workloads_area_power_pareto]
    x_values.reverse()
    y_values.reverse()
    ax.scatter(x_values, y_values, label="cross workload pareto front",marker="o")
    #for idx,_ in  enumeate(x_values):
    #    plt.text(x_values[idx], y_values[idx], s=)

    #plt.text([ for el in x)
    ax.legend(loc="upper right")  # bbox_to_anchor=(1, 1), loc="upper left")


    ax.set_xlabel("power", fontsize=fontSize)
    ax.set_ylabel("area", fontsize=fontSize)
    plt.tight_layout()

    # dump in the top folder
    output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
    output_dir = os.path.join(output_base_dir, "budget_optimality/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, "budget_optimality.png"))
    #plt.show()
    plt.close('all')


def get_budget_optimality_error():
    X = ['CAVA', "ED", "Audio"]
    optimal = [10, 20, 20, 40]
    Myopic_budgetting = [20, 30, 25, 30]

    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.2, Ygirls, 0.4, label='Girls')
    plt.bar(X_axis + 0.2, Zboys, 0.4, label='Boys')

    plt.xticks(X_axis, X)
    plt.xlabel("Groups")
    plt.ylabel("Number of Students")
    plt.title("Number of Students in each group")
    plt.legend()
    plt.show()

def get_budget_optimality(input_dir_names,all_result_files, reg_summary_res_column_name_number, a_e_h_summary_res_column_name_number):
    #methodology 1:
    e = {"power":0.00211483379112716, "area":0.00000652737916}
    a = {"power": 0.00133071013064968, "area":0.0000048380066802}
    hc = {"power": 0.000471573339922609, "area":0.0000019979449678}
    e_budget = {"power":0.00190575, "area":0.0000038115}
    a_budget = {"power":0.0015485, "area":0.0000030975}
    hc_budget = {"power":0.001005, "area":0.00000201}


    e_dis_budget = {"power":0, "area":0}
    a_dis_budget = {"power":0, "area":0}
    hc_dis_budget = {"power":0, "area":0}

    for el in e_dis_budget.keys():
        e_dis_budget[el] = (e_budget[el]-e[el])/e[el]

    for el in a_dis_budget.keys():
        a_dis_budget[el] = (a_budget[el]-a[el])/a[el]

    for el in hc_dis_budget.keys():
        hc_dis_budget[el] = (hc_budget[el]-hc[el])/hc[el]





    combined_design_methodology_A = {"power":0, "area":0}
    for el in e.keys() :
        combined_design_methodology_A[el] += e[el]
    for el in a.keys() :
        combined_design_methodology_A[el] += e[el]
    for el in hc.keys() :
        combined_design_methodology_A[el] += e[el]


    def get_equivalent_total(charac):
        if charac == "ips_avg_freq":
            return "ip_cnt"
        if charac == "cluster_pe_cnt_avg":
            return "ip_cnt"
        elif charac == "avg_accel_parallelism":
            return "ip_cnt"
        elif charac in ["local_memory_avg_freq"]:
            return "local_mem_cnt"
        elif charac in ["local_bus_avg_actual_bandwidth", "local_bus_avg_theoretical_bandwidth", "local_bus_avg_bus_width", "avg_freq"]:
            return "local_bus_count"
        else:
            return charac

    def find_sys_char(power,area, results_with_sys_char):
        for vals in results_with_sys_char:
            for power_area , sys_chars in vals.items():
                power_ = power_area[0]
                area_ = power_area[1]
                if power == power_ and area_ == area:
                    return sys_chars

    def points_exceed_one_of_the_budgets(point, base_budget, budget_scaling_to_consider):
        power = point[0]
        area = point[1]
        if power > base_budgets["power"] * budget_scale_to_consider and area > base_budgets[
            "area"] * budget_scale_to_consider:
            return True
        return False


    workload_results = {}
    results_with_sys_char = []

    system_char_to_keep_track_of = {"memory_total_area", "local_memory_total_area","pe_total_area", "ip_cnt","ips_total_area", "ips_avg_freq",  "local_mem_cnt",
                                    "local_bus_avg_actual_bandwidth", "local_bus_avg_theoretical_bandwidth", "local_memory_avg_freq", "local_bus_count", "local_bus_avg_bus_width", "avg_freq", "local_total_traffic",
                                    "global_total_traffic","local_memory_avg_freq", "global_memory_avg_freq", "gpps_total_area", "avg_gpp_parallelism", "avg_accel_parallelism", "channel_cnt",
                                   "local_total_traffic_reuse_with_read_in_bytes",
                                    }
    system_char_to_keep_track_of = {"ip_cnt"
                                    }
    #system_char_to_show = ["local_memory_total_area"]
    #system_char_to_show = ["avg_accel_parallelism"]
    #system_char_to_show = ["avg_gpp_parallelism"]
    #system_char_to_show = ["local_bus_avg_actual_bandwidth"]
    #system_char_to_show = ["avg_freq"]  # really is buses avg freq
    #system_char_to_show = ["local_memory_avg_freq"]  # really is buses avg freq
    #system_char_to_show = ["ips_avg_freq"]
    #system_char_to_show = ["gpps_total_area"]
    #system_char_to_show = ["local_bus_avg_bus_width"]
    #system_char_to_show = ["local_memory_avg_freq"]
    #system_char_to_show = ["ips_total_area"]
    system_char_to_show = ["ip_cnt"]
    #system_char_to_show = ["local_mem_cnt"]
    #system_char_to_show = ["global_memory_avg_freq"]
    #system_char_to_show = ["local_bus_avg_theoretical_bandwidth"]
    #system_char_to_show = ["local_memory_avg_freq"]
    #system_char_to_show = ["local_total_traffic"]
    #system_char_to_show = ["global_total_traffic"]
    #system_char_to_show = ["local_total_traffic_reuse_with_read_in_bytes"]
    #system_char_to_show = ["local_bus_count"]
    #system_char_to_show = ["cluster_pe_cnt_avg"]
    #system_char_to_show = ["ip_cnt"]
    #system_char_to_show = ["channel_cnt"]

    # budget scaling to consider
    budget_scale_to_consider = .5
    # get budget first
    base_budgets = {}
    """
    for file in all_result_files:
        with open(file, newline='') as csvfile:
            if not ("a_e_h"  in file and "lat_1__pow_1__area_1" in file):
                continue
            resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for i, row in enumerate(resultReader):
                if i == 1:
                    print("file"+file)
                    if float(row[reg_summary_res_column_name_number["budget_scaling_latency"]]) == 1 and\
                            float(row[reg_summary_res_column_name_number["budget_scaling_power"]]) == 1 and \
                            float(row[reg_summary_res_column_name_number["budget_scaling_area"]]) == 1:
                        base_budgets["power"] = float(row[summary_res_column_name_number["power_budget"]])
                        base_budgets["area"] = float(row[summary_res_column_name_number["area_budget"]])
                        break
    """
    base_budgets = {"power":.008738,"area":.000017475}

    for file in all_result_files:
        if "a_e_h" in file and "lat_1__pow_1__area_1" in file:
            continue
        if ("a_e_h" in file):
            summary_res_column_name_number = a_e_h_summary_res_column_name_number
        else:
            summary_res_column_name_number = reg_summary_res_column_name_number

        with open(file, newline='') as csvfile:
            resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for i, row in enumerate(resultReader):
                if i == 1:
                    workload_set_name = row[summary_res_column_name_number["workload_set"]]
                    if workload_set_name not in workload_results.keys():
                        workload_results[workload_set_name] = []
                    latency = ((row[summary_res_column_name_number["latency"]].split(";"))[0].split("="))[1]
                    latency_budget = ((row[summary_res_column_name_number["latency_budget"]].split(";"))[0].split("="))[1]
                    if float(latency)  > float(latency_budget):
                        continue

                    power = float(row[summary_res_column_name_number["power"]])
                    area = float(row[summary_res_column_name_number["area"]])

                    system_complexity = row[summary_res_column_name_number["ip_cnt"]] # + row[summary_res_column_name_number["gpp_cnt"]]
                    #workload_results[workload_set_name].append((float(power),float(area), float(system_complexity)))
                    workload_results[workload_set_name].append((power,area))
                    system_char = {}
                    for el in system_char_to_keep_track_of:
                        #if "latency" == el:
                        #    system_char[el] = row[summary_res_column_name_number[el]]
                        #else:
                        system_char[el] = float(row[summary_res_column_name_number[el]])
                        system_char["file"] = file
                    point_system_char = {(power, area): system_char}
                    results_with_sys_char.append(point_system_char)



    workload_pareto_points = {}
    for workload, points in workload_results.items():
        workload_pareto_points[workload] = find_pareto_points(list(set(points)))

    """" 
    # combine the results
    combined_area_power = []
    for results_combined in itertools.product(*list(workload_pareto_points.values())):
        combined_power_area_tuple = [0,0]
        for el in results_combined:
            combined_power_area_tuple[0] += el[0]
            combined_power_area_tuple[1] += el[1]
        combined_area_power.append(combined_power_area_tuple[:])
    """


    all_points_in_isolation = []
    all_points_cross_workloads = []

    workload_in_isolation = {}
    for workload, points in workload_results.items():
        if "cava" in workload and "audio" in workload and "edge_detection" in workload:
            for point in points:
                all_points_cross_workloads.append(point)
        else:
            workload_in_isolation[workload] = points


    ctr = 0
    workload_in_isolation_pareto = {}
    for workload, points in workload_in_isolation.items():
        optimal_points = find_pareto_points(list(set(points)))
        workload_in_isolation_pareto[workload] = optimal_points
        #workload_in_isolation_pareto[workload] = points  # show all points instead


    combined_area_power_in_isolation= []
    combined_area_power_in_isolation_with_sys_char = []

    s = time.time()
    for results_combined in itertools.product(*list(workload_in_isolation_pareto.values())):
        # add up all the charactersitics
        combined_sys_chars = {}

        system_char_to_keep_track_of.add("file")
        for el in system_char_to_keep_track_of:
            if el =="file":
                combined_sys_chars[el] = []
            else:
                combined_sys_chars[el] = (0,0)

        # add up area,power
        combined_power_area_tuple = [0,0]
        for el in results_combined:
            combined_power_area_tuple[0] += el[0]
            combined_power_area_tuple[1] += el[1]

            sys_char = find_sys_char(el[0], el[1], results_with_sys_char)
            for el_,val_ in sys_char.items():
                if el_ == "file":
                    combined_sys_chars[el_].append(val_)
                    continue

                if "avg" in el_:
                    total = sys_char[get_equivalent_total(el_)]
                    coeff = total
                else:
                    coeff = 1
                #if "latency" in el_:
                #    combined_sys_chars[el_] = (combined_sys_chars[el_][0]+coeff, str(combined_sys_chars[el_][1])+"_"+val_)
                #else:
                combined_sys_chars[el_] = (combined_sys_chars[el_][0]+coeff, combined_sys_chars[el_][1]+coeff*float(val_))

        for key, values in combined_sys_chars.items():
            if "avg" in key:
                combined_sys_chars[key] = values[1] /max(values[0],.00000000000000000000000000000001)
            elif "file" in key:
                combined_sys_chars[key] = values
            else:
                combined_sys_chars[key] = values[1]

        if float(combined_sys_chars[system_char_to_show[0]]) == 53675449.0:
            for f in  combined_sys_chars["file"]:
                print("/".join(f.split("/")[:-2])+"./runs/0/system_image.dot")
        #if float(combined_sys_chars[system_char_to_show[0]]) == 53008113:

        #combined_area_power_in_isolation.append((combined_power_area_tuple[0],combined_power_area_tuple[1], combined_power_area_tuple[2]))
        combined_area_power_in_isolation.append((combined_power_area_tuple[0],combined_power_area_tuple[1]))
        combined_area_power_in_isolation_with_sys_char.append({(combined_power_area_tuple[0],combined_power_area_tuple[1]): combined_sys_chars})

        #if len(combined_area_power_in_isolation)%100000 == 0:
        #    print("time passed is" + str(time.time()-s))

    combined_area_power_in_isolation_filtered = []
    for point in combined_area_power_in_isolation:
        if not points_exceed_one_of_the_budgets(point, base_budgets, budget_scale_to_consider):
            combined_area_power_in_isolation_filtered.append(point)
    combined_area_power_pareto = find_pareto_points(list(set(combined_area_power_in_isolation_filtered)))


    all_points_cross_workloads_filtered = []
    for point in all_points_cross_workloads:
        if not points_exceed_one_of_the_budgets(point, base_budgets, budget_scale_to_consider):
            all_points_cross_workloads_filtered.append(point)
    all_points_cross_workloads_area_power_pareto = find_pareto_points(list(set(all_points_cross_workloads_filtered)))


    # prepare for plotting and plot
    fig = plt.figure(figsize=(12, 12))
    #plt.rc('font', **axis_font)
    ax = fig.add_subplot(111)
    fontSize = 20

    x_values = [el[0] for el in combined_area_power_in_isolation_filtered]
    y_values = [el[1] for el in combined_area_power_in_isolation_filtered]
    x_values.reverse()
    y_values.reverse()
    ax.scatter(x_values, y_values, label="isolated design methodology",marker=".")


    # plt.tight_layout()
    x_values = [el[0] for el in combined_area_power_pareto]
    y_values = [el[1] for el in combined_area_power_pareto]
    x_values.reverse()
    y_values.reverse()
    ax.scatter(x_values, y_values, label="isolated design methodology pareto front",marker="x")
    for idx, _ in enumerate(x_values) :
        power= x_values[idx]
        area = y_values[idx]
        sys_char = find_sys_char(power, area, combined_area_power_in_isolation_with_sys_char)

        value_to_show = 0
        value_to_show  = sys_char[system_char_to_show[0]]
        #for el in system_char_to_show:
        #    value_to_show += sys_char[el]

        #if system_char_to_show[0] == "latency":
        #    value_in_scientific_notation = value_to_show
        #else:
        #value_to_show = sys_char["local_total_traffic"]/(sys_char["local_memory_total_area"]*4*10**12)
        value_in_scientific_notation = "{:.10e}".format(value_to_show)
        value_in_scientific_notation = value_to_show
        #if idx ==0:

        #plt.text(power,area, value_in_scientific_notation)


    x_values = [el[0] for el in all_points_cross_workloads_filtered]
    y_values = [el[1] for el in all_points_cross_workloads_filtered]
    x_values.reverse()
    y_values.reverse()
    ax.scatter(x_values, y_values, label="cross workload methodology",marker="8")
    ax.legend(loc="upper right")  # bbox_to_anchor=(1, 1), loc="upper left")
    for idx, _ in enumerate(x_values) :
        power= x_values[idx]
        area = y_values[idx]
        sys_char = find_sys_char(power, area, results_with_sys_char)

        value_to_show = 0
        value_to_show  = sys_char[system_char_to_show[0]]
        #for el in system_char_to_show:
        #    value_to_show += sys_char[el]

        #if system_char_to_show[0] == "latency":
        #    value_in_scientific_notation = value_to_show
        #else:
        #value_to_show = sys_char["local_total_traffic"]/(sys_char["local_memory_total_area"]*4*10**12)
        value_in_scientific_notation = "{:.2e}".format(value_to_show)

        #plt.text(power,area, value_in_scientific_notation)
        #plt.text(power,area, value_in_scientific_notation)



    x_values = [el[0] for el in all_points_cross_workloads_area_power_pareto]
    y_values = [el[1] for el in all_points_cross_workloads_area_power_pareto]
    x_values.reverse()
    y_values.reverse()
    ax.scatter(x_values, y_values, label="cross workload pareto front",marker="o")

    for idx, _ in enumerate(x_values) :
        power= x_values[idx]
        area = y_values[idx]
        sys_char = find_sys_char(power, area, results_with_sys_char)

        value_to_show = sys_char[system_char_to_show[0]]

        #if system_char_to_show[0] == "latency":
        #    value_in_scientific_notation = value_to_show
        #else:
        #value_to_show = sys_char["local_total_traffic"]/sys_char["local_memory_total_area"]
        #value_to_show = sys_char["local_total_traffic"]/(sys_char["local_memory_total_area"]*4*10**12)
        value_in_scientific_notation = "{:.2e}".format(value_to_show)

        #plt.text(power,area, value_in_scientific_notation)


    ax.set_xlabel("power", fontsize=fontSize)
    ax.set_ylabel("area", fontsize=fontSize)
    plt.tight_layout()

    # dump in the top folder
    output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
    output_dir = os.path.join(output_base_dir, "budget_optimality/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    ax.scatter(combined_design_methodology_A["power"], combined_design_methodology_A["area"], label="methodology A",marker="+")
    ax.legend(loc="upper right")  # bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_title(system_char_to_show[0] +" for FARSI vs in isolation")
    #ax.set_title("memory_reuse for FARSI vs in isolation")
    fig.savefig(os.path.join(output_dir, system_char_to_show[0] + "_budget_optimality.png"))


    #plt.show()
    plt.close('all')

def get_budget_optimality_for_paper(input_dir_names,all_result_files, reg_summary_res_column_name_number, a_e_h_summary_res_column_name_number):
    #methodology 1:
    e = {"power":0.00211483379112716, "area":0.00000652737916}
    a = {"power": 0.00133071013064968, "area":0.0000048380066802}
    hc = {"power": 0.000471573339922609, "area":0.0000019979449678}
    e_budget = {"power":0.00190575, "area":0.0000038115}
    a_budget = {"power":0.0015485, "area":0.0000030975}
    hc_budget = {"power":0.001005, "area":0.00000201}


    e_dis_budget = {"power":0, "area":0}
    a_dis_budget = {"power":0, "area":0}
    hc_dis_budget = {"power":0, "area":0}

    for el in e_dis_budget.keys():
        e_dis_budget[el] = (e_budget[el]-e[el])/e[el]

    for el in a_dis_budget.keys():
        a_dis_budget[el] = (a_budget[el]-a[el])/a[el]

    for el in hc_dis_budget.keys():
        hc_dis_budget[el] = (hc_budget[el]-hc[el])/hc[el]





    combined_design_methodology_A = {"power":0, "area":0}
    for el in e.keys() :
        combined_design_methodology_A[el] += e[el]
    for el in a.keys() :
        combined_design_methodology_A[el] += e[el]
    for el in hc.keys() :
        combined_design_methodology_A[el] += e[el]


    def get_equivalent_total(charac):
        if charac == "ips_avg_freq":
            return "ip_cnt"
        if charac == "cluster_pe_cnt_avg":
            return "ip_cnt"
        elif charac == "avg_accel_parallelism":
            return "ip_cnt"
        elif charac in ["local_memory_avg_freq"]:
            return "local_mem_cnt"
        elif charac in ["local_bus_avg_actual_bandwidth", "local_bus_avg_theoretical_bandwidth", "local_bus_avg_bus_width", "avg_freq"]:
            return "local_bus_count"
        else:
            return charac

    def find_sys_char(power,area, results_with_sys_char):
        for vals in results_with_sys_char:
            for power_area , sys_chars in vals.items():
                power_ = power_area[0]
                area_ = power_area[1]
                if power == power_ and area_ == area:
                    return sys_chars

    def points_exceed_one_of_the_budgets(point, base_budget, budget_scaling_to_consider):
        power = point[0]
        area = point[1]
        if power > base_budgets["power"] * budget_scale_to_consider and area > base_budgets[
            "area"] * budget_scale_to_consider:
            return True
        return False


    workload_results = {}
    results_with_sys_char = []

    system_char_to_keep_track_of = {"memory_total_area", "local_memory_total_area","pe_total_area", "ip_cnt","ips_total_area", "ips_avg_freq",  "local_mem_cnt",
                                    "local_bus_avg_actual_bandwidth", "local_bus_avg_theoretical_bandwidth", "local_memory_avg_freq", "local_bus_count", "local_bus_avg_bus_width", "avg_freq", "local_total_traffic",
                                    "global_total_traffic","local_memory_avg_freq", "global_memory_avg_freq", "gpps_total_area", "avg_gpp_parallelism", "avg_accel_parallelism", "channel_cnt",
                                   "local_total_traffic_reuse_with_read_in_bytes",
                                    }
    system_char_to_keep_track_of = {"ip_cnt"
                                    }
    #system_char_to_show = ["local_memory_total_area"]
    #system_char_to_show = ["avg_accel_parallelism"]
    #system_char_to_show = ["avg_gpp_parallelism"]
    #system_char_to_show = ["local_bus_avg_actual_bandwidth"]
    #system_char_to_show = ["avg_freq"]  # really is buses avg freq
    #system_char_to_show = ["local_memory_avg_freq"]  # really is buses avg freq
    #system_char_to_show = ["ips_avg_freq"]
    #system_char_to_show = ["gpps_total_area"]
    #system_char_to_show = ["local_bus_avg_bus_width"]
    #system_char_to_show = ["local_memory_avg_freq"]
    #system_char_to_show = ["ips_total_area"]
    system_char_to_show = ["ip_cnt"]
    #system_char_to_show = ["local_mem_cnt"]
    #system_char_to_show = ["global_memory_avg_freq"]
    #system_char_to_show = ["local_bus_avg_theoretical_bandwidth"]
    #system_char_to_show = ["local_memory_avg_freq"]
    #system_char_to_show = ["local_total_traffic"]
    #system_char_to_show = ["global_total_traffic"]
    #system_char_to_show = ["local_total_traffic_reuse_with_read_in_bytes"]
    #system_char_to_show = ["local_bus_count"]
    #system_char_to_show = ["cluster_pe_cnt_avg"]
    #system_char_to_show = ["ip_cnt"]
    #system_char_to_show = ["channel_cnt"]

    # budget scaling to consider
    budget_scale_to_consider = .5
    # get budget first
    base_budgets = {}
    """
    for file in all_result_files:
        with open(file, newline='') as csvfile:
            if not ("a_e_h"  in file and "lat_1__pow_1__area_1" in file):
                continue
            resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for i, row in enumerate(resultReader):
                if i == 1:
                    print("file"+file)
                    if float(row[reg_summary_res_column_name_number["budget_scaling_latency"]]) == 1 and\
                            float(row[reg_summary_res_column_name_number["budget_scaling_power"]]) == 1 and \
                            float(row[reg_summary_res_column_name_number["budget_scaling_area"]]) == 1:
                        base_budgets["power"] = float(row[summary_res_column_name_number["power_budget"]])
                        base_budgets["area"] = float(row[summary_res_column_name_number["area_budget"]])
                        break
    """
    base_budgets = {"power":.008738,"area":.000017475}

    for file in all_result_files:
        if "a_e_h" in file and "lat_1__pow_1__area_1" in file:
            continue
        if ("a_e_h" in file):
            summary_res_column_name_number = a_e_h_summary_res_column_name_number
        else:
            summary_res_column_name_number = reg_summary_res_column_name_number

        with open(file, newline='') as csvfile:
            resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for i, row in enumerate(resultReader):
                if i == 1:
                    workload_set_name = row[summary_res_column_name_number["workload_set"]]
                    if workload_set_name not in workload_results.keys():
                        workload_results[workload_set_name] = []
                    latency = ((row[summary_res_column_name_number["latency"]].split(";"))[0].split("="))[1]
                    latency_budget = ((row[summary_res_column_name_number["latency_budget"]].split(";"))[0].split("="))[1]
                    if float(latency)  > float(latency_budget):
                        continue

                    power = float(row[summary_res_column_name_number["power"]])
                    area = float(row[summary_res_column_name_number["area"]])

                    system_complexity = row[summary_res_column_name_number["ip_cnt"]] # + row[summary_res_column_name_number["gpp_cnt"]]
                    #workload_results[workload_set_name].append((float(power),float(area), float(system_complexity)))
                    workload_results[workload_set_name].append((power,area))
                    system_char = {}
                    for el in system_char_to_keep_track_of:
                        #if "latency" == el:
                        #    system_char[el] = row[summary_res_column_name_number[el]]
                        #else:
                        system_char[el] = float(row[summary_res_column_name_number[el]])
                        system_char["file"] = file
                    point_system_char = {(power, area): system_char}
                    results_with_sys_char.append(point_system_char)



    workload_pareto_points = {}
    for workload, points in workload_results.items():
        workload_pareto_points[workload] = find_pareto_points(list(set(points)))

    """" 
    # combine the results
    combined_area_power = []
    for results_combined in itertools.product(*list(workload_pareto_points.values())):
        combined_power_area_tuple = [0,0]
        for el in results_combined:
            combined_power_area_tuple[0] += el[0]
            combined_power_area_tuple[1] += el[1]
        combined_area_power.append(combined_power_area_tuple[:])
    """


    all_points_in_isolation = []
    all_points_cross_workloads = []

    workload_in_isolation = {}
    for workload, points in workload_results.items():
        if "cava" in workload and "audio" in workload and "edge_detection" in workload:
            for point in points:
                all_points_cross_workloads.append(point)
        else:
            workload_in_isolation[workload] = points


    ctr = 0
    workload_in_isolation_pareto = {}
    for workload, points in workload_in_isolation.items():
        optimal_points = find_pareto_points(list(set(points)))
        workload_in_isolation_pareto[workload] = optimal_points
        #workload_in_isolation_pareto[workload] = points  # show all points instead


    combined_area_power_in_isolation= []
    combined_area_power_in_isolation_with_sys_char = []

    s = time.time()
    for results_combined in itertools.product(*list(workload_in_isolation_pareto.values())):
        # add up all the charactersitics
        combined_sys_chars = {}

        system_char_to_keep_track_of.add("file")
        for el in system_char_to_keep_track_of:
            if el =="file":
                combined_sys_chars[el] = []
            else:
                combined_sys_chars[el] = (0,0)

        # add up area,power
        combined_power_area_tuple = [0,0]
        for el in results_combined:
            combined_power_area_tuple[0] += el[0]
            combined_power_area_tuple[1] += el[1]

            sys_char = find_sys_char(el[0], el[1], results_with_sys_char)
            for el_,val_ in sys_char.items():
                if el_ == "file":
                    combined_sys_chars[el_].append(val_)
                    continue

                if "avg" in el_:
                    total = sys_char[get_equivalent_total(el_)]
                    coeff = total
                else:
                    coeff = 1
                #if "latency" in el_:
                #    combined_sys_chars[el_] = (combined_sys_chars[el_][0]+coeff, str(combined_sys_chars[el_][1])+"_"+val_)
                #else:
                combined_sys_chars[el_] = (combined_sys_chars[el_][0]+coeff, combined_sys_chars[el_][1]+coeff*float(val_))

        for key, values in combined_sys_chars.items():
            if "avg" in key:
                combined_sys_chars[key] = values[1] /max(values[0],.00000000000000000000000000000001)
            elif "file" in key:
                combined_sys_chars[key] = values
            else:
                combined_sys_chars[key] = values[1]

        if float(combined_sys_chars[system_char_to_show[0]]) == 53675449.0:
            for f in  combined_sys_chars["file"]:
                print("/".join(f.split("/")[:-2])+"./runs/0/system_image.dot")
        #if float(combined_sys_chars[system_char_to_show[0]]) == 53008113:

        #combined_area_power_in_isolation.append((combined_power_area_tuple[0],combined_power_area_tuple[1], combined_power_area_tuple[2]))
        combined_area_power_in_isolation.append((combined_power_area_tuple[0],combined_power_area_tuple[1]))
        combined_area_power_in_isolation_with_sys_char.append({(combined_power_area_tuple[0],combined_power_area_tuple[1]): combined_sys_chars})

        #if len(combined_area_power_in_isolation)%100000 == 0:
        #    print("time passed is" + str(time.time()-s))

    combined_area_power_in_isolation_filtered = []
    for point in combined_area_power_in_isolation:
        if not points_exceed_one_of_the_budgets(point, base_budgets, budget_scale_to_consider):
            combined_area_power_in_isolation_filtered.append(point)
    combined_area_power_pareto = find_pareto_points(list(set(combined_area_power_in_isolation_filtered)))


    all_points_cross_workloads_filtered = []
    for point in all_points_cross_workloads:
        if not points_exceed_one_of_the_budgets(point, base_budgets, budget_scale_to_consider):
            all_points_cross_workloads_filtered.append(point)
    all_points_cross_workloads_area_power_pareto = find_pareto_points(list(set(all_points_cross_workloads_filtered)))


    # prepare for plotting and plot
    fig = plt.figure(figsize=(7, 7))    # Ying: (7.5, 7.5) for the main paper, (5, 5) for the extended abstract
    axis_font = {'size':'30'}
    fontSize = 30
    plt.rc('font', **axis_font)
    ax = fig.add_subplot(111)
    # ax.set_xscale('log')

    x_values = [(el[0]) for el in combined_area_power_in_isolation_filtered]
    y_values = [(el[1]) for el in combined_area_power_in_isolation_filtered]
    x_values.reverse()
    y_values.reverse()
    ax.scatter(x_values, y_values, label="Myopic Opt",marker="+", color='gold', alpha=1, s=700)


    # plt.tight_layout()
    x_values = [(el[0]) for el in combined_area_power_pareto]
    y_values = [(el[1]) for el in combined_area_power_pareto]
    x_values.reverse()
    y_values.reverse()
    ax.scatter(x_values, y_values, label="Myopic Opt Pareto Front",marker="+", color='darkorange', alpha=0.8, s=700)
    # for idx, _ in enumerate(x_values) :
    #     power= x_values[idx]
    #     area = y_values[idx]
    #     sys_char = find_sys_char(power, area, combined_area_power_in_isolation_with_sys_char)
    #
    #     value_to_show = 0
    #     value_to_show  = sys_char[system_char_to_show[0]]
    #     #for el in system_char_to_show:
    #     #    value_to_show += sys_char[el]
    #
    #     #if system_char_to_show[0] == "latency":
    #     #    value_in_scientific_notation = value_to_show
    #     #else:
    #     #value_to_show = sys_char["local_total_traffic"]/(sys_char["local_memory_total_area"]*4*10**12)
    #     value_in_scientific_notation = "{:.10e}".format(value_to_show)
    #     value_in_scientific_notation = value_to_show
    #     #if idx ==0:
    #
    #     #plt.text(power,area, value_in_scientific_notation)


    x_values = [(el[0]) for el in all_points_cross_workloads_filtered]
    y_values = [(el[1]) for el in all_points_cross_workloads_filtered]
    x_values.reverse()
    y_values.reverse()
    ax.scatter(x_values, y_values, label="FARSI",marker="*", color='limegreen', alpha=1, s=700)
    # ax.legend(loc="upper right")  # bbox_to_anchor=(1, 1), loc="upper left")
    # for idx, _ in enumerate(x_values) :
    #     power= x_values[idx]
    #     area = y_values[idx]
    #     sys_char = find_sys_char(power, area, results_with_sys_char)
    #
    #     value_to_show = 0
    #     value_to_show  = sys_char[system_char_to_show[0]]
    #     #for el in system_char_to_show:
    #     #    value_to_show += sys_char[el]
    #
    #     #if system_char_to_show[0] == "latency":
    #     #    value_in_scientific_notation = value_to_show
    #     #else:
    #     #value_to_show = sys_char["local_total_traffic"]/(sys_char["local_memory_total_area"]*4*10**12)
    #     value_in_scientific_notation = "{:.2e}".format(value_to_show)
    #
    #     #plt.text(power,area, value_in_scientific_notation)
    #     #plt.text(power,area, value_in_scientific_notation)



    x_values = [(el[0]) for el in all_points_cross_workloads_area_power_pareto]
    y_values = [(el[1]) for el in all_points_cross_workloads_area_power_pareto]
    x_values.reverse()
    y_values.reverse()
    ax.scatter(x_values, y_values, label="FARSI Pareto Front",marker="*", color='darkgreen', alpha=0.8, s=700)

    # for idx, _ in enumerate(x_values) :
    #     # power= x_values[idx]
    #     # area = y_values[idx]
    #     # sys_char = find_sys_char(power, area, results_with_sys_char)
    #     #
    #     # value_to_show = sys_char[system_char_to_show[0]]
    #     #
    #     # #if system_char_to_show[0] == "latency":
    #     # #    value_in_scientific_notation = value_to_show
    #     # #else:
    #     # #value_to_show = sys_char["local_total_traffic"]/sys_char["local_memory_total_area"]
    #     # #value_to_show = sys_char["local_total_traffic"]/(sys_char["local_memory_total_area"]*4*10**12)
    #     # value_in_scientific_notation = "{:.2e}".format(value_to_show)
    #     #
    #     # #plt.text(power,area, value_in_scientific_notation)

    # ax.set_xticks(np.arange(0.002, 0.009, 0.002))

    ax.set_xlabel("Power (W)", fontsize=fontSize)
    ax.set_ylabel("Area (mm2)", fontsize=fontSize)


    # dump in the top folder
    output_base_dir = '/'.join(input_dir_names[0].split("/")[:-2])
    output_dir = os.path.join(output_base_dir, "budget_optimality/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    ax.scatter(combined_design_methodology_A["power"], combined_design_methodology_A["area"], label="Myopic Budgetting",marker="x", color='red', s=700)   # Ying: comment out for the extended abstract
    ax.legend(loc="upper left", fontsize=fontSize, bbox_to_anchor=(1, 1.1), borderpad=1) # , borderpad=0)  # bbox_to_anchor=(1, 1), loc="upper left")
    # ax.set_title(system_char_to_show[0] +" for FARSI vs in isolation")
    #ax.set_title("memory_reuse for FARSI vs in isolation")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, system_char_to_show[0] + "_budget_optimality.png"), bbox_inches='tight')


    #plt.show()
    plt.close('all')

def find_pareto_points(points):
    efficients = is_pareto_efficient_dumb(np.array(points))
    pareto_points_array = [points[idx] for idx, el in enumerate(efficients) if el]

    return pareto_points_array

    pareto_points = []
    for el in pareto_points_array:
        list_ = []
        for el_ in el:
            list.append(el)
        pareto_points.append(list_)

    return pareto_points


def is_pareto_efficient_dumb(costs):
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
    return is_efficient



###########################################

# the main function. comment out the plots if you do not need them
if __name__ == "__main__":
    # populate parameters
    run_folder_name =  config_plotting.run_folder_name
    if  config_plotting.run_folder_name == "":
        run_folder_name = find_the_most_recent_directory(config_plotting.top_result_folder)[0]

    zoneNum = config_plotting.zoneNum
    # get all the experiments under the run folder
    print(run_folder_name)
    experiment_full_addr_list = get_experiment_dir_list(run_folder_name)

    # according to the plot type, plot
    all_res_column_name_number = get_column_name_number(experiment_full_addr_list[0], "all")
    all_results_files = get_experiment_full_file_addr_list(experiment_full_addr_list)
    aggregate_res_column_name_number = get_column_name_number(experiment_full_addr_list[0], "aggregate")

    summary_res_column_name_number = get_column_name_number(experiment_full_addr_list[0], "simple")
    case_studies = {}
    case_studies["bandwidth_analysis"] = ["local_bus_avg_theoretical_bandwidth",
                                          "local_bus_max_actual_bandwidth",
                                          "local_bus_avg_actual_bandwidth",
                                          "system_bus_avg_theoretical_bandwidth",
                                          "system_bus_max_actual_bandwidth",
                                          "system_bus_avg_actual_bandwidth",
                                          "local_channel_avg_actual_bandwidth",
                                          "local_channel_max_actual_bandwidth"
                                          ]


    case_studies["freq_analysis"] = [
        "global_memory_avg_freq", "local_memory_avg_freq", "local_bus_avg_freq",]

    case_studies["bus_width_analysis"] = [
        "global_memory_avg_bus_width","local_memory_avg_bus_width","local_bus_avg_bus_width"]

    case_studies["traffic_analysis"] = ["global_total_traffic", "local_total_traffic",
                                        "local_memory_traffic_per_mem_avg",
                                        "locality_in_bytes",
                                        "local_memory_traffic_per_mem_avg",
                                        "local_bus_traffic_avg",
                                        ]


    case_studies["local_mem_re_use"] =[
        "local_total_traffic_reuse_no_read_ratio",
        "local_total_traffic_reuse_no_read_in_bytes",
        "local_total_traffic_reuse_no_read_in_size",
        "local_total_traffic_reuse_with_read_ratio",
        "local_total_traffic_reuse_with_read_in_bytes",
        "local_total_traffic_reuse_with_read_in_size",
        "local_total_traffic_reuse_no_read_in_bytes_per_cluster_avg",
        "local_total_traffic_reuse_no_read_in_size_per_cluster_avg",
        "local_total_traffic_reuse_with_read_in_bytes_per_cluster_avg",
        "local_total_traffic_reuse_with_read_in_size_per_cluster_avg"
    ]

    case_studies["global_mem_re_use"] =[
        "global_total_traffic_reuse_no_read_ratio",
        "global_total_traffic_reuse_with_read_ratio",
        "global_total_traffic_reuse_with_read_in_bytes",
        "global_total_traffic_reuse_with_read_in_size",
        "global_total_traffic_reuse_no_read_in_bytes",
        "global_total_traffic_reuse_no_read_in_size",
    ]


    case_studies["area_analysis"] = ["global_memory_total_area", "local_memory_total_area", "ips_total_area",
                                     "gpps_total_area",
                                     ]
    case_studies["area_in_bytes_analysis"] = ["global_memory_total_bytes", "local_memory_total_bytes", "local_memory_bytes_avg"
                                     ]

    case_studies["accel_paral_analysis"] = ["ip_cnt","max_accel_parallelism", "avg_accel_parallelism",
                                            "gpp_cnt", "max_gpp_parallelism", "avg_gpp_parallelism"]
    case_studies["system_complexity"] = ["system block count", "routing complexity", "system PE count",
                                         "local_mem_cnt", "local_bus_cnt","local_channel_count_per_bus_avg", "channel_cnt",
                                         "loop_itr_ratio_avg",
                                         ]  # , "channel_cnt"]

    case_studies["heterogeneity_var_system_compleixty"] = [
        "local_channel_count_per_bus_coeff_var",
        "loop_itr_ratio_var",
        # "cluster_pe_cnt_coeff_var"
    ]
    
    case_studies["heterogeneity_std_system_compleixty"] = [
        "local_channel_count_per_bus_std",
        "loop_itr_ratio_std"    # , "cluster_pe_cnt_std"
    ]

    """
    case_studies["speedup"] = [
        # "customization_speed_up_full_system",
        # "loop_unrolling_parallelism_speed_up_full_system",
        "customization_speed_up_full_system",
        "task_level_parallelism_speed_up_full_system",
        "interference_degradation_avg"]
    """
    """
      [ 
        "customization_first_speed_up_avg",
        "customization_second_speed_up_avg",
        "parallelism_first_speed_up_avg",
        "parallelism_second_speed_up_avg",
        "interference_degradation_avg",
        "customization_first_speed_up_full_system",
        "customization_second_speed_up_full_system",
        "parallelism_first_speed_up_full_system",
        "parallelism_second_speed_up_full_system",
    ]
    case_studies["speedup"] = [
        "interference_degradation_avg",
        "customization_speed_up_full_system",
        "parallelism_speed_up_full_system",
        "parallelism_nd_speed_up_full_system",
    ]


    """


    case_studies["heterogenity_area"] = [
                                     "local_memory_area_coeff_var",
                                     "ips_area_coeff_var",
                                     "pes_area_coeff_var",

    ]


    case_studies["heterogenity_std_re_use"] = [
        "local_total_traffic_reuse_no_read_in_bytes_per_cluster_std",
        "local_total_traffic_reuse_no_read_in_size_per_cluster_std",
        "local_total_traffic_reuse_with_read_in_bytes_per_cluster_std",
        "local_total_traffic_reuse_with_read_in_size_per_cluster_std",
    ]

    case_studies["heterogenity_var_re_use"] = [
    "local_total_traffic_reuse_no_read_in_bytes_per_cluster_var",
    "local_total_traffic_reuse_no_read_in_size_per_cluster_var",
    "local_total_traffic_reuse_with_read_in_bytes_per_cluster_var",
    "local_total_traffic_reuse_with_read_in_size_per_cluster_var",
    ]

    case_studies["heterogenity_var_freq"] =[
        "local_bus_freq_coeff_var",
        "local_memory_freq_coeff_var",
        "ips_freq_coeff_var",
        "pes_freq_coeff_var"]

    case_studies["heterogenity_std_freq"] =[
        "local_memory_freq_std",
        "local_bus_freq_std",
    ]



    case_studies["heterogenity_std_bus_width"] =[
        "local_memory_bus_width_std",
        "local_bus_bus_width_std",
    ]

    case_studies["heterogenity_var_bus_width"] =[
        "local_memory_bus_width_coeff_var",
        "local_bus_bus_width_coeff_var",
    ]




    case_studies["heterogenity_std_bandwidth"]=[
    "local_bus_actual_bandwidth_std",
    "local_channel_actual_bandwidth_std"]

    case_studies["heterogenity_var_bandwidth"]=[
    "local_bus_actual_bandwidth_coeff_var",
    "local_channel_actual_bandwidth_coeff_var"]



    case_studies["heterogenity_std_traffic"] =[
                                    "local_memory_bytes_std",
                                    "local_memory_traffic_per_mem_coeff_var",
                                    "local_bus_traffic_coeff_var",
                                    ]


    case_studies["heterogenity_var_traffic"] =[
                                    "local_memory_bytes_coeff_var",
                                    "local_memory_traffic_per_mem_coeff_var",
                                    "local_bus_traffic_coeff_var",
                                    ]

    if "heuristic_comparison" in config_plotting.plot_list:  # Ying: optimal_budgetting_problem_08_1
        #experiment_full_addr_list = get_experiment_dir_list(config_plotting.heuristic_comparison_folder)
        #all_dirs = [x[0] for x in os.walk(config_plotting.heuristic_comparison_folder)]

        metrics = ["heuristic_type", "ref_des_dist_to_goal_non_cost"]
        all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(config_plotting.heuristic_comparison_folder) for f in filenames if
                  os.path.splitext(f)[1] == '.csv']
        aggregate_results = [f for f in all_files if  "aggregate_all_results" in f and not ("prev_iter" in f)]

        heuristic_comparison(experiment_full_addr_list, aggregate_results, aggregate_res_column_name_number)

    if "heuristic_scaling" in config_plotting.plot_list:  # Ying: optimal_budgetting_problem_08_1
        metrics = ["heuristic_type", "ref_des_dist_to_goal_non_cost"]
        all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(config_plotting.heuristic_comparison_folder) for f in filenames if
                  os.path.splitext(f)[1] == '.csv']
        aggregate_results = [f for f in all_files if  "aggregate_all_results" in f and not ("prev_iter" in f)]

        heuristic_scaling(experiment_full_addr_list, aggregate_results, aggregate_res_column_name_number)

    """
    if "pareto_studies" in config_plotting.plot_list:
        metrics = ["heuristic_type", "ref_des_dist_to_goal_non_cost"]
        all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(config_plotting.heuristic_comparison_folder)
                     for f in filenames if
                     os.path.splitext(f)[1] == '.csv']
        aggregate_results = [f for f in all_files if "aggregate_all_results" in f and not ("prev_iter" in f)]

        heuristic_hv = pareto_studies(experiment_full_addr_list, aggregate_results, aggregate_res_column_name_number)
    """


    if "budget_optimality" in config_plotting.plot_list:    # Ying: optimal_budgetting_problem_08_1
        #get_budget_optimality_advanced(experiment_full_addr_list, all_results_files, summary_res_column_name_number)

        for el in  experiment_full_addr_list:
            if "a_e_h" in el and not "lat_1__pow_1__area_1" in el:
                a_e_h_summary_res_column_name_number = get_column_name_number(el, "simple")

        if config_plotting.draw_for_paper:
            get_budget_optimality_for_paper(experiment_full_addr_list, all_results_files, summary_res_column_name_number, a_e_h_summary_res_column_name_number)
        else:
            get_budget_optimality(experiment_full_addr_list, all_results_files, summary_res_column_name_number, a_e_h_summary_res_column_name_number)

    if "cross_workloads" in config_plotting.plot_list:  # Ying: from for_paper/workload_awareness (PE, Mem, IC, TLP, comm_comp); or blind_study_smart_krnel_selection/blind_vs_arch_ware; or blind_study_dumb_kernel_selection/blind_vs_arch_aware; or blind_study_all_dumb_versions/blind_vs_arch_aware (SA and other aware)
        # get column orders (assumption is that the column order doesn't change between experiments)
        if config_plotting.draw_for_paper:
            column_column_value_experiment_frequency_dict = plot_codesign_nav_breakdown_cross_workload_for_paper(
            experiment_full_addr_list, all_res_column_name_number)
            plot_convergence_cross_workloads_for_paper(experiment_full_addr_list, all_res_column_name_number)

        else:
            column_column_value_experiment_frequency_dict = plot_codesign_nav_breakdown_cross_workload(experiment_full_addr_list, all_res_column_name_number)
            plot_convergence_cross_workloads(experiment_full_addr_list, all_res_column_name_number)

        for key, val in case_studies.items():
            case_study = {key:val}
            # plot_system_implication_analysis(experiment_full_addr_list, summary_res_column_name_number, case_study)
        plot_co_design_nav_breakdown_post_processing(experiment_full_addr_list, column_column_value_experiment_frequency_dict)
        if config_plotting.draw_for_paper:
            plot_codesign_rate_efficacy_cross_workloads_updated_for_paper(experiment_full_addr_list, all_res_column_name_number)
        else:
            #plot_codesign_rate_efficacy_cross_workloads_updated(experiment_full_addr_list, all_res_column_name_number)
            pass

    if "single_workload" in config_plotting.plot_list:  # Ying: blind_study_all_dumb_versions/blind_vs_arch_aware
        #single workload
        # plot_codesign_progression_per_workloads(experiment_full_addr_list, all_res_column_name_number)
        _ = plot_codesign_nav_breakdown_per_workload(experiment_full_addr_list, all_res_column_name_number)

        if config_plotting.draw_for_paper:
            # plot_convergence_per_workloads_for_paper(experiment_full_addr_list, all_res_column_name_number)
            plot_convergence_vs_time_for_paper(experiment_full_addr_list, all_res_column_name_number)
        else:
            plot_convergence_per_workloads(experiment_full_addr_list, all_res_column_name_number)
            plot_convergence_vs_time(experiment_full_addr_list, all_res_column_name_number)

    if "plot_3d" in config_plotting.plot_list:
        plot_3d(experiment_full_addr_list, summary_res_column_name_number)

    if "pie_chart" in config_plotting.plot_list:       # Ying: 1_1_1_for_paper_07-31
        pie_chart_case_study = {"Performance Breakdown": ["transformation generation time", "simulation time",
                            "neighbour selection time"],
                                "Transformation_Generation_Breakdown": ["metric selection time", "dir selection time", "kernel selection time",
                                                                        "block selection time",  "transformation selection time",
                                                                "design duplication time", "metric selection time"]}
                                # , "architectural principle", "high level optimization name", "exact optimization name"]

        for case_study_ in pie_chart_case_study.items():
            if config_plotting.draw_for_paper:
                pie_chart_for_paper(experiment_full_addr_list, all_res_column_name_number, case_study_)
            else:
                pie_chart(experiment_full_addr_list, all_res_column_name_number, case_study_)

    if "pandas_plots" in config_plotting.plot_list: # Ying: from scaling_of_1_2_4_across_all_budgets_07-31
        #pandas_case_studies = {}
        case_studies["system_complexity"] = ["system block count", "routing complexity", "system PE count",
                                             "local_mem_cnt", "local_bus_cnt" , "channel_cnt", "ip_cnt", "gpp_cnt"]

        case_studies["pe_parallelism"] = ["max_accel_parallelism", "avg_accel_parallelism", "avg_gpp_parallelism", "max_gpp_parallelism"]

        case_studies["ip_frequency"] = ["ips_avg_freq", "gpps_avg_freq", "ips_freq_std", "pes_freq_std",
                                            "ips_freq_coeff_var", "pes_freq_coeff_var"]

        case_studies["pe_area"] = ["ips_total_area", "gpps_total_area", "ips_area_std", "pes_area_std",
                                            "ips_area_coeff_var", "pes_area_coeff_var"]

        case_studies["mem_frequency"] = ["local_memory_avg_freq", "global_memory_avg_freq",
                                                "local_memory_freq_std","local_memory_freq_coeff_var"]

        case_studies["mem_area"] = ["local_memory_total_area", "global_memory_total_area", "local_memory_area_std",
                                           "local_memory_area_coeff_var"]

        case_studies["traffic"] = ["local_total_traffic", "global_total_traffic"]


        case_studies["bus_width"] = ["local_bus_avg_bus_width",
                                            "system_bus_avg_bus_width"]


        case_studies["bus_bandwidth"] = ["local_bus_avg_actual_bandwidth",  "system_bus_avg_actual_bandwidth",
                                                "local_bus_avg_theoretical_bandwidth", "system_bus_avg_theoretical_bandwidth",
                                                "local_bus_max_actual_bandwidth", "system_bus_max_actual_bandwidth"]


        for case_study_name, metrics in case_studies.items():
            for metric in metrics:
                if config_plotting.draw_for_paper:
                    pandas_plots_for_paper(experiment_full_addr_list, all_results_files, metric)
                else:
                    pandas_plots(experiment_full_addr_list, all_results_files, metric)

    # get the the workload_set folder
    # each workload_set has a bunch of experiments underneath it
    workload_set_folder_list = os.listdir(run_folder_name)

    if "drivers" in config_plotting.plot_list:
        simple_stack_bar_plot(run_folder_name)

    #  iterate and generate plots
    for workload_set_folder in workload_set_folder_list:
        # ignore irelevant files
        if workload_set_folder in config_plotting.ignore_file_names:
            continue

        # start plotting
        #plotBudgets3d(run_folder_name, workload_set_folder)


        """
        # get experiment folder
        workload_set_full_addr = os.path.join(run_folder_name,workload_set_folder)
        folder_list = os.listdir(workload_set_full_addr)
        for experiment_name_relative_addr in folder_list:
            print(experiment_name_relative_addr)
            if experiment_name_relative_addr in config_plotting.ignore_file_names:
                continue
            experiment_full_addr = os.path.join(workload_set_full_addr, experiment_name_relative_addr)

            all_res_column_name_number = get_column_name_number(experiment_full_addr, "all")
            summary_res_column_name_number = get_column_name_number(experiment_full_addr, "simple")

            workload_set_full_addr +="/" # this is because you didn't use join
            commcompColNum = columnNum(workload_set_full_addr, experiment_name_relative_addr, "comm_comp", "all")
            trueNum = columnNum(workload_set_full_addr, experiment_name_relative_addr, "move validity", "all")
            optColNum = columnNum(workload_set_full_addr, experiment_name_relative_addr, "high level optimization name", "all")
            archColNum = columnNum(workload_set_full_addr, experiment_name_relative_addr, "architectural principle", "all")
            sysBlkNum = columnNum(workload_set_full_addr, experiment_name_relative_addr, "system block count", "all")
            simColNum = columnNum(workload_set_full_addr, experiment_name_relative_addr, "simulation time", "all")
            movGenColNum = columnNum(workload_set_full_addr, experiment_name_relative_addr, "transformation generation time", "all")
            movColNum = columnNum(workload_set_full_addr, experiment_name_relative_addr, "move name", "all")
            itrNum = columnNum(workload_set_full_addr, experiment_name_relative_addr, "iteration cnt", "all")
            distColNum = columnNum(workload_set_full_addr, experiment_name_relative_addr, "dist_to_goal_non_cost", "all")
            refDistColNum = columnNum(workload_set_full_addr, experiment_name_relative_addr, "ref_des_dist_to_goal_non_cost", "all")
            latNum = columnNum(workload_set_full_addr, experiment_name_relative_addr, "latency", "all")
            powNum = columnNum(workload_set_full_addr, experiment_name_relative_addr, "power", "all")
            areaNum = columnNum(workload_set_full_addr, experiment_name_relative_addr, "area", "all")

            # comment or uncomment the following functions for your plottings
            plotDistToGoalVSitr([experiment_full_addr], all_res_column_name_number)
            plotCommCompAll(workload_set_full_addr, experiment_name_relative_addr, all_res_column_name_number)
            plothighLevelOptAll(workload_set_full_addr, experiment_name_relative_addr, all_res_column_name_number)
            plotArchVarImpAll(workload_set_full_addr, experiment_name_relative_addr, archColNum, trueNum)
            plotSimTimeVSblk(workload_set_full_addr, experiment_name_relative_addr, sysBlkNum, simColNum, trueNum)
            plotMoveGenTimeVSblk(workload_set_full_addr, experiment_name_relative_addr, sysBlkNum, movGenColNum, trueNum)
            plotRefDistToGoalVSitr(workload_set_full_addr, experiment_name_relative_addr, itrNum, refDistColNum, trueNum)
            plotSimTimeVSmoveNameZoneDist(workload_set_full_addr, experiment_name_relative_addr, zoneNum, movColNum, distColNum, simColNum, trueNum)
            plotMovGenTimeVSmoveNameZoneDist(workload_set_full_addr, experiment_name_relative_addr, zoneNum, movColNum, distColNum, movGenColNum, trueNum)
            plotSimTimeVScommCompZoneDist(workload_set_full_addr, experiment_name_relative_addr, zoneNum, commcompColNum, distColNum, simColNum, trueNum)
            plotMovGenTimeVScommCompZoneDist(workload_set_full_addr, experiment_name_relative_addr, zoneNum, commcompColNum, distColNum, movGenColNum, trueNum)
            plotSimTimeVShighLevelOptZoneDist(workload_set_full_addr, experiment_name_relative_addr, zoneNum, optColNum, distColNum, simColNum, trueNum)
            plotMovGenTimeVShighLevelOptZoneDist(workload_set_full_addr, experiment_name_relative_addr, zoneNum, optColNum, distColNum, movGenColNum, trueNum)
            plotSimTimeVSarchVarImpZoneDist(workload_set_full_addr, experiment_name_relative_addr, zoneNum, archColNum, distColNum, simColNum, trueNum)
            plotMovGenTimeVSarchVarImpZoneDist(workload_set_full_addr, experiment_name_relative_addr, zoneNum, archColNum, distColNum, movGenColNum, trueNum)
        """
