import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("..")
#from plot_validations import *
from sklearn.linear_model import LinearRegression
from settings import config_plotting
import os
def abline(slope, intercept, color):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', color = color)

def get_df_as_avg_for_each_x_coord(reformatted_df, x_coord_name, y_coord_name = "Simulation Time"):
   avg_df_lst = []
   for x_coord in set(reformatted_df[x_coord_name]):
      #print("hola")
      #print(reformatted_df.loc[(reformatted_df[x_coord_name] == x_coord) & (reformatted_df["FARSI or PA"] == "FARSI")])
      simtimes_farsi = list(reformatted_df.loc[(reformatted_df[x_coord_name] == x_coord) & (reformatted_df["FARSI or PA"] == "FARSI")][y_coord_name])
      simtimes_pa = list(reformatted_df.loc[(reformatted_df[x_coord_name] == x_coord) & (reformatted_df["FARSI or PA"] == "PA")][y_coord_name])
      print("simtimes_farsi")
      print(simtimes_farsi)
      print(np.average(simtimes_farsi))
      print("simtimes_pa")
      print(simtimes_pa)
      print(np.average(simtimes_pa))
      avg_df_lst.append([np.average(simtimes_farsi), "FARSI", x_coord])
      avg_df_lst.append([np.average(simtimes_pa), "PA", x_coord])
   return pd.DataFrame(avg_df_lst, columns = ["Simulation Time", "FARSI or PA", x_coord_name])

#not used yet in this script
def get_df_as_avg_for_each_x_coord(reformatted_df, x_coord_name, y_coord_name = "Simulation Time", hue_col = "FARSI or PA"):
   hues = set(list(reformatted_df[hue_col]))
   avg_df_lst = []
   for x_coord in set(reformatted_df[x_coord_name]):
      #print("hola")
      #print(reformatted_df.loc[(reformatted_df[x_coord_name] == x_coord) & (reformatted_df["FARSI or PA"] == "FARSI")])
      for hue in hues:
         selectedy_hue = list(reformatted_df.loc[(reformatted_df[x_coord_name] == x_coord) & (reformatted_df[hue_col] == hue)][y_coord_name])
         avg_df_lst.append([np.average(selectedy_hue), hue, x_coord])
      #simtimes_pa = list(reformatted_df.loc[(reformatted_df[x_coord_name] == x_coord) & (reformatted_df["FARSI or PA"] == "PA")][y_coord_name])
   return pd.DataFrame(avg_df_lst, columns = [y_coord_name, hue_col, x_coord_name])



def plot_sim_time_vs_system_char_minimal(output_dir, csv_file_addr):
    data = pd.read_csv(csv_file_addr)
    blk_cnt = list(data["blk_cnt"])
    pa_sim_time = list(data["PA simulation time"])
    farsi_sim_time = list(data["FARSI simulation time"])
    tmp_reformatted_df_data = [blk_cnt * 2,  pa_sim_time + farsi_sim_time,
                               ["PA"] * len(blk_cnt) + ["FARSI"] * len(blk_cnt)]
    reformatted_df_data = [[tmp_reformatted_df_data[j][i] for j in range(len(tmp_reformatted_df_data))] for i in
                           range(len(blk_cnt) * 2)]
    # print(reformatted_df_data[0:3])
    # exit()
    # for col in reformatted_df_data:
    #    print("Len of col is {}".format(len(col)))
    reformatted_df = pd.DataFrame(reformatted_df_data,
                                  columns=["Block counts", "Simulation Time",
                                           "FARSI or PA"])
    print(reformatted_df.head())

    df_blk_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "Block counts")



    df_avg = get_df_as_avg_for_each_x_coord(reformatted_df, x_coord_name = "Block counts", y_coord_name = "Simulation Time", hue_col = "FARSI or PA")

    #df_pe_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "PE counts")
    #df_mem_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "Mem counts")
    #df_bus_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "Bus counts")

    #print("Bola")
    #print(df_blk_avg)



    splot = sns.scatterplot(data=df_avg, x="Block counts", y="Simulation Time", hue="FARSI or PA")
    splot.set(yscale="log")

    color_per_hue = {"FARSI" : "green", "PA" : "orange"}
    hues = set(list(df_avg["FARSI or PA"]))
    for hue in hues:
       #x required to be in matrix format in sklearn
       print(np.isnan(df_avg["Simulation Time"]))
       xs_hue = [[x] for x in list(df_avg.loc[(df_avg["FARSI or PA"] == hue) & (df_avg["Simulation Time"].notnull())]["Block counts"])]
       ys_hue = np.array(list(df_avg.loc[(df_avg["FARSI or PA"] == hue) & (df_avg["Simulation Time"].notnull())]["Simulation Time"]))
       print("xs_hue")
       print(xs_hue)

       print("ys_hue")
       print(ys_hue)
       reg = LinearRegression().fit(xs_hue, ys_hue)
       m = reg.coef_[0]
       n = reg.intercept_
       abline(m, n, color_per_hue[hue])
    #plt.set_ylim(top = 10)


    plt.savefig(os.path.join(output_dir,'block_counts_vs_simtime.png'))

    plt.close("all")

def plot_sim_time_vs_system_char_minimal_for_paper(output_dir, csv_file_addr):
    data = pd.read_csv(csv_file_addr)
    blk_cnt = list(data["blk_cnt"])
    pa_sim_time = list(data["PA simulation time"])
    farsi_sim_time = list(data["FARSI simulation time"])
    tmp_reformatted_df_data = [blk_cnt * 2,  pa_sim_time + farsi_sim_time,
                               ["PA"] * len(blk_cnt) + ["FARSI"] * len(blk_cnt)]
    reformatted_df_data = [[tmp_reformatted_df_data[j][i] for j in range(len(tmp_reformatted_df_data))] for i in
                           range(len(blk_cnt) * 2)]
    # print(reformatted_df_data[0:3])
    # exit()
    # for col in reformatted_df_data:
    #    print("Len of col is {}".format(len(col)))
    reformatted_df = pd.DataFrame(reformatted_df_data,
                                  columns=["Block Counts", "Simulation Time",
                                           "FARSI or PA"])
    print(reformatted_df.head())

    df_blk_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "Block Counts")



    df_avg = get_df_as_avg_for_each_x_coord(reformatted_df, x_coord_name = "Block Counts", y_coord_name = "Simulation Time", hue_col = "FARSI or PA")

    #df_pe_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "PE counts")
    #df_mem_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "Mem counts")
    #df_bus_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "Bus counts")

    #print("Bola")
    #print(df_blk_avg)

    axis_font = {'size': '20'}
    fontSize = 20
    sns.set(font_scale=2, rc={'figure.figsize': (6, 4)})
    sns.set_style("white")
    color_per_hue = {'PA': 'hotpink', 'FARSI': 'green'}
    splot = sns.scatterplot(data=df_avg, x="Block Counts", y="Simulation Time", hue="FARSI or PA", sizes=(6, 6), palette=color_per_hue)
    splot.set(yscale="log")
    splot.legend(title="", fontsize=fontSize, loc="center right")

    hues = set(list(df_avg["FARSI or PA"]))
    for hue in hues:
       #x required to be in matrix format in sklearn
       print(np.isnan(df_avg["Simulation Time"]))
       xs_hue = [[x] for x in list(df_avg.loc[(df_avg["FARSI or PA"] == hue) & (df_avg["Simulation Time"].notnull())]["Block Counts"])]
       ys_hue = np.array(list(df_avg.loc[(df_avg["FARSI or PA"] == hue) & (df_avg["Simulation Time"].notnull())]["Simulation Time"]))
       print("xs_hue")
       print(xs_hue)

       print("ys_hue")
       print(ys_hue)
       reg = LinearRegression().fit(xs_hue, ys_hue)
       m = reg.coef_[0]
       n = reg.intercept_
       abline(m, n, color_per_hue[hue])
    #plt.set_ylim(top = 10)

    plt.xticks(np.arange(0, 30, 10.0))
    plt.yticks(np.power(10.0, [-1, 0, 1, 2, 3]))
    plt.xlabel("Block Counts")
    plt.ylabel("Simulation Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,'block_counts_vs_simtime.png'), bbox_inches='tight')
    # plt.show()
    plt.close("all")

"""
def plot_sim_time_vs_system_char(output_dir, csv_file_addr):
    data = pd.read_csv(csv_file_addr)
    blk_cnt = list(data["blk_cnt"])
    pe_cnt = list(data["pe_cnt"])
    mem_cnt = list(data["mem_cnt"])
    bus_cnt = list(data["bus_cnt"])
    pa_sim_time = list(data["PA simulation time"])
    farsi_sim_time = list(data["FARSI simulation time"])
    tmp_reformatted_df_data = [blk_cnt * 2, pe_cnt * 2, mem_cnt * 2, bus_cnt * 2, pa_sim_time + farsi_sim_time,
                               ["PA"] * len(blk_cnt) + ["FARSI"] * len(blk_cnt)]
    reformatted_df_data = [[tmp_reformatted_df_data[j][i] for j in range(len(tmp_reformatted_df_data))] for i in
                           range(len(blk_cnt) * 2)]
    # print(reformatted_df_data[0:3])
    # exit()
    # for col in reformatted_df_data:
    #    print("Len of col is {}".format(len(col)))
    reformatted_df = pd.DataFrame(reformatted_df_data,
                                  columns=["Block counts", "PE counts", "Mem counts", "Bus counts", "Simulation Time",
                                           "FARSI or PA"])
    print(reformatted_df.head())

    df_blk_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "Block counts")
    df_pe_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "PE counts")
    df_mem_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "Mem counts")
    df_bus_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "Bus counts")

    print("Bola")
    print(df_blk_avg)



    splot = sns.scatterplot(data=df_blk_avg, x="Block counts", y="Simulation Time", hue="FARSI or PA")
    splot.set(yscale="log")

    splot_1 = sns.scatterplot(data=df_pe_avg, x="PE counts", y="Simulation Time", hue="FARSI or PA")
    splot_1.set(yscale="log")

    splot_2 = sns.scatterplot(data=df_mem_avg, x="Mem counts", y="Simulation Time", hue="FARSI or PA")
    splot_1.set(yscale="log")
    splot_3 = sns.scatterplot(data=df_bus_avg, x="Bus counts", y="Simulation Time", hue="FARSI or PA")
    splot_1.set(yscale="log")

    plt.savefig(os.path.join(output_dir,'block_counts_vs_simtime.png'))

"""




def plot_error_vs_system_char(output_dir, csv_file_addr):
    data = pd.read_csv(csv_file_addr)
    error = list(data["error"])
    blk_cnt = list(data["blk_cnt"])
    pe_cnt = list(data["pe_cnt"])
    mem_cnt = list(data["mem_cnt"])
    bus_cnt = list(data["bus_cnt"])
    #channel_cnt = list(data["channel_cnt"])
    pa_sim_time = list(data["PA simulation time"])
    farsi_sim_time = list(data["FARSI simulation time"])

    num_counts_cols = 4
    tmp_reformatted_df_data = [blk_cnt+pe_cnt+mem_cnt+bus_cnt, ["Block Counts"]*len(blk_cnt)+["PE Counts"]*len(blk_cnt) + ["Mem Counts"]*len(blk_cnt) + ["Bus Counts"]*len(bus_cnt) , error*num_counts_cols]

    reformatted_df_data = [[tmp_reformatted_df_data[j][i] for j in range(len(tmp_reformatted_df_data))] for i in range(len(blk_cnt)*num_counts_cols) ]



    #print(reformatted_df_data[0:3])
    #exit()
    #for col in reformatted_df_data:
    #    print("Len of col is {}".format(len(col)))
    reformatted_df = pd.DataFrame(reformatted_df_data, columns = ["Counts", "ArchParam", "Error"])
    print(reformatted_df.tail())


    df_avg = get_df_as_avg_for_each_x_coord(reformatted_df, x_coord_name = "Counts", y_coord_name = "Error", hue_col = "ArchParam")

    color_per_hue = {"Bus Counts" : "green", "Mem Counts" : "orange", "PE Counts" : "blue", "Block Counts" : "red", "Channel Counts" : "pink"}
    #df_avg = df_avg.loc[df_avg["ArchParam"] != "Bus Counts"]
    splot = sns.scatterplot(data=df_avg, y = "Error", x = "Counts", hue = "ArchParam", palette = color_per_hue)
    #splot.set(yscale = "log")



    #sklearn.linear_model.LinearRegression()
    hues = set(list(df_avg["ArchParam"]))
    for hue in hues:
       #x required to be in matrix format in sklearn
       print(np.isnan(df_avg["Error"]))
       xs_hue = [[x] for x in list(df_avg.loc[(df_avg["ArchParam"] == hue) & (df_avg["Error"].notnull())]["Counts"])]
       ys_hue = np.array(list(df_avg.loc[(df_avg["ArchParam"] == hue) & (df_avg["Error"].notnull())]["Error"]))
       print("xs_hue")
       print(xs_hue)

       print("ys_hue")
       print(ys_hue)
       reg = LinearRegression().fit(xs_hue, ys_hue)
       m = reg.coef_[0]
       n = reg.intercept_
       abline(m, n, color_per_hue[hue])
    #plt.set_ylim(top = 10)

    output_file = os.path.join(output_dir, "error_vs_system_char.png")
    plt.savefig(output_file)
    plt.close("all")

def plot_error_vs_system_char_for_paper(output_dir, csv_file_addr):
    data = pd.read_csv(csv_file_addr)
    error = list(data["error"])
    blk_cnt = list(data["blk_cnt"])
    pe_cnt = list(data["pe_cnt"])
    mem_cnt = list(data["mem_cnt"])
    bus_cnt = list(data["bus_cnt"])
    #channel_cnt = list(data["channel_cnt"])
    pa_sim_time = list(data["PA simulation time"])
    farsi_sim_time = list(data["FARSI simulation time"])

    num_counts_cols = 4
    tmp_reformatted_df_data = [blk_cnt+pe_cnt+mem_cnt+bus_cnt, ["Block Counts"]*len(blk_cnt)+["PE Counts"]*len(blk_cnt) + ["Memory Counts"]*len(blk_cnt) + ["NoC Counts"]*len(bus_cnt) , error*num_counts_cols]

    reformatted_df_data = [[tmp_reformatted_df_data[j][i] for j in range(len(tmp_reformatted_df_data))] for i in range(len(blk_cnt)*num_counts_cols) ]



    #print(reformatted_df_data[0:3])
    #exit()
    #for col in reformatted_df_data:
    #    print("Len of col is {}".format(len(col)))
    reformatted_df = pd.DataFrame(reformatted_df_data, columns = ["Counts", "ArchParam", "Error"])
    print(reformatted_df.tail())


    df_avg = get_df_as_avg_for_each_x_coord(reformatted_df, x_coord_name = "Counts", y_coord_name = "Error", hue_col = "ArchParam")

    color_per_hue = {"NoC Counts" : "green", "Memory Counts" : "orange", "PE Counts" : "blue", "Block Counts" : "red", "Channel Counts" : "pink"}
    #df_avg = df_avg.loc[df_avg["ArchParam"] != "Bus Counts"]
    axis_font = {'size': '20'}
    fontSize = 20
    sns.set(font_scale=2, rc={'figure.figsize': (6, 4.2)})
    sns.set_style("white")
    splot = sns.scatterplot(data=df_avg, y = "Error", x = "Counts", hue = "ArchParam", palette = color_per_hue, hue_order= ["NoC Counts", "Memory Counts", "PE Counts", "Block Counts"], sizes=(8, 8))
    #splot.set(yscale = "log")

    #sklearn.linear_model.LinearRegression()
    hues = set(list(df_avg["ArchParam"]))
    splot.legend(title="", fontsize=fontSize, loc="upper right")
    for hue in hues:
        #x required to be in matrix format in sklearn
        print(np.isnan(df_avg["Error"]))
        xs_hue = [[x] for x in list(df_avg.loc[(df_avg["ArchParam"] == hue) & (df_avg["Error"].notnull())]["Counts"])]
        ys_hue = np.array(list(df_avg.loc[(df_avg["ArchParam"] == hue) & (df_avg["Error"].notnull())]["Error"]))
        print("xs_hue")
        print(xs_hue)

        print("ys_hue")
        print(ys_hue)
        reg = LinearRegression().fit(xs_hue, ys_hue)
        m = reg.coef_[0]
        n = reg.intercept_
        abline(m, n, color_per_hue[hue])
    #plt.set_ylim(top = 10)

    plt.xticks(np.arange(-5, 30, 10.0))
    plt.yticks(np.arange(-5, 50, 10.0))
    plt.xlabel("Block Counts")
    plt.ylabel("Error (%)")
    plt.tight_layout()
    output_file = os.path.join(output_dir, "error_vs_system_char.png")
    plt.savefig(output_file, bbox_inches='tight')
    # plt.show()
    plt.close("all")

def plot_latency_vs_sim_time(output_dir, csv_file_addr):
    data = pd.read_csv(csv_file_addr)
    blk_cnt = list(data["blk_cnt"])
    pe_cnt = list(data["pe_cnt"])
    mem_cnt = list(data["mem_cnt"])
    bus_cnt = list(data["bus_cnt"])
    pa_sim_time = list(data["PA simulation time"])
    farsi_sim_time = list(data["FARSI simulation time"])
    pa_predicted_lat = list(data["PA_predicted_latency"])
    tmp_reformatted_df_data = [pa_predicted_lat * 2, pa_sim_time + farsi_sim_time,
                               ["PA"] * len(blk_cnt) + ["FARSI"] * len(blk_cnt)]
    reformatted_df_data = [[tmp_reformatted_df_data[j][i] for j in range(len(tmp_reformatted_df_data))] for i in
                           range(len(blk_cnt) * 2)]
    # print(reformatted_df_data[0:3])
    # exit()
    # for col in reformatted_df_data:
    #    print("Len of col is {}".format(len(col)))
    reformatted_df = pd.DataFrame(reformatted_df_data,
                                  columns=["PA Predicted Latency", "Simulation Time", "FARSI or PA"])

    reformatted_df = pd.DataFrame(reformatted_df_data,
                                  columns=["PA _predicted_latencys", "Simulation Time",
                                           "FARSI or PA"])
    print(reformatted_df.head())

    df_blk_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "PA _predicted_latencys")

    df_avg = get_df_as_avg_for_each_x_coord(reformatted_df, x_coord_name="PA _predicted_latencys", y_coord_name="Simulation Time",
                                            hue_col="FARSI or PA")

    # df_pe_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "PE counts")
    # df_mem_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "Mem counts")
    # df_bus_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "Bus counts")

    # print("Bola")
    # print(df_blk_avg)

    splot = sns.scatterplot(data=df_avg, x="PA _predicted_latencys", y="Simulation Time", hue="FARSI or PA")
    splot.set(yscale="log")

    color_per_hue = {"FARSI": "green", "PA": "orange"}
    hues = set(list(df_avg["FARSI or PA"]))
    for hue in hues:
        # x required to be in matrix format in sklearn
        print(np.isnan(df_avg["Simulation Time"]))
        xs_hue = [[x] for x in list(
            df_avg.loc[(df_avg["FARSI or PA"] == hue) & (df_avg["Simulation Time"].notnull())]["PA _predicted_latencys"])]
        ys_hue = np.array(
            list(df_avg.loc[(df_avg["FARSI or PA"] == hue) & (df_avg["Simulation Time"].notnull())]["Simulation Time"]))
        print("xs_hue")
        print(xs_hue)

        print("ys_hue")
        print(ys_hue)
        reg = LinearRegression().fit(xs_hue, ys_hue)
        m = reg.coef_[0]
        n = reg.intercept_
        abline(m, n, color_per_hue[hue])
    # plt.set_ylim(top = 10)

    #plt.savefig(os.path.join(output_dir, 'block_counts_vs_simtime.png'))
    plt.savefig(os.path.join(output_dir,'latency_vs_sim_time.png'))
    plt.close("all")

    """
    df_avg = get_df_as_avg_for_each_x_coord(reformatted_df, x_coord_name = "counts", y_coord_name = "Simulation Time", hue_col = "FARSI or PA")

    
    print(reformatted_df.head())
    splot = sns.scatterplot(data=reformatted_df, x="PA Predicted Latency", y="Simulation Time", hue="FARSI or PA")
    splot.set(yscale="log")

    output_file = os.path.join(output_dir, "sim_time_vs_latency.png")
    plt.savefig(output_file)
    plt.close("all")
    """

def plot_latency_vs_sim_time_for_paper(output_dir, csv_file_addr):
    data = pd.read_csv(csv_file_addr)
    blk_cnt = list(data["blk_cnt"])
    pe_cnt = list(data["pe_cnt"])
    mem_cnt = list(data["mem_cnt"])
    bus_cnt = list(data["bus_cnt"])
    pa_sim_time = list(data["PA simulation time"])
    farsi_sim_time = list(data["FARSI simulation time"])
    pa_predicted_lat = list(data["PA_predicted_latency"])
    tmp_reformatted_df_data = [pa_predicted_lat * 2, pa_sim_time + farsi_sim_time,
                               ["PA"] * len(blk_cnt) + ["FARSI"] * len(blk_cnt)]
    reformatted_df_data = [[tmp_reformatted_df_data[j][i] for j in range(len(tmp_reformatted_df_data))] for i in
                           range(len(blk_cnt) * 2)]
    # print(reformatted_df_data[0:3])
    # exit()
    # for col in reformatted_df_data:
    #    print("Len of col is {}".format(len(col)))
    reformatted_df = pd.DataFrame(reformatted_df_data,
                                  columns=["PA Predicted Latency", "Simulation Time", "FARSI or PA"])

    reformatted_df = pd.DataFrame(reformatted_df_data,
                                  columns=["PA _predicted_latencys", "Simulation Time",
                                           "FARSI or PA"])
    print(reformatted_df.head())

    df_blk_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "PA _predicted_latencys")

    df_avg = get_df_as_avg_for_each_x_coord(reformatted_df, x_coord_name="PA _predicted_latencys", y_coord_name="Simulation Time",
                                            hue_col="FARSI or PA")

    # df_pe_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "PE counts")
    # df_mem_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "Mem counts")
    # df_bus_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "Bus counts")

    # print("Bola")
    # print(df_blk_avg)

    axis_font = {'size': '20'}
    fontSize = 20
    sns.set(font_scale=2, rc={'figure.figsize': (6, 4)})
    sns.set_style("white")
    color_per_hue = {'PA': 'hotpink', 'FARSI': 'green'}
    splot = sns.scatterplot(data=df_avg, x="PA _predicted_latencys", y="Simulation Time", hue="FARSI or PA", palette=color_per_hue)
    splot.set(yscale="log")
    splot.legend(title="", fontsize=fontSize, loc="center right")

    hues = set(list(df_avg["FARSI or PA"]))
    for hue in hues:
        # x required to be in matrix format in sklearn
        print(np.isnan(df_avg["Simulation Time"]))
        xs_hue = [[x] for x in list(
            df_avg.loc[(df_avg["FARSI or PA"] == hue) & (df_avg["Simulation Time"].notnull())]["PA _predicted_latencys"])]
        ys_hue = np.array(
            list(df_avg.loc[(df_avg["FARSI or PA"] == hue) & (df_avg["Simulation Time"].notnull())]["Simulation Time"]))
        print("xs_hue")
        print(xs_hue)

        print("ys_hue")
        print(ys_hue)
        reg = LinearRegression().fit(xs_hue, ys_hue)
        m = reg.coef_[0]
        n = reg.intercept_
        abline(m, n, color_per_hue[hue])
    # plt.set_ylim(top = 10)

    plt.xticks(np.arange(0, 60, 10.0))
    plt.yticks(np.power(10.0, [-1, 0, 1, 2, 3]))
    plt.xlabel("Execution latency")
    plt.ylabel("Simulation Time (s)")
    plt.tight_layout()
    #plt.savefig(os.path.join(output_dir, 'block_counts_vs_simtime.png'))
    plt.savefig(os.path.join(output_dir,'latency_vs_sim_time.png'), bbox_inches='tight')
    # plt.show()
    plt.close("all")

if __name__ == "__main__":  # Ying: for aggregate_data
    run_folder_name = config_plotting.run_folder_name
    csv_file_addr = os.path.join(run_folder_name, "input_data","aggregate_data.csv")
    output_dir = os.path.join(run_folder_name, "validation")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if config_plotting.draw_for_paper:  # Ying: "cross_workloads", from aggregate_data
        plot_error_vs_system_char_for_paper(output_dir, csv_file_addr)
        plot_sim_time_vs_system_char_minimal_for_paper(output_dir, csv_file_addr)
        plot_latency_vs_sim_time_for_paper(output_dir, csv_file_addr)
    else:
        plot_error_vs_system_char(output_dir, csv_file_addr)
        plot_sim_time_vs_system_char_minimal(output_dir, csv_file_addr)
        plot_latency_vs_sim_time(output_dir, csv_file_addr)
