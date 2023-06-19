#Author: Iulian Brumar
#Script that plots average simulation time (PA/FARSI hue) vs different architectural parameters

import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import numpy as np


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

if __name__ == "__main__":
   data = pd.read_csv(sys.argv[1])
   blk_cnt = list(data["blk_cnt"])
   pe_cnt = list(data["pe_cnt"])
   mem_cnt = list(data["mem_cnt"])
   bus_cnt = list(data["bus_cnt"])
   pa_sim_time = list(data["PA simulation time"])
   farsi_sim_time = list(data["FARSI simulation time"])
   tmp_reformatted_df_data = [blk_cnt*2, pe_cnt*2, mem_cnt*2, bus_cnt*2, pa_sim_time+farsi_sim_time, ["PA"]*len(blk_cnt)+["FARSI"]*len(blk_cnt)]
   reformatted_df_data = [[tmp_reformatted_df_data[j][i] for j in range(len(tmp_reformatted_df_data))] for i in range(len(blk_cnt)*2) ]
   #print(reformatted_df_data[0:3])
   #exit()
   #for col in reformatted_df_data:
   #    print("Len of col is {}".format(len(col)))
   reformatted_df = pd.DataFrame(reformatted_df_data, columns = ["Block counts", "PE counts", "Mem counts", "Bus counts", "Simulation Time", "FARSI or PA"])
   print(reformatted_df.head())


   df_blk_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "Block counts")
   df_pe_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "PE counts")
   df_mem_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "Mem counts")
   df_bus_avg = get_df_as_avg_for_each_x_coord(reformatted_df, "Bus counts")

   print("Bola")
   print(df_blk_avg)

   splot = sns.scatterplot(data=df_blk_avg, x = "Block counts", y = "Simulation Time", hue = "FARSI or PA")
   splot.set(yscale = "log")
   plt.savefig('block_counts_vs_simtime.png')

   plt.clf()
   splot = sns.scatterplot(data=df_pe_avg, x = "PE counts", y = "Simulation Time", hue = "FARSI or PA")
   splot.set(yscale = "log")
   plt.savefig('PE_counts_vs_simtime.png')


   plt.clf()
   splot = sns.scatterplot(data=df_mem_avg, x = "Mem counts", y = "Simulation Time", hue = "FARSI or PA")
   splot.set(yscale = "log")
   plt.savefig('MEM_counts_vs_simtime.png')


   plt.clf()
   splot = sns.scatterplot(data=df_bus_avg, x = "Bus counts", y = "Simulation Time", hue = "FARSI or PA")
   splot.set(yscale = "log")
   plt.savefig('BUS_counts_vs_simtime.png')
