#Author: Iulian Brumar
#Script that plots Simulation time vs PA predicted latency

import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv(sys.argv[1])
blk_cnt = list(data["blk_cnt"])
pe_cnt = list(data["pe_cnt"])
mem_cnt = list(data["mem_cnt"])
bus_cnt = list(data["bus_cnt"])
pa_sim_time = list(data["PA simulation time"])
farsi_sim_time = list(data["FARSI simulation time"])
pa_predicted_lat = list(data["PA_predicted_latency"])
tmp_reformatted_df_data = [pa_predicted_lat*2, pa_sim_time+farsi_sim_time, ["PA"]*len(blk_cnt)+["FARSI"]*len(blk_cnt)]
reformatted_df_data = [[tmp_reformatted_df_data[j][i] for j in range(len(tmp_reformatted_df_data))] for i in range(len(blk_cnt)*2) ]
#print(reformatted_df_data[0:3])
#exit()
#for col in reformatted_df_data:
#    print("Len of col is {}".format(len(col)))
reformatted_df = pd.DataFrame(reformatted_df_data, columns = ["PA Predicted Latency", "Simulation Time", "FARSI or PA"])
print(reformatted_df.head())
splot = sns.scatterplot(data = reformatted_df, x = "PA Predicted Latency", y = "Simulation Time", hue = "FARSI or PA")
splot.set(yscale="log")
plt.savefig('pa_lat_vs_simtime_logy.png')

