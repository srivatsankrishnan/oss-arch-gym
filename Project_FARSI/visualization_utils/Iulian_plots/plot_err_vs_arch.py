#Author: Iulian Brumar
#Script that plots error with average per different arch parameters

import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("..")
from plot_validations import *
from sklearn.linear_model import LinearRegression

def abline(slope, intercept, color):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', color = color)


data = pd.read_csv(sys.argv[1])
error = list(data["error"])
blk_cnt = list(data["blk_cnt"])
pe_cnt = list(data["pe_cnt"])
mem_cnt = list(data["mem_cnt"])
bus_cnt = list(data["bus_cnt"])
channel_cnt = list(data["channel_cnt"])
pa_sim_time = list(data["PA simulation time"])
farsi_sim_time = list(data["FARSI simulation time"])

num_counts_cols = 5
tmp_reformatted_df_data = [blk_cnt+pe_cnt+mem_cnt+bus_cnt+channel_cnt, ["Block Counts"]*len(blk_cnt)+["PE Counts"]*len(blk_cnt) + ["Mem Counts"]*len(blk_cnt) + ["Bus Counts"]*len(bus_cnt) + ["Channel Counts"]*len(bus_cnt), error*num_counts_cols]

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
plt.savefig('error_vs_c_reg.png')




