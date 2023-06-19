#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

from design_utils.design import *
import os
from sys import platform

# ------------------------------
# Functionality:
#       visualize all the stats associated with the results.
# Variables:
#       dp_stats: design point stats. statistical information (such as latency, energy, ...) associated with the design.
# ------------------------------
def vis_stats(dpstats):
    sorted_kernels = dpstats.get_kernels_sort()
    stats_output_file = config.stats_output
    with open(stats_output_file, "w") as output:
        for kernel in sorted_kernels:
            output.write("\n-------------\n") 
            output.write("kernel name:" + kernel.get_task_name()+ "\n")
            output.write("      total work:"+ str(kernel.get_total_work()) + "\n")
            output.write("      blocks mapped to" + str(kernel.get_block_list_names()) + "\n")
            output.write("      latency" + str(kernel.stats.latency) + "\n")
            phase_block_duration_bottleneck = kernel.stats.phase_block_duration_bottleneck
            phase_block_duration_bottleneck_printable = [(phase, block_duration[0].instance_name, block_duration[1]) for phase, block_duration in phase_block_duration_bottleneck.items()]
            output.write("      phase,block,bottlneck_duration:" + str(phase_block_duration_bottleneck_printable) + "\n")
        
    if platform == "linux" or platform == "linux2":
        os.system("soffice --convert-to png " + stats_output_file)
        os.system("convert " + stats_output_file+".png" " to " + stats_output_file+".pdf")
    elif platform == "darwin":
        os.system("textutil -convert html " + stats_output_file)
        os.system("cupsfilter " + stats_output_file +".html > test.pdf")