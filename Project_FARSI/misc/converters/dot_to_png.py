#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import os
import sys
import pygraphviz as pgv
from sys import argv


def converter():
    file = sys.argv[1]
    hardware_dot_graph = pgv.AGraph(file)
    hardware_dot_graph.layout()
    hardware_dot_graph.layout(prog='circo')
    output_file_name_1 = file.split(".dot")[0] + ".png"
    output_file_1 = os.path.join("./", output_file_name_1)
    # output_file_real_time_vis = os.path.join(".", output_file_name)  # this is used for realtime visualization
    hardware_dot_graph.draw(output_file_1, format = "png", prog='circo')


if __name__ == "__main__":
    converter()
