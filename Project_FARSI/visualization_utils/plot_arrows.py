#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
from settings import config
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from copy import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# fine tuning how we plot arrows
def plot3d_arrow():
    input_list = [[0, 0, 0, 1, 3, 1], [1,1,1, 3,3,3]]
    min_bounds = {}
    max_bounds = {}

    # prepare the input
    for el in ["x", "y", "z"]:
        min_bounds[el] = 1000
        max_bounds[el] = -1000

    for input in input_list:
        x_0, y_0, z_0, x_1, y_1, z_1   = input
        min_bounds["x"] = min(min_bounds["x"], x_0, x_1)
        min_bounds["y"] = min(min_bounds["y"], y_0, y_1)
        min_bounds["z"] = min(min_bounds["z"], z_0, z_1)
        max_bounds["x"] = max(max_bounds["x"], x_0, x_1)
        max_bounds["y"] = max(max_bounds["y"], y_0, y_1)
        max_bounds["z"] = max(max_bounds["z"], z_0, z_1)

    soa = np.array([[1, 1, 1, 4, 4, 4], [4,4,4, 1,1,1]])

    # plot
    X, Y, Z, U, V, W = zip(*soa)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V,W, arrow_length_ratio=.05)
    ax.set_xlim(.8*min_bounds["x"], 1.4*max_bounds["x"])
    ax.set_ylim(.8*min_bounds["y"], 1.4*max_bounds["y"])
    ax.set_zlim(.8*min_bounds["z"], 1.4*max_bounds["z"])
    plt.show()

plot3d_arrow()
