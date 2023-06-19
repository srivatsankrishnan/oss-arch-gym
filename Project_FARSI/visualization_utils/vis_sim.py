#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import plotly.graph_objs as go
from design_utils.design import *

# some global variables
x_min = -3
vertical_slice_size =.5
vertical_slice_distance = 2*vertical_slice_size
color_palette =['orange', 'green', 'red']

# ------------------------------
# Functionality:
#       used to offset the bars for broken bar graph
# ------------------------------
def next_y_offset():
    global vertical_slice_distance
    offset = vertical_slice_distance
    while True:
        yield offset
        offset += vertical_slice_distance

# ------------------------------
# Functionality:
#       broken bar type plots.
# Variables:
#       xstart: starting point for the plot on x axis
#       xwidth: with of the plot
#       ystart: starting point for the plot on y axis
#       colors: colors associated with the plot
# ------------------------------
def broken_bars(xstart, xwidth, ystart, yh, colors):
    if len(xstart) != len(xwidth) or len(xstart) != len(colors):
        raise ValueError('xstart, xwidth and colors must have the same length')
    shapes = []
    for k in range(len(xstart)):
        shapes.append(dict(type="rect",
                           x0=xstart[k],
                           y0=ystart,
                           x1=xstart[k] + xwidth[k],
                           y1=ystart + yh,
                           fillcolor=colors[k],
                           line_color=colors[k],
                           opacity = .4,
                           ))
    return shapes

# ------------------------------
# Functionality:
#       broken barh type plots.
# Variables:
#       name__start_width_metric_unit_dict: name of the metric, starting point, with, and it's unit,
#                                           compacted into a dictionary.
# ------------------------------
def plotly_broken_barh(name__start_width_metric_unit_dict):
    global x_min
    name_shapes_dict = {}  # name_plotly dict
    name_centroid_metric_unit_dict = {}
    next_y = next_y_offset()
    max_x = 0
    ctr = 0

    # extract all the values
    for name, values in name__start_width_metric_unit_dict.items():
        start_list = [value[0] for value in values]
        width_list = [value[1] for value in values]
        metric_list = [value[2] for value in values]
        unit_list = [value[3] for value in values]
        y = next(next_y)

        # set the text
        centroid_metric_unit_list = []
        for value in values:
            ctr_x = value[0]+float(value[1]/2)
            ctr_y = y + vertical_slice_size/2
            metric = value[2]
            unit = value[3]
            centroid_metric_unit_list.append((ctr_x, ctr_y, metric, unit))
        name_centroid_metric_unit_dict[name] = centroid_metric_unit_list

        # set the shapes
        max_x = max(max_x, (start_list[-1] + width_list[-1]))
        name_shapes_dict[name] = broken_bars(start_list, width_list, y, vertical_slice_size,
                                             len(start_list)*[color_palette[ctr % len(color_palette)]])
        ctr += 1

    fig = go.Figure()

    # add all the shapes
    list_of_lists_of_shapes = list(name_shapes_dict.values())
    flattented_list_of_shapes = [item for sublist in list_of_lists_of_shapes for item in sublist]

    # add annotations to the figure
    flattned_list_of_centroid_metric_unit = [item for sublist in list(name_centroid_metric_unit_dict.values()) for item in sublist]
    get_ctroid_x  = lambda input_list: [el[0] for el in input_list]
    get_ctroid_y  = lambda input_list: [el[1]  for el in input_list]
    get_metric_unit  = lambda input_list: [str(el[2])+"("+el[3]+")" for el in input_list]

    x = get_ctroid_x(flattned_list_of_centroid_metric_unit),

    fig.add_trace(go.Scatter(
        x= get_ctroid_x(flattned_list_of_centroid_metric_unit),
        y= get_ctroid_y(flattned_list_of_centroid_metric_unit),
        mode="text",
        text= get_metric_unit(flattned_list_of_centroid_metric_unit),
        textposition="bottom center"
    ))
    x_min = -.1*max_x
    name_zero_y = []
    for name, value in name_centroid_metric_unit_dict.items():
        name_zero_y.append((name, x_min, value[0][1]))
    fig.add_trace(go.Scatter(
        x= [el[1] for el in name_zero_y],
        y= [el[2] for el in name_zero_y],
        mode="text",
        text= [el[0] for el in name_zero_y],
        textposition="top right"
    ))
    fig.update_layout(
        xaxis = dict(range=[x_min, 1.1*max_x], title="time"),
        yaxis = dict(range=[0, next(next_y)], visible=False),
        shapes= flattented_list_of_shapes)

    return fig

# ------------------------------
# Functionality:
#       save the result into a html for visualization.
# Variables:
#       fig: figure to save.
#       file_addr: address of the file to output the result to.
# ------------------------------
def save_to_html(fig, file_addr):
    fig_json = fig.to_json()

    # a simple HTML template
    template = """<html>
    <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div id='divPlotly'></div>
        <script>
            var plotly_data = {}
            Plotly.react('divPlotly', plotly_data.data, plotly_data.layout);
        </script>
    </body>

    </html>"""

    # write the JSON to the HTML template
    with open(file_addr, 'w') as f:
        f.write(template.format(fig_json))

# ------------------------------
# Functionality:
#       plot simulation progress.
# Variables:
#       dp_stats: design point stats. statistical information (such as latency, energy, ...) associated with the design.
#       ex_dp: example design point.
#       result_folder: the folder to dump the results in.
# ------------------------------
def plot_sim_data(dp_stats, ex_dp, result_folder):
    # plot latency
    kernel__metric_value_per_SOC = dp_stats.get_sim_progress("latency")
    for kernel__metrtic_value in kernel__metric_value_per_SOC:
        name__start_width_metric_unit_dict = {}
        kernel_end_time_dict = {}
        for kernel in kernel__metrtic_value.keys():
            name__start_width_metric_unit_dict[kernel.get_task_name()] = []


        phase_latency_dict = dp_stats.get_phase_latency()
        phase_start_end_dict = {}
        start = 0
        for phase, duration in phase_latency_dict.items():
            phase_start_end_dict[phase] = [start, start + duration]
            start += duration

        for kernel, values in kernel__metrtic_value.items():
            for value in values:
                metric = value[2]
                unit = value[3]
                break
            break


        for kernel,phases_operating_state in dp_stats.dp.krnl_phase_present_operating_state.items():
            last_phase_number = 9999
            last_operating_state = "na"
            starting_phase = True
            for phase, operating_state in phases_operating_state:
                start, end = phase_start_end_dict[phase]
                if last_phase_number == phase - 1 and last_operating_state == operating_state:
                    start = (name__start_width_metric_unit_dict[kernel.get_task_name()][-1])[0]
                    label = operating_state[0]+":"+str("{:.4f}".format(end))
                    name__start_width_metric_unit_dict[kernel.get_task_name()][-1] = (start, end - start, label, "")
                else:
                    label = operating_state[0]+":"+str("{:.4f}".format(end))
                    name__start_width_metric_unit_dict[kernel.get_task_name()].append((start, end-start, label, ""))
                if starting_phase:
                    kernel_end_time_dict[kernel.get_task_name()] = start
                    starting_phase = False
                last_phase_number = phase
                last_operating_state = operating_state
        """
        for kernel, values in kernel__metrtic_value.items():
            first_start_time = values[0][0]
            for value in values:
                start = value[0]
                width = value[1]
                metric = value[2]
                unit = value[3]
                name__start_width_metric_unit_dict[kernel.get_task_name()].append((start, width, metric, unit))
                last_start = start
                last_width = width
            kernel_end_time_dict[kernel.get_task_name()] = first_start_time
        """
        # now sort it based on end time
        sorted_kernel_end_time = sorted(kernel_end_time_dict.items(),
                          key=operator.itemgetter(1), reverse=True)

        sorted_name__start_width_metric_unit_dict = {}
        for element in sorted_kernel_end_time:
            kernel_name = element[0]
            sorted_name__start_width_metric_unit_dict[kernel_name] = name__start_width_metric_unit_dict[kernel_name]
        fig = plotly_broken_barh(sorted_name__start_width_metric_unit_dict)
        save_to_html(fig, result_folder+"/"+"latest.html")

    # color map
    my_cmap = ["hotpink", "olive",  "gold", "darkgreen", "turquoise", "crimson",
               "lightblue", "darkorange", "yellow",
               "chocolate", "darkorchid", "greenyellow"]

    # plot utilization:
    fig, ax = plt.subplots()
    ax.set_ylabel('Utilizaiton (%)', fontsize=15)
    ax.set_xlabel('Phase', fontsize=15)
    ax.set_title('Block Utilization', fontsize=15)
    ctr = 0
    for type,id in ex_dp.get_designs_SOCs():
        block_phase_utilization = dp_stats.get_SOC_s_sim_utilization(type, id)
        for block, phase_utilization in block_phase_utilization.items():
            if not block.type == "ic":  # for now, mainly interested in memory
                continue
            block_name = block.instance_name
            phase = list(phase_utilization.keys())
            utilization = [x*100 for x in list(phase_utilization.values())]
            plt.plot(phase, utilization, marker='>', linewidth=.6, color=my_cmap[ctr%len(my_cmap)], ms=1, label=block_name)
            ctr +=1
        ax.legend(prop={'size': 10}, ncol=1, bbox_to_anchor=(1.01, 1), loc='upper left')
        fig.tight_layout()
        plt.savefig(result_folder+"/FARSI_estimated_Block_utilization_"+str(type)+str(id))

    sorted_listified_phase_latency_dict = sorted(dp_stats.dp.phase_latency_dict.items(), key=operator.itemgetter(0))
    phase_begin_end = {}
    for phase,duration in sorted_listified_phase_latency_dict:
        if phase == -1:
            phase_begin_end[phase] = (0,0)
        else:
            last_phase_end = phase_begin_end[phase - 1][1]
            phase_begin_end[phase] = (last_phase_end, last_phase_end + duration)

    phase_ending_time = {}
    #for phase, duration in sorted_listified_phase_latency_dict:
    #    if phase == -1
    for dir__ in ["write", "read"]:
        # plot  bandwidth:
        fig, ax = plt.subplots()
        ax.set_ylabel('Bandwidth write (%)', fontsize=15)
        ax.set_xlabel('Phase', fontsize=15)
        ax.set_title('Path Bandwidth for ' + dir__, fontsize=15)
        ctr = 0
        markers = ["o", ">", "1", "8", "s", "p"]
        seen = []
        seen_values = {}
        for type, id in ex_dp.get_designs_SOCs():
            for pipe_cluster, pipe_phase_work_rate in dp_stats.get_SOC_s_pipe_cluster_pathlet_phase_work_rate(type,
                                                                                                           id).items():
                if pipe_cluster.cluster_type == "dummy" or not pipe_cluster.get_dir() == dir__:
                    continue
                if not pipe_cluster.get_block_ref().type  == "ic":
                    continue
                block_name = '_'.join(pipe_cluster.get_block_ref().instance_name.split("_")[-3:])
                dir_ = pipe_cluster.get_dir()
                for path, phase_work_rate in pipe_phase_work_rate.items():
                    in_pipe, out_pipe  = path.get_in_pipe(), path.get_out_pipe()
                    if in_pipe.get_master().type == "pe":
                        master_name = '_'.join(in_pipe.get_master().instance_name.split("_")[:3])
                    else:
                        master_name = '_'.join(in_pipe.get_master().instance_name.split("_")[-3:])
                    if out_pipe == None:
                        slave_name == "non"
                    else:
                        if out_pipe.get_slave().type == "pe":
                            slave_name = '_'.join(out_pipe.get_slave().instance_name.split("_")[:3])
                        else:
                            slave_name = '_'.join(out_pipe.get_slave().instance_name.split("_")[-3:])
                    name = master_name + "__" + block_name + "__" + slave_name + "__" + dir_
                    if name in seen:
                        continue
                    seen.append(name)
                    phases = list(phase_work_rate.keys())
                    bandwidths = [int(x / 1000000) for x in list(phase_work_rate.values())]
                    x = []
                    y = []
                    for phase in list(phase_work_rate.keys()):
                        bandwidth = int(phase_work_rate[phase]/1000000)
                        begin_t, end_t = phase_begin_end[phase]
                        x.append(begin_t)
                        x.append(end_t)
                        y.append(bandwidth)
                        y.append(bandwidth)
                    values = '_'.join([str((el[0], el[1])) for el in zip(phases, bandwidths)])
                    if values in seen_values.keys():
                        ctr_ = seen_values[values]
                        plt.plot(x, y, marker=markers[ctr % len(markers)], linewidth=4,
                                 color=my_cmap[ctr_ % len(my_cmap)], ms=1,
                                 label=name)
                    else:
                        seen_values[values] = ctr
                        plt.plot(x, y, marker=markers[ctr % len(markers)], linewidth=4,
                                 color=my_cmap[ctr % len(my_cmap)], ms=1,
                                 label=name)
                        ctr += 1
                    for a, b in zip(x, y):
                        plt.text(a, b, str(b))

            ax.legend(prop={'size': 8}, ncol=1, loc='best')
            fig.tight_layout()
            plt.savefig(result_folder + "/FARSI_estimated_pathlet_bandwidth_"+dir__ + str(type) + str(id))

    """ 
    # draw pathlet latency
    for dir__ in ["write", "read"]:
        # plot  bandwidth:
        fig, ax = plt.subplots()
        ax.set_ylabel('pathlet latency (in cycles)', fontsize=15)
        ax.set_xlabel('Phase', fontsize=15)
        ax.set_title('Path latency for ' + dir__, fontsize=15)
        ctr = 0
        markers = ["o", ">", "1", "8", "s", "p"]
        seen = []
        seen_values = {}
        for type, id in ex_dp.get_designs_SOCs():
            for pipe_cluster, pathlet_phase_latency in dp_stats.get_SOC_s_pipe_cluster_path_phase_latency(type,
                                                                                                           id).items():
                if pipe_cluster.cluster_type == "dummy" or not pipe_cluster.get_dir() == dir__:
                    continue
                if not pipe_cluster.get_block_ref().type  == "ic":
                    continue
                block_name = '_'.join(pipe_cluster.get_block_ref().instance_name.split("_")[-3:])
                dir_ = pipe_cluster.get_dir()
                for pathlet_, phase_latency in pathlet_phase_latency.items():
                    in_pipe, out_pipe  = pathlet_.get_in_pipe(), pathlet_.get_out_pipe()
                    if in_pipe.get_master().type == "pe":
                        master_name = '_'.join(in_pipe.get_master().instance_name.split("_")[:3])
                    else:
                        master_name = '_'.join(in_pipe.get_master().instance_name.split("_")[-3:])
                    if out_pipe == None:
                        slave_name == "non"
                    else:
                        if out_pipe.get_slave().type == "pe":
                            slave_name = '_'.join(out_pipe.get_slave().instance_name.split("_")[:3])
                        else:
                            slave_name = '_'.join(out_pipe.get_slave().instance_name.split("_")[-3:])
                    name = master_name + "__" + block_name + "__" + slave_name + "__" + dir_
                    if name in seen:
                        continue
                    seen.append(name)
                    phases = list(phase_latency.keys())
                    latencies= [x  for x in list(phase_latency.values())]
                    x = []
                    y = []
                    for phase in list(phase_latency.keys()):
                        latency = phase_latency[phase]
                        begin_t, end_t = phase_begin_end[phase]
                        x.append(begin_t)
                        x.append(end_t)
                        y.append(latency)
                        y.append(latency)
                    values = '_'.join([str((el[0], el[1])) for el in zip(phases, latencies)])
                    if values in seen_values.keys():
                        ctr_ = seen_values[values]
                        plt.plot(x, y, marker=markers[ctr % len(markers)], linewidth=4,
                                 color=my_cmap[ctr_ % len(my_cmap)], ms=1,
                                 label=name)
                    else:
                        seen_values[values] = ctr
                        plt.plot(x, y, marker=markers[ctr % len(markers)], linewidth=4,
                                 color=my_cmap[ctr % len(my_cmap)], ms=1,
                                 label=name)
                        ctr += 1
                    for a, b in zip(x, y):
                        plt.text(a, b, str(b))

            ax.legend(prop={'size': 8}, ncol=1, loc='best')
            fig.tight_layout()
            plt.savefig(result_folder + "/FARSI_estimated_pathlet_latency_"+dir__ + str(type) + str(id))
    """

    plt.close('all')
