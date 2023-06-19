#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import pygraphviz as pgv
from design_utils.design import *
from sys import platform
import time
import sys

# global  value
ctr = 0

# This class contains information about each node (block) in the hardware graph.
# Only used for visualization purposes.
class node_content():
    # ------------------------------
    # Functionality
    #       constructor
    # Variables:
    #       block: block of interest
    #       mode: obfuscated or not (for outside or inside distribution) we obfuscate the name and content of the nodes.
    #       task_obfuscated_table: contains a table mapping task's real name to their obfuscated names
    #       block_obfuscated_table: contains a table mapping blocks' real name to their obfuscated names
    # ------------------------------
    def __init__(self, block:Block, mode, task_obfuscated_table, block_obfuscated_table):
        self.block = block
        self.content = ""
        if (mode == "block"):
            self.content = block_obfuscated_table[self.block.instance_name]
        elif (mode == "block_extra"):
            self.content = block_obfuscated_table[self.block.instance_name]
            if self.block.type == "ic":
                self.content += " , width(Bits):" + str((self.block.peak_work_rate/database_input.ref_ic_clock)*8) +\
                    " , clock(MHz)" + str(database_input.ref_ic_clock/10**6)
            elif self.block.type == "mem":
                self.content += " , width(Bits):" + str((self.block.peak_work_rate/database_input.ref_mem_clock)*8) +\
                    " , clock(MHz)" + str(database_input.ref_mem_clock/10**6) \
                                 + " , size(kB):" + str(self.block.get_area()*database_input.ref_mem_work_over_area/10**3)
            elif self.block.subtype == "gpp":
                tasks = self.block.get_task_dirs_of_block()
                task_names_dirs = [task_obfuscated_table[str(task_.name)] for task_,dir in tasks]
                self.content += ":"+ ",".join(task_names_dirs)
        elif (mode == "block_task"):
            tasks = self.block.get_task_dirs_of_block()
            task_names_dirs = [str((task_obfuscated_table[task_.name],dir)) for task_,dir in tasks]
            if not(self.block.instance_name in block_obfuscated_table.keys()):
                print("what")
            self.content = block_obfuscated_table[self.block.instance_name] +"("+ self.block.type +")" +": "+ ",".join(task_names_dirs)
        if self.only_dummy_tasks(block): self.only_dummy = True
        else: self.only_dummy = False

    # ------------------------------
    # Functionality:
    #       determines whether a task is a dummy or real task
    #       (PS: dummy tasks are located at the root (named with souurce suffix)
    #       and leaf (named with siink suffix) of the graph
    # Variables:
    #       block: variable of interest.
    # ------------------------------
    def only_dummy_tasks(self, block):
        num_of_tasks = len(block.get_tasks_of_block())
        if num_of_tasks == 2:
            a = [task.name for task in block.get_tasks_of_block()]
            if any("souurce" in task.name for task in block.get_tasks_of_block()) and \
                    any("siink" in task.name for task in block.get_tasks_of_block()):
                return True
        elif num_of_tasks == 1:
            a = [task.name for task in block.get_tasks_of_block()]
            if any("souurce" in task.name for task in block.get_tasks_of_block()) or \
                    any("siink" in task.name for task in block.get_tasks_of_block()):
                return True
        else:
            return False

    def format_content(self, content):
        n = 50
        chunks = [content[i:i + n] for i in range(0, len(content), n)]
        return "\n".join(chunks)

    # ------------------------------
    # Functionality:
    #      get all the contents: tasks, blocks and information about each.
    # ------------------------------
    def get_content(self):
        return self.format_content(self.content)

    # ------------------------------
    # Functionality:
    #      get coloring associated with each processing block.
    # ------------------------------
    def get_color(self):
        if self.block.type == "mem":
            return "cyan3"
        elif self.block.type == "ic":
            return "white"
        else:
            if self.block.subtype == "gpp" and "A53" in self.block.instance_name:
                return "gold"
            elif self.block.subtype == "gpp" and "G" in self.block.instance_name:
                return "goldenrod3"
            elif self.block.subtype == "gpp" and "P" in self.block.instance_name:
                return "goldenrod2"
            else:
                return "orange"


# ------------------------------
# Functionality:
#      build a dot compatible graph recursively. This is provided to the dot visualizer after.
# Variables:
#       parent_block: parent of a block (block it reads from).
#       child_block: child of a block (block it writes to).
#       blocks_visited: blocks already visit  (used for preventing double graphing in depth first search).
#       hardware_dot_graph: dot compatible graph associated with our hardware graph.
#       task_obfuscated_table: contains a table mapping task's real name to their obfuscated name.
#       block_obfuscated_table: contains a table mapping block's real name to their obfuscated names.
#       graphing_mode:  mode determines whether to obfuscate (Names and content) or
#       not (for outside distribution or not).
# ------------------------------
def build_dot_recursively(parent_block, child_block, blocks_visited, hardware_dot_graph,
                          task_obfuscated_table, block_obfuscatred_table,  graphing_mode):
    if (child_block, parent_block) in blocks_visited:
        return None
    global ctr
    if parent_block:
        parent_node = node_content(parent_block, graphing_mode, task_obfuscated_table, block_obfuscatred_table)
        if not parent_node.only_dummy:
            hardware_dot_graph.add_node(parent_node.get_content(),  fillcolor=parent_node.get_color())
        child_node = node_content(child_block, graphing_mode, task_obfuscated_table, block_obfuscatred_table)
        if not child_node.only_dummy:
            hardware_dot_graph.add_node(child_node.get_content(),  fillcolor=child_node.get_color())
        ctr +=1
        if not parent_node.only_dummy and not child_node.only_dummy:
            hardware_dot_graph.add_edge(parent_node.get_content(), child_node.get_content())
    blocks_visited.append((child_block, parent_block))
    parent_block = child_block 
    for child_block_ in parent_block.neighs:
        build_dot_recursively(parent_block, child_block_, blocks_visited, hardware_dot_graph, task_obfuscated_table,
                              block_obfuscatred_table, graphing_mode)

# ------------------------------
# Functionality:
#      generating the obfuscation table, so the result can be distributed for outside companies/acadamia as well.
#      Obfuscation table contains mapping from real names to fake names.
#      In addition, the contents (e.g., computational load) are eliminated.
# Variables:
#       sim_dp: design point simulation.
# ------------------------------
def gen_obfuscation_table(sim_dp:SimDesignPoint):
    block_names = [block.instance_name for block in sim_dp.hardware_graph.blocks]
    ctr = 0
    task_obfuscated_table = {}
    block_obfuscated_table = {}
    # obfuscate the tasks
    for task in sim_dp.get_tasks():
        if config.DATA_DELIVEYRY == "obfuscate":
            task_obfuscated_table[task.name] = "T" + str(ctr)
            ctr +=1
        else:
            task_obfuscated_table[task.name] = task.name

    # obfuscate the blocks
    dsp_ctr = 0
    gpp_ctr = 0
    mem_ctr = 0
    ic_ctr = 0
    for block in sim_dp.get_blocks():
        got_name = False
        if block.type == "pe" and config.DATA_DELIVEYRY == "obfuscate":
            if block.subtype == "ip":
                name = ""
                for task in block.get_tasks_of_block():
                    name += (task_obfuscated_table[task.name]+"_")
                block_obfuscated_table[block.instance_name] = name + "ip"
            else:  # gpp
                if "G" in block.instance_name:
                    block_obfuscated_table[block.instance_name] = "DSP_G3_"  + str(dsp_ctr)
                    dsp_ctr += 1
                elif "P" in block.instance_name:
                    block_obfuscated_table[block.instance_name] = "DSP_P6_" + str(dsp_ctr)
                    dsp_ctr += 1
                elif "A" in block.instance_name:
                    block_obfuscated_table[block.instance_name] = "GPP" + str(gpp_ctr)
                    gpp_ctr += 1
                else:
                    print("this processor" + str(block.instance_name) + "is not obfuscated")
                    exit(0)
        elif block.type == "mem" and config.DATA_DELIVEYRY == "obfuscate":
            block_obfuscated_table[block.instance_name] = block.instance_name.split("_")[0][1:]+str(mem_ctr)
            mem_ctr +=1
        elif block.type == "ic" and config.DATA_DELIVEYRY == "obfuscate":
            block_obfuscated_table[block.instance_name] = "NOC"+str(ic_ctr)
            ic_ctr +=1
        else:
            block_obfuscated_table[block.instance_name] = block.instance_name

    return task_obfuscated_table, block_obfuscated_table


# ------------------------------
# Functionality:
#      visualizing the hardware graph and task dependencies and mapping between the two using dot graph.
# Variables:
#       sim_dp: design point simulation.
#       graphing_mode:  mode determines whether to
#       obfuscate (Names and content) or not (for outside distribution or not).
#       sim_dp: design point simulation.
# ------------------------------
def vis_hardware(sim_dp:SimDesignPoint, graphing_mode=config.hw_graphing_mode, output_folder=config.latest_visualization,
                 output_file_name="system_image.pdf"):

    try:
        output_file_name_1 = os.path.join(output_folder, output_file_name)
        output_file_name_2 =  config.latest_visualization+"/system_image.pdf"
        if not os.path.exists(config.latest_visualization):
            os.system("mkdir -p " + config.latest_visualization)

        global ctr
        ctr = 0

        if not (sys.platform == "darwin"):
            output_file_name_1 = output_file_name_1.split(".pdf")[0] + ".dot"
            output_file_name_2 = output_file_name_2.split(".pdf")[0] + ".dot"

        hardware_dot_graph =pgv.AGraph()
        hardware_dot_graph.node_attr['style'] = 'filled'
        hardware_graph = sim_dp.get_hardware_graph()
        root = hardware_graph.get_root()

        task_obfuscated_table, block_obfuscated_table = gen_obfuscation_table(sim_dp)

        build_dot_recursively(None, root, [], hardware_dot_graph, task_obfuscated_table, block_obfuscated_table, graphing_mode )
        blah = ctr
        hardware_dot_graph.layout()
        hardware_dot_graph.layout(prog='circo')
        hardware_dot_graph
        time.sleep(.0008)

        output_file_1 = os.path.join(output_folder, output_file_name_1)
        output_file_2 = os.path.join(output_folder, output_file_name_2)
        #output_file_real_time_vis = os.path.join(".", output_file_name)  # this is used for realtime visualization
        if graphing_mode == "block_extra":
            hardware_dot_graph.draw(output_file_1,prog='circo')
            hardware_dot_graph.draw(output_file_2, prog='circo')
        else:
            hardware_dot_graph.draw(output_file_1,prog='circo')
            hardware_dot_graph.draw(output_file_2,prog='circo')
    except:
        print("could not draw the system_image. Moving on for now. Fix Later.")