import os

#Add dse_template to the path to allow calling dse
import sys
sys.path.append('../../sims/CFU-Playground/CFU-Playground/proj/dse_template')
import dse_framework

#Get params from file
file = open('CFU_log', 'r')

action = file.read().split(',')
file.close()

#Move into CFU-Playground symbiflow directory to update the F4PGA install directory 
os.chdir('../../sims/CFU-Playground/CFU-Playground/env/symbiflow')
os.environ["F4PGA_INSTALL_DIR"] = os.getcwd()

#Move to dse_framework directory
os.chdir('../../proj/dse_template')

cycles, cells = dse_framework.dse(
    "mcycle",
    bool(action[0]),
    bool(action[1]),
    int(action[2]),
    bool(action[3]),
    int(action[4]),
    bool(action[5]),
    action[6],
    bool(action[7]),
    bool(action[8]),
    bool(action[9]),
    action[10]
)

#Back into envs directory to write results
os.chdir('../../../../../arch_gym/envs')
file = open('CFU_log', 'w')
file.write(str(cycles) + ' ' + str(cells))
file.close()
