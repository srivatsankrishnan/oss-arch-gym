import os

#Add dse_template to the path to allow calling dse
import sys
sys.path.append('../../sims/CFU-Playground/proj/dse_template')
import dse_framework

#Get params from file
file = open('CFU_log', 'r')

action = file.read().split(',')
file.close()

#Move into CFU-Playground symbiflow directory to update the F4PGA install directory 
os.chdir('../../sims/CFU-Playground/env/symbiflow')
os.environ["F4PGA_INSTALL_DIR"] = os.getcwd()

#Move to dse_framework directory
os.chdir('../../proj/dse_template')

cycles, cells = dse_framework.dse(
    action[0],
    int(action[1]),
    int(action[2]),
    int(action[3]),
    int(action[4]),
    int(action[5]),
    int(action[6]),
    action[7],
    int(action[8]),
    int(action[9]),
    int(action[10]),
    action[11]
)

#Back into envs directory to write results
os.chdir('../../../../arch_gym/envs')
file = open('CFU_log', 'w')
file.write(str(cycles) + ' ' + str(cells))
file.close()
