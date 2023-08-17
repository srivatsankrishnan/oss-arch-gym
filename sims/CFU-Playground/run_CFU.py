import os

#Add dse_template to the path to allow calling dse
import sys
sys.path.append('./CFU-Playground/proj/dse_template')
import dse_framework

#Get params from file
file = open('CFU_log', 'r')

action = file.read().split(',')
file.close()

#Move into CFU-Playground symbiflow directory to update the F4PGA install directory 
os.chdir('./CFU-Playground/env/symbiflow')
os.environ["F4PGA_INSTALL_DIR"] = os.getcwd()

#Move to dse_framework directory
os.chdir('../../proj/dse_template')

cycles, cells = dse_framework.dse(
    "mcycle",
    False if action[0] == '0' else True,
    False if action[1] == '0' else True,
    int(action[2]),
    False if action[3] == '0' else True,
    int(action[4]),
    False if action[5] == '0' else True,
    action[6],
    False if action[7] == '0' else True,
    False if action[8] == '0' else True,
    False if action[9] == '0' else True,
    action[10],
    action[11]
)

#Back into envs directory to write results
os.chdir('../../..')
file = open('CFU_log', 'w')
file.write(str(cycles) + ' ' + str(cells))
file.close()
