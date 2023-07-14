import os

#Add dse_template to the path to allow calling dse
import sys
sys.path.append('../../sims/CFU-Playground/proj/dse_template')
import dse_framework

#Get params from file
file = open('CFU_log', 'r')
action = file.read().split()

action = [int(a) for a in action]

file.close()

#Move into CFU-Playground symbiflow directory to update the F4PGA install directory 
os.chdir('../../sims/CFU-Playground/env/symbiflow')
os.environ["F4PGA_INSTALL_DIR"] = os.getcwd()

#Move to dse_framework directory
os.chdir('../../proj/dse_template')

Branch_predict_types = ['None', 'Static', 'Dynamic', 'Dynamic Target']

cycles, cells = dse_framework.dse(
    "mcycle",
    True if action[0] == 1 else False,           # Bypass
    #True if action[1] == 2 else False           # CFU_enable 
    False,           # CFU_enable (currently set to false)
    0 if action[2] == 0 else (1<<(4+action[2])), # Data cache size
    True if action[3] == 1 else False,           # Hardware Divider
    0 if action[4] == 0 else (1<<(4+action[4])), # Instruction cache size
    True if action[5] == 1 else False,           # Hardware Multiplier
    Branch_predict_types[action[6]],             # Branch predictor
    True if action[7] == 1 else False,           # Safe mode
    True if action[8] == 1 else False,           # Single Cycle Shifter
    True if action[9] == 1 else False,           # Single Cycle Multiplier
    "digilent_arty"
)

#Back into envs directory to write results
os.chdir('../../../../arch_gym/envs')

file = open('CFU_log', 'w')
file.write(str(cycles) + ' ' + str(cells))
file.close()