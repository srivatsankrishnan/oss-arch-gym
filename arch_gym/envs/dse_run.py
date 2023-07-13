import os

#Add dse_template to the path to allow calling dse
import sys
sys.path.append('../../sims/CFU-Playground/proj/dse_template')
import dse_framework

#Get params from file
file = open('CFU_log', 'r')
action = file.read().split()

i = 0
for a in action:
    action[i] = int(a)
    i += 1

file.close()


#dse_module = __import__('...sims.CFU-Playground.proj.dse_template.dse_framework')
#from ...sims.CFU-Playground.proj.dse_template.dse_framework import dse
#dse_module = importlib.import_module('...sims.CFU-Playground.proj.dse_template.dse_framework', package='oss_arch_gym')

#Move into dse_template directory to ensure smooth running
os.chdir('../../sims/CFU-Playground/proj/dse_template')

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