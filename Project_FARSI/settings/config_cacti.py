import os

# get the base path of arch-gym
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


cact_bin_addr = os.path.join(base_path, "Project_FARSI/cacti_for_FARSI/cacti")

print(cact_bin_addr, os.path.exists(cact_bin_addr))

cacti_param_addr = os.path.join(base_path, "Project_FARSI/cacti_for_FARSI/farsi_gen.cfg")

print(cacti_param_addr, os.path.exists(cacti_param_addr))

cacti_data_log_file = os.path.join(base_path, "Project_FARSI/cacti_for_FARSI/data_log.csv")

print(cacti_data_log_file, os.path.exists(cacti_data_log_file))

