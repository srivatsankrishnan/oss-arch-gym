import csv
import os

# set path to /sims/AstraSim/frontend
settings_file_path = os.path.realpath(__file__)
settings_dir_path = os.path.dirname(settings_file_path)


def parse_csv_to_knobs(input_csv):
    system_knobs = {}
    network_knobs = {}
    workload_knobs = {}
    all_constraints = []

    current_knob_dict = None


    with open(input_csv, 'r') as csvfile:
        csvreader = csv.reader(csvfile)

        for row in csvreader:
            if not row[0] or row[0] == 'Parameter':
                continue
            elif row and row[0] == 'System':
                current_knob_dict = system_knobs
            elif row and row[0] == 'Network':
                current_knob_dict = network_knobs
            elif row and row[0] == 'Workload':
                current_knob_dict = workload_knobs

            elif current_knob_dict is not None:
                parameter = row[0]
                # row[1] is a set or a tuple. Tuple is a range, and set is a set of possible values
                range_ = row[1]
                samePerDimension = row[2]
                constraints = row[3]

                print("cols: ")
                print("range: ", range_)
                print("samePerDimension", samePerDimension)
                print("constraints", constraints)
                # print(parameter, eval(range_), samePerDimension)
                current_knob_dict[parameter] = [eval(range_), samePerDimension]

                """
                parse constraints
                constraint-type arg1-dict arg1 arg2-dict arg2 operator arg3-dict arg3 arg4-dict arg4

                """
                if constraints:
                    all_constraints.append(constraints)

    return system_knobs, network_knobs, workload_knobs, all_constraints


input_csv_file = 'parameter_specs.csv'

settings_file_path = os.path.realpath(__file__)
settings_dir_path = os.path.dirname(settings_file_path)
proj_root_path = os.path.abspath(settings_dir_path)

parameter_specs = os.path.join(proj_root_path, input_csv_file)

SYSTEM_KNOBS, NETWORK_KNOBS, WORKLOAD_KNOBS, all_constraints = parse_csv_to_knobs(parameter_specs)

# write system_knobs and network_knobs to a separate python file, as dicts
with open(settings_dir_path + '/parameter_knobs.py', 'w') as knobs_file:
    knobs_file.write("SYSTEM_KNOBS = ")
    knobs_file.write(str(SYSTEM_KNOBS))
    knobs_file.write("\n")
    knobs_file.write("NETWORK_KNOBS = ")
    knobs_file.write(str(NETWORK_KNOBS))
    knobs_file.write("\n")
    knobs_file.write("WORKLOAD_KNOBS = ")
    knobs_file.write(str(WORKLOAD_KNOBS))
    knobs_file.write("\n")
    knobs_file.write("CONSTRAINTS = ")
    knobs_file.write(str(all_constraints))

print("SYSTEM_KNOBS:")
print(SYSTEM_KNOBS)
print("\nNETWORK_KNOBS:")
print(NETWORK_KNOBS)
print("\nWORKLOAD_KNOBS:")
print(WORKLOAD_KNOBS)
print("\nALL_CONSTRAINTS:")
print(all_constraints)