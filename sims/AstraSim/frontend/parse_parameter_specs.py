import csv
import os

def parse_csv_to_knobs(input_csv):
    system_knobs = {}
    network_knobs = {}
    workload_knobs = {}

    current_knob_dict = None

    with open(input_csv, 'r') as csvfile:
        csvreader = csv.reader(csvfile)

        for row in csvreader:
            if row and row[0] == 'System':
                current_knob_dict = system_knobs
            elif row and row[0] == 'Network':
                current_knob_dict = network_knobs
            elif row and row[0] == 'Workload':
                current_knob_dict = workload_knobs
            elif row[0] == 'Parameter':
                next(csvreader)
            elif current_knob_dict is not None:
                parameter = row[0]
                # row[1] is a set or a tuple. Tuple is a range, and set is a set of possible values
                range = row[1]
                samePerDimension = row[2]
                current_knob_dict[parameter] = (eval(range), samePerDimension)

    return system_knobs, network_knobs

# Example usage
input_csv_file = 'parameter_specs.csv'

settings_file_path = os.path.realpath(__file__)
settings_dir_path = os.path.dirname(settings_file_path)
proj_root_path = os.path.abspath(settings_dir_path)

parameter_specs = os.path.join(proj_root_path, input_csv_file)


SYSTEM_KNOBS, NETWORK_KNOBS = parse_csv_to_knobs(parameter_specs)

print("SYSTEM_KNOBS:")
print(SYSTEM_KNOBS)
print("\nNETWORK_KNOBS:")
print(NETWORK_KNOBS)