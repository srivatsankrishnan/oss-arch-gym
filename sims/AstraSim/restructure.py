import os
import csv

def read_csv_column(file_path):
    """
    Reads a CSV file and returns a list of the values in the first column.
    """
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        column_data = [row if row else "NA" for row in reader]
    return column_data

def merge_columns(file1_path, file2_path, output_file_path):
    column1_data = read_csv_column(file1_path)
    column2_data = read_csv_column(file2_path)

    # Ensure both columns have the same length
    if len(column1_data) != len(column2_data):
        raise ValueError("The columns must have the same length.")
    # Merge the columns into a list of dictionaries
    merged_data = [{'x': column1_data[i], 'y': column2_data[i]} for i in range(len(column1_data))]

    # Write the merged data to a new CSV file
    with open(output_file_path, 'w', newline='') as outfile:
        fieldnames = ['x', 'y']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_data)

if __name__ == "__main__":
    log_path = "random_walker_logs/latency/resnet18_num_steps_6_num_episodes_1/"
    file1_path = os.path.join(log_path, "actions.csv")
    file2_path = os.path.join(log_path, "observations.csv") 
    output_file_path = os.path.join(log_path, "merged.csv")
    merge_columns(file1_path, file2_path, output_file_path)
