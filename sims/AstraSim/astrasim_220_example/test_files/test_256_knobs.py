# MUST BE HYPHENATED
SYSTEM_KNOBS = {
    'scheduling-policy': ({"LIFO", "FIFO"}, 'N/A'),
    'preferred-dataset-splits': ({2, 4, 8, 16}, 'N/A'),
    'all-reduce-implementation': ({"ring", "direct", "halvingDoubling", "doubleBinaryTree"}, 'FALSE'),
    'collective-optimization': ({"baseline", "localBWAware"}, 'N/A')
}

# MUST BE HYPHENATED
NETWORK_KNOBS = {
    'topology': ({"Ring", "Switch", "FullyConnected"}, 'FALSE'),
    'npus-count': ({2, 4, 8}, 'FALSE'),
}

# MUST USE UNDERSCORES
WORKLOAD_KNOBS = {
}

DERIVED_KNOBS = ["network bandwidth", "system implementations"]

CONSTRAINTS = ["product network npus-count == num workload num_npus", "mult workload dp workload sp workload pp <= num workload num_npus"]