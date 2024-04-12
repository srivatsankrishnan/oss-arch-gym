# MUST BE HYPHENATED
SYSTEM_KNOBS = {
    'scheduling-policy': ({"LIFO", "FIFO"}, 'N/A'),
    'preferred-dataset-splits': ({2, 4, 8, 16}, 'N/A'),
    'collective-optimization': ({"baseline", "localBWAware"}, 'N/A')
}

# MUST BE HYPHENATED
NETWORK_KNOBS = {
    'bandwidth': ({100, 200, 300, 400, 500}, 'FALSE')
}

# MUST USE UNDERSCORES
WORKLOAD_KNOBS = {
    'dp': ({1, 2, 4, 8, 6, 32, 64, 128, 256, 512}, 'N/A'),
    'pp': ({1, 2, 4}, 'N/A'),
    'sp': ({1, 2, 4, 8, 6, 32, 64, 128, 256, 512}, 'N/A'),
	'weight_sharded': ((0, 1, 1), 'N/A')
}

DERIVED_KNOBS = ["network bandwidth-links"]

CONSTRAINTS = ["product network npus-count == num workload num_npus", "mult workload dp workload sp workload pp <= num workload num_npus"]

# DESIGN SPACE = 96 million