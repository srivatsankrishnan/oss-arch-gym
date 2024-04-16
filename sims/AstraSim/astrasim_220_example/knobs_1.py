# MUST BE HYPHENATED
SYSTEM_KNOBS = {
}

# MUST BE HYPHENATED
NETWORK_KNOBS = {
}

# MUST USE UNDERSCORES
WORKLOAD_KNOBS = {
    'dp': ({1, 2, 4, 8, 16, 32, 64, 128, 256, 512}, 'N/A'),
    'pp': ({1, 2, 4}, 'N/A'),
    'sp': ({1, 2, 4, 8, 16, 32, 64, 128, 256, 512}, 'N/A'),
	'weight_sharded': ((0, 1, 1), 'N/A')
}

DERIVED_KNOBS = []

CONSTRAINTS = ["product network npus-count == num workload num_npus", "mult workload dp workload sp workload pp <= num workload num_npus"]

# DESIGN SPACE = 600