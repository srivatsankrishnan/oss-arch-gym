# MUST BE HYPHENATED
SYSTEM_KNOBS = {
}

# MUST BE HYPHENATED
NETWORK_KNOBS = {
    'topology': ({"Ring", "Switch", "FullyConnected"}, 'FALSE'),
    'npus-count': ({4, 8, 16}, 'FALSE')
}

# MUST USE UNDERSCORES
WORKLOAD_KNOBS = {
    'dp': ({1, 2, 4, 8, 16, 32, 64, 128, 256, 512}, 'N/A'),
    'pp': ({1, 2, 4}, 'N/A'),
    'sp': ({1, 2, 4, 8, 16, 32, 64, 128, 256, 512}, 'N/A'),
	'weight_sharded': ((0, 1, 1), 'N/A')
}

DERIVED_KNOBS = ["network bandwidth", "system all-reduce-implementation", "system all-gather-implementation", "system reduce-scatter-implementation", "system all-to-all-implementation"]

CONSTRAINTS = ["product network npus-count == num workload num_npus", "mult workload dp workload sp workload pp <= num workload num_npus"]

# DESIGN SPACE = 4 million (3,936,600)