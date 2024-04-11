SYSTEM_KNOBS = {
    'scheduling-policy': ({'FIFO', 'LIFO'}, 'N/A'),
    'collective-optimization': ({'localBWAware', 'baseline'}, 'N/A'),
    'preferred-dataset-splits': ({1, 2, 4, 8, 16, 32, 64}, 'N/A'),
}

NETWORK_KNOBS = {
    'topology': ({"Ring", "Switch", "FullyConnected"}, 'FALSE')
}

WORKLOAD_KNOBS = {
	'dp': ({1, 2, 4, 8}, 'N/A'),
    'sp': ({1, 2, 4, 8}, 'N/A'),
    'pp': ({1, 2, 4, 8}, 'N/A'),
	'weight_sharded': ((0, 1, 1), 'N/A')
}

DERIVED_KNOBS = ["network bandwidth", "system all-reduce-implementation", "system all-gather-implementation",
                 "system reduce-scatter-implementation", "system all-to-all-implementation"]

CONSTRAINTS = ["product network npus-count == num workload num_npus", "mult workload dp workload sp workload pp <= num workload num_npus"]