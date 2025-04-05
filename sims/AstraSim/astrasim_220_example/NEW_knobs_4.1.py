# MUST BE HYPHENATED
SYSTEM_KNOBS = {
    'scheduling-policy': ({"LIFO", "FIFO"}, 'N/A'),
    'preferred-dataset-splits': ({2, 4, 8, 16}, 'N/A'),
    'all-reduce-implementation': ({"ring", "direct", "halvingDoubling", "doubleBinaryTree"}, 'FALSE'),
    'all-gather-implementation': ({"ring", "direct", "halvingDoubling", "doubleBinaryTree"}, 'FALSE'),
    'reduce-scatter-implementation': ({"ring", "direct", "halvingDoubling", "doubleBinaryTree"}, 'FALSE'),
    'all-to-all-implementation': ({"ring", "direct", "halvingDoubling", "doubleBinaryTree"}, 'FALSE'),
    'collective-optimization': ({"baseline", "localBWAware"}, 'N/A')
}

# MUST BE HYPHENATED
NETWORK_KNOBS = {
    'topology': ({"Ring", "Switch", "FullyConnected"}, 'FALSE'),
    'npus-count': ({4, 8, 16}, 'FALSE'),
    'bandwidth': ({ 50, 100, 150, 200, 250, 300, 350, 400, 450, 500 }, 'FALSE')
}

# MUST USE UNDERSCORES
WORKLOAD_KNOBS = {
    'dp': ({1, 2, 4, 8, 16, 32, 64, 128, 256, 512}, 'N/A'),
    'pp': ({1, 2, 4}, 'N/A'),
    'sp': ({1, 2, 4, 8, 16, 32, 64, 128, 256, 512}, 'N/A'),
	'weight_sharded': ((0, 1, 1), 'N/A')
}

DERIVED_KNOBS = ["network bandwidth divided"]

CONSTRAINTS = ["product network npus-count == num workload num_npus", "mult workload dp workload sp workload pp <= num workload num_npus"]