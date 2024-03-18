SYSTEM_KNOBS = {
    # 'scheduling-policy': ({'FIFO', 'LIFO'}, 'N/A'),
    # "active-chunks-per-dimension": ((1, 32, 1), 'N/A'),       # int type, range [1, 32], default value=1
    # "preferred-dataset-splits": ((1, 32, 1), 'N/A'),        # int type, range [16, 1024], default value=64
    # 'collective-optimization': ({'localBWAware', 'baseline'}, 'N/A'),
    # 'intra-dimension-scheduling': ({'FIFO', 'SCF'}, 'N/A'),
    # 'inter-dimension-scheduling': ({'themis', 'baseline'}, 'N/A'),
    # 'all-reduce-implementation': ({"ring", "direct", "oneRing", "oneDirect", "hierarchicalRing", "doubleBinaryTree"}, 'FALSE')
}

NETWORK_KNOBS = {
    # 'topology': ({"Ring", "Switch", "FullyConnected"}, 'FALSE'),
    'dimensions-count': ({3}, 'N/A')
}

WORKLOAD_KNOBS = {
    # 'num_npus': ((64, 64, 1), 'N/A'),
	'dp': ({1, 2, 4, 8}, 'N/A'),
    'sp': ({1, 2, 4, 8}, 'N/A'),
    'pp': ({1, 2, 4, 8}, 'N/A'),
	'weight_sharded': ((0, 1, 1), 'N/A')
}

# dp * sp * pp <= num_npus: leads to node.is = 4 and not 5 error failure

CONSTRAINTS = ["product network npus-count == num workload num_npus", "mult workload dp workload sp workload pp <= num workload num_npus"]