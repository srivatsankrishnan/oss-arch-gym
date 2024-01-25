SYSTEM_KNOBS = {
    'scheduling-policy': ({'FIFO', 'LIFO'}, 'N/A')
    # # "active-chunks-per-dimension": ((1, 32, 1), 'N/A'),       # int type, range [1, 32], default value=1
    # # "preferred-dataset-splits": ((1, 32, 1), 'N/A'),        # int type, range [16, 1024], default value=64
    # 'collective-optimization': ({'localBWAware', 'baseline'}, 'N/A')
    # 'intra-dimension-scheduling': ({'FIFO', 'SCF'}, 'N/A'),
    # 'inter-dimension-scheduling': ({'themis', 'baseline'}, 'N/A')
    # 'all-reduce-implementation': ({"ring", "direct", "oneRing", "oneDirect", "hierarchicalRing", "doubleBinaryTree"}, 'FALSE')
}


NETWORK_KNOBS = {
    'topology': ({"Ring", "Switch", "FullyConnected"}, 'TRUE')
    # 'dimensions-count': ({1, 2, 3, 4}, 'N/A')
}

WORKLOAD_KNOBS = {
    'num_npus': ((1, 64, 1), 'N/A'),
	'dp': ((1, 8, 1), 'N/A'),
	'weight_sharded': ((1, 4, 1), 'N/A')
}

CONSTRAINTS = ['product network links-count >= num network dimensions-count']