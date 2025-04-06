# MUST BE HYPHENATED
SYSTEM_KNOBS = {
}

# MUST BE HYPHENATED
NETWORK_KNOBS = {
    'topology': ({"Ring", "Switch", "FullyConnected"}, 'FALSE'),
    'npus-count': ({4, 8, 16}, 'FALSE'),
    'bandwidth': ({ 50, 100, 150, 200, 250, 300, 350, 400, 450, 500 }, 'FALSE')
}

# MUST USE UNDERSCORES
WORKLOAD_KNOBS = {
}

DERIVED_KNOBS = ["network bandwidth divided", "system all-reduce-implementation", "system all-gather-implementation", "system reduce-scatter-implementation", "system all-to-all-implementation"]

CONSTRAINTS = ["product network npus-count == num workload num_npus", "mult workload dp workload sp workload pp <= num workload num_npus"]