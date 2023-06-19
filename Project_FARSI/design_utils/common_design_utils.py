#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.


from settings import config
# ------------------------------
# Functionality:
#   this function helps us to convert energy to power by slicing the energy collected in time to
#   smaller phases (PCP) and calculate the power
#   PCP: power collection period
# ------------------------------
def slice_phases_with_PWP(sorted_phase_latency_dict):
    lower_bound_idx, upper_bound_idx = 0, 0
    phase_bounds_list = []
    budget_before_next_collection = config.PCP
    for phase, latency in sorted_phase_latency_dict.items():
        if latency == 0:
            continue
        if latency > budget_before_next_collection:
            budget_before_next_collection = config.PCP
            phase_bounds_list.append(
                (lower_bound_idx, upper_bound_idx + 1))  # we increment by 1, cause this is used as the upper bound
            lower_bound_idx = upper_bound_idx + 1
        else:
            budget_before_next_collection -= latency
        upper_bound_idx += 1

    # add whatever wasn't included at the end
    if not phase_bounds_list:
        phase_bounds_list.append((0, len(list(sorted_phase_latency_dict.values()))))
    elif not phase_bounds_list[-1][1] == len(list(sorted_phase_latency_dict.values())):
        phase_bounds_list.append((phase_bounds_list[-1][1], len(list(sorted_phase_latency_dict.values()))))
    return phase_bounds_list