#!/usr/bin/env python3

# from sims.Timeloop.timeloop_wrapper import TimeloopWrapper
import os
import sys

os.sys.path.insert(0, os.path.abspath('../../'))


from absl import flags
from absl import app
from absl import logging


def simulate_timeloop(script_dir=None, output_dir=None, arch_dir=None, mapper_dir=None, workload_dir=None,
                      arch_params=None, runtime="docker"):
    if runtime == "docker":
        from sims.Timeloop.timeloop_wrapper import TimeloopWrapper

    elif runtime == "singularity":
        # Run in singularity container.
        from sims.Timeloop.timeloop_wrapper_singularity import TimeloopWrapper
    else:
        raise ValueError("Runtime should be either docker or singularity")

    timeloop = TimeloopWrapper(script_dir, output_dir, arch_dir, mapper_dir, workload_dir)
    if arch_params is not None:
        timeloop.update_arch(arch_params)
    energy, area, cycles = timeloop.launch_timeloop()
    print("Energy: " + str(energy))
    print("Area:   " + str(area))
    print("Cycles: " + str(cycles))
    return energy, area, cycles


def simulate_timeloop_batch(script_dirs, output_dirs, arch_dirs, mapper_dir, workload_dir, multi_params):
    energy, area, cycles = [], [], []
    for sd, od, ad, ap in zip(script_dirs, output_dirs, arch_dirs, multi_params):
        e, a, c = simulate_timeloop(sd, od, ad, mapper_dir, workload_dir, ap)
        energy.append(e)
        area.append(a)
        cycles.append(c)

    return energy, area, cycles


def main(_):
    simulate_timeloop(FLAGS.script_dir, FLAGS.output_dir, FLAGS.arch_dir, FLAGS.mapper_dir, FLAGS.workload_dir)


if __name__ == '__main__':
    app.run(main)

# python simulate_timeloop.py --script "/home/sprakash/Documents/Repos/Timeloop/script" --output "/home/sprakash/Documents/Repos/Timeloop/output3" --arch "/home/sprakash/Documents/Repos/Timeloop/arch" --mapper "/home/sprakash/Documents/Repos/Timeloop/mapper" --workload "/home/sprakash/Documents/Repos/Timeloop/layer_shapes/AlexNet"
