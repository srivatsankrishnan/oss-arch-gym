#!/usr/bin/env python3

import os
import sys

from simulate_timeloop import simulate_timeloop

from absl import flags
from absl import app

# get the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

flags.DEFINE_string('script_dir', str(BASE_DIR) + "/script", 'Path to the script directory')
flags.DEFINE_string('output_dir', str(BASE_DIR) + "/output", 'Path to the output directory')
flags.DEFINE_string('arch_dir', str(BASE_DIR) + "/arch", 'Path to the architecture directory')
flags.DEFINE_string('mapper_dir', str(BASE_DIR) + "/mapper", 'Path to the mapper directory')
flags.DEFINE_string('workload_dir', str(BASE_DIR) + "/layer_shapes/AlexNet", 'Path to the workload directory')
flags.DEFINE_string('runtime', "docker", 'Path to the architecture parameters')
FLAGS = flags.FLAGS


def timeloop_4_archgym_example():
    # Run Timeloop for baseline numbers
    energy, area, cycles = simulate_timeloop(script_dir=FLAGS.script_dir,
                                             output_dir=FLAGS.output_dir,
                                             arch_dir=FLAGS.arch_dir,
                                             mapper_dir=FLAGS.mapper_dir,
                                             workload_dir=FLAGS.workload_dir,
                                             runtime=FLAGS.runtime)

    if FLAGS.runtime == "docker":
        from timeloop_wrapper import TimeloopWrapper
    elif FLAGS.runtime == "singularity":
        from timeloop_wrapper_singularity import TimeloopWrapper

    # Get arch params
    timeloop = TimeloopWrapper()
    arch_params = timeloop.get_arch_param_template()

    # Change number of PEs
    arch_params['NUM_PEs'] = 'PE[0..237]'

    # Run Timeloop
    energy, area, cycles = simulate_timeloop(script_dir=FLAGS.script_dir,
                                             output_dir=FLAGS.output_dir,
                                             arch_dir=FLAGS.arch_dir,
                                             mapper_dir=FLAGS.mapper_dir,
                                             workload_dir=FLAGS.workload_dir,
                                             arch_params=arch_params,
                                             runtime=FLAGS.runtime)

    print("Simulation complete!")


def main(_):
    timeloop_4_archgym_example()


if __name__ == '__main__':
    app.run(main)

# sh build.sh
# sudo docker run -it --entrypoint=/bin/bash timeloop_4_archgym
# python ./arch-gym/sims/Timeloop/hello_timeloop_4_archgym.py --script_dir=/home/workspace/src/timeloop-examples/workspace/final-project/example_designs/eyeriss_like/arch-gym/sims/Timeloop/script --output_dir=/home/workspace/src/timeloop-examples/workspace/final-project/example_designs/eyeriss_like/arch-gym/sims/Timeloop/output --arch_dir=/home/workspace/src/timeloop-examples/workspace/final-project/example_designs/eyeriss_like/arch-gym/sims/Timeloop/arch --workload_dir=/home/workspace/src/timeloop-examples/workspace/final-project/layer_shapes/AlexNet
