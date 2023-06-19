#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

from SIM_utils.components.perf_sim import *
from SIM_utils.components.pow_sim import *
#from OSSIM_utils.components.pow_knob_sim import *
from design_utils.design import *
from settings import config

# This module is our top level simulator containing all simulators (perf, and pow simulator)
class OSASimulator:
    def __init__(self, dp, database, pk_dp=""):
        self.time_elapsed = 0  # time elapsed from the beginning of the simulation
        self.dp = dp  # design point to simulate
        self.perf_sim = PerformanceSimulator(self.dp)  # performance simulator instance
        self.pow_sim = PowerSimulator(self.dp)  # power simulator instance

        self.database = database
        if config.simulation_method == "power_knobs":
            self.pk_dp = pk_dp
            #self.knob_change_sim = PowerKnobSimulator(self.dp, self.pk_dp, self.database)
        self.completion_time = -1   # time passed for the simulation to complete
        self.program_status = "idle"
        self.cur_tick_time = self.next_tick_time = 0  # current tick time

    # ------------------------------
    # Functionality:
    #       whether the simulation should terminate
    # ------------------------------
    def terminate(self, program_status):
        if config.termination_mode == "workload_completion":
            return program_status == "done"
        elif config.termination_mode == "time_budget_reahced":
            return self.time_elapsed >= config.time_budge
        else:
            return False

    # ------------------------------
    # Functionality:
    #   ticking the simulation. Note that the tick time varies depending on what is (dynamically) happening in the
    #   system
    # ------------------------------
    def tick(self):
        self.cur_tick_time = self.next_tick_time

    # ------------------------------
    # Functionality
    #   progress the simulation for clock_time forward
    # ------------------------------
    def step(self, clock_time):
        self.next_tick_time, self.program_status = self.perf_sim.simulate(clock_time)

    # ------------------------------
    # Functionality:
    #   simulation
    # ------------------------------
    def simulate(self):
        blah = time.time()
        while not self.terminate(self.program_status):
            self.tick()
            self.step(self.cur_tick_time)

        if config.use_cacti:
            self.dp.correct_power_area_with_cacti(self.database)

        # collect all the stats upon completion of simulation
        self.dp.collect_dp_stats(self.database)

        if config.simulation_method == "power_knobs":
            self.knob_change_sim.launch()

        self.completion_time = self.next_tick_time
        self.dp.set_serial_design_time(self.perf_sim.serial_latency)
        self.dp.set_par_speedup(self.perf_sim.serial_latency/self.completion_time)
        self.dp.set_simulation_time_analytical_portion(self.perf_sim.task_update_time + self.perf_sim.phase_interval_calc_time)
        self.dp.set_simulation_time_phase_driven_portion(self.perf_sim.phase_scheduling_time)
        self.dp.set_simulation_time_phase_calculation_portion(self.perf_sim.phase_interval_calc_time)
        self.dp.set_simulation_time_task_update_portion(self.perf_sim.task_update_time)
        self.dp.set_simulation_time_phase_scheduling_portion(self.perf_sim.phase_scheduling_time)

        return self.dp