#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

from design_utils.design import  *
from functools import reduce


# This class is the performance simulator of FARSI
class PerformanceSimulator:
    def __init__(self, sim_design):
        self.design = sim_design  # design to simulate
        self.scheduled_kernels = []   # kernels already scheduled
        self.driver_waiting_queue = []   # kernels whose trigger condition is met but can not run for various reasons
        self.completed_kernels_for_memory_sizing = []   # kernels already completed
        # List of all the kernels that are not scheduled yet (to be launched)
        self.yet_to_schedule_kernels = self.design.get_kernels()[:]  # kernels to be scheduled
        self.all_kernels = self.yet_to_schedule_kernels[:]
        self.task_token_queue = []
        self.old_clock_time = self.clock_time = 0
        self.program_status = "idle"  # specifying the status of the program at the current tick
        self.phase_num = -1
        self.krnl_latency_if_run_in_isolation = {}
        self.serial_latency = 0
        self.workload_time_if_each_kernel_run_serially()
        self.phase_interval_calc_time = 0
        self.phase_scheduling_time = 0
        self.task_update_time = 0


    def workload_time_if_each_kernel_run_serially(self):
        self.serial_latency = 0
        for krnl in self.yet_to_schedule_kernels:
            self.krnl_latency_if_run_in_isolation[krnl] = krnl.get_latency_if_krnel_run_in_isolation()

        for krnl, latency in self.krnl_latency_if_run_in_isolation.items():
            self.serial_latency += latency


    def reset_perf_sim(self):
        self.scheduled_kernels = []
        self.completed_kernels_for_memory_sizing = []
        # List of all the kernels that are not scheduled yet (to be launched)
        self.yet_to_schedule_kernels = self.design.get_kernels()[:]
        self.old_clock_time = self.clock_time = 0
        self.program_status = "idle"  # specifying the status of the program at the current tick
        self.phase_num = -1


    # ------------------------------
    # Functionality:
    #   tick the simulator clock_time forward
    # ------------------------------
    def tick(self, clock_time):
        self.clock_time = clock_time

    # ------------------------------
    # Functionality:
    #   find the next kernel to be scheduled time
    # ------------------------------
    def next_kernel_to_be_scheduled_time(self):
        timely_sorted_kernels = sorted(self.yet_to_schedule_kernels, key=lambda kernel: kernel.get_schedule().starting_time)
        return timely_sorted_kernels[0].get_schedule().starting_time

    # ------------------------------
    # Functionality:
    #   convert the task to kernel
    # ------------------------------
    def get_kernel_from_task(self, task):
        for kernel in self.design.get_kernels()[:]:
            if kernel.get_task() == task:
                return kernel
        raise Exception("kernel associated with task with name" + task.name + " is not found")

    # ------------------------------
    # Functionality:
    #   find the completion time of kernel that will be done the fastest
    # ------------------------------
    def next_kernel_to_be_completed_time(self):
        comp_time_list = []  # contains completion time of the running kernels
        for kernel in self.scheduled_kernels:
            comp_time_list.append(kernel.calc_kernel_completion_time())
        return min(comp_time_list) + self.clock_time
        #else:
        #    return self.clock_time
    """
    # ------------------------------
    # Functionality:
    #   all the dependencies of a kernel are done or no?
    # ------------------------------
    def kernel_parents_all_done(self, kernel):
        kernel_s_task = kernel.get_task()
        parents_s_task = self.design.get_hardware_graph().get_task_graph().get_task_s_parents(kernel_s_task)
        completed_tasks = [kernel.get_task() for kernel in self.completed_kernels_for_memory_sizing]
        for task in parents_s_task:
            if task not in completed_tasks:
                return False
        return True
    """

    def kernel_s_parents_done(self, krnl):
        kernel_s_task = krnl.get_task()
        parents_s_task = self.design.get_hardware_graph().get_task_graph().get_task_s_parents(kernel_s_task)
        for parent in parents_s_task:
            if not (parent, kernel_s_task) in self.task_token_queue:
                return False
        return True


    # launch: Every iteration, we launch the kernel, i.e,
    # we set the operating state appropriately, and size the hardware accordingly
    def kernel_ready_to_be_launched(self, krnl):
        if self.kernel_s_parents_done(krnl) and krnl not in self.scheduled_kernels and not self.krnl_done_iterating(krnl):
            return True
        return False

    def kernel_ready_to_fire(self, krnl):
        if krnl.get_type() == "throughput_based" and krnl.throughput_time_trigger_achieved(self.clock_time):
           return True
        else:
            return False

    def remove_parents_from_token_queue(self, krnl):
        kernel_s_task = krnl.get_task()
        parents_s_task = self.design.get_hardware_graph().get_task_graph().get_task_s_parents(kernel_s_task)
        for parent in parents_s_task:
            self.task_token_queue.remove((parent, kernel_s_task))

    def krnl_done_iterating(self, krnl):
        if krnl.iteration_ctr == -1 or krnl.iteration_ctr > 0:
            return False
        elif krnl.iteration_ctr == 0:
            return True


    # if multiple kernels running on the same PE's driver,
    # only one can access it at a time. Thus, this function only keeps one on the PE.
    def serialize_DMA(self):
        # append waiting kernels and to be sch kernels to the already scheduled kernels
        scheduled_kernels_tmp = []
        for el in self.scheduled_kernels:
            scheduled_kernels_tmp.append(el)
        for kernel in self.driver_waiting_queue:
            scheduled_kernels_tmp.append(kernel)

        PE_blocks_used = []
        scheduled_kernels = []
        driver_waiting_queue = []
        for el in scheduled_kernels_tmp:
            # only for read/write we serialize
            if el.get_operating_state() in ["read", "write", "none"]:
                pe = [blk for blk in el.get_blocks() if blk.type == "pe"][0]
                if pe in PE_blocks_used:
                    driver_waiting_queue.append(el)
                else:
                    scheduled_kernels.append(el)
                    PE_blocks_used.append(pe)
            else:
                scheduled_kernels.append(el)
        return scheduled_kernels, driver_waiting_queue

    # ------------------------------
    # Functionality:
    #   Finds the kernels that are free to be scheduled (their parents are completed)
    # ------------------------------
    def schedule_kernels_token_based(self):
        for krnl in self.all_kernels:
            if self.kernel_ready_to_be_launched(krnl):
                # launch: Every iteration, we launch the kernel, i.e,
                # we set the operating state appropriately, and size the hardware accordingly
                self.kernel_s_parents_done(krnl)
                self.remove_parents_from_token_queue(krnl)
                self.scheduled_kernels.append(krnl)
                if krnl in self.yet_to_schedule_kernels:
                    self.yet_to_schedule_kernels.remove(krnl)

                # initialize #insts, tick, and kernel progress status
                krnl.launch(self.clock_time)
                # update PE's that host the kernel
                krnl.update_mem_size(1)
                krnl.update_pe_size()
                krnl.update_ic_size()
            elif krnl.status == "in_progress"  and not krnl.get_task().is_task_dummy() and self.kernel_ready_to_fire(krnl):
                if krnl in self.scheduled_kernels:
                    print("a throughput based kernel was scheduled before it met its desired throughput. "
                          "This can cause issues in the models. Fix Later")
                self.scheduled_kernels.append(krnl)

        # filter out kernels based on DMA serialization, i.e., only keep one kernel using the PE's driver.
        if config.DMA_mode == "serialized_read_write":
            self.scheduled_kernels, self.driver_waiting_queue = self.serialize_DMA()

    # ------------------------------
    # Functionality:
    #   Finds the kernels that are free to be scheduled (their parents are completed)
    # ------------------------------
    def schedule_kernels(self):
        if config.scheduling_policy == "FRFS":
            kernels_to_schedule = [kernel_ for kernel_ in self.yet_to_schedule_kernels
                                   if self.kernel_parents_all_done(kernel_)]
        elif config.scheduling_policy == "time_based":
            kernels_to_schedule = [kernel_ for kernel_ in self.yet_to_schedule_kernels
                                   if self.clock_time >= kernel_.get_schedule().starting_time]
        else:
            raise Exception("scheduling policy not supported")

        for kernel in kernels_to_schedule:
            self.scheduled_kernels.append(kernel)
            self.yet_to_schedule_kernels.remove(kernel)
            # initialize #insts, tick, and kernel progress status
            kernel.launch(self.clock_time)
            # update memory size -> allocate memory regions on different mem blocks
            kernel.update_mem_size(1)
            # update pe allocation -> allocate a part of pe quantum for current task
            # (Hadi Note: allocation looks arbitrary and without any meaning though - just to know that something
            # is allocated or it is floating)
            kernel.update_pe_size()
            # empty function!
            kernel.update_ic_size()


    def update_parallel_tasks(self):
        # keep track of krnls that are present per phase
        for krnl in self.scheduled_kernels:
            if krnl.get_task_name() in ["souurce", "siink", "dummy_last"]:
                continue
            if krnl not in self.design.krnl_phase_present.keys():
                self.design.krnl_phase_present[krnl] = []
                self.design.krnl_phase_present_operating_state[krnl] = []
            self.design.krnl_phase_present[krnl].append(self.phase_num)
            self.design.krnl_phase_present_operating_state[krnl].append((self.phase_num, krnl.operating_state))
            self.design.phase_krnl_present[self.phase_num] = self.scheduled_kernels[:]

        """
        scheduled_kernels = self.scheduled_kernels[:]
        for idx, krnl in enumerate(scheduled_kernels):
            if krnl.get_task_name() in ["souurce", "siink"]:
                continue
            if krnl not in self.design.parallel_kernels.keys():
                self.design.parallel_kernels[krnl] = []
            for idx, krnl_2 in enumerate(scheduled_kernels):
                if krnl_2 == krnl:
                    continue
                elif not krnl_2 in self.design.parallel_kernels[krnl]:
                    self.design.parallel_kernels[krnl].append(krnl_2)

        pass
        """
    # ------------------------------
    # Functionality:
    #   update the status of each kernel, this means update
    #   how much work is left for each kernel (that is already schedulued)
    # ------------------------------
    def update_scheduled_kernel_list(self):
        scheduled_kernels = self.scheduled_kernels[:]
        for kernel in scheduled_kernels:
            if kernel.status == "completed":
                self.scheduled_kernels.remove(kernel)
                self.completed_kernels_for_memory_sizing.append(kernel)
                kernel.set_stats()
                for child_task in kernel.get_task().get_children():
                    self.task_token_queue.append((kernel.get_task(), child_task))
                # iterate though parents and check if for each parent, all the children are completed.
                # if so, retract the memory
                all_parent_kernels = [self.get_kernel_from_task(parent_task) for parent_task in
                                      kernel.get_task().get_parents()]
                for parent_kernel in all_parent_kernels:
                    all_children_kernels = [self.get_kernel_from_task(child_task) for child_task in
                                           parent_kernel.get_task().get_children()]

                    if all([child_kernel in self.completed_kernels_for_memory_sizing for child_kernel in all_children_kernels]):
                        parent_kernel.update_mem_size(-1)
                        for child_kernel in all_children_kernels:
                            self.completed_kernels_for_memory_sizing.remove(child_kernel)

            elif kernel.type == "throughput_based" and kernel.throughput_work_achieved():
                #del kernel.data_work_left_to_meet_throughput[kernel.operating_state][0]
                del kernel.firing_work_to_meet_throughput[kernel.operating_state][0]
                self.scheduled_kernels.remove(kernel)


    # ------------------------------
    # Functionality:
    #   iterate through all kernels and step them
    # ------------------------------
    def step_kernels(self):
        # by stepping the kernels, we calculate how much work each kernel has done and how much of their
        # work is left for them
        _ = [kernel_.step(self.time_step_size, self.phase_num) for kernel_ in self.scheduled_kernels]

        # update kernel's status, sets the progress
        _ = [kernel_.update_status(self.time_step_size, self.clock_time) for kernel_ in self.scheduled_kernels]

    # ------------------------------
    # Functionality:
    #   update the status of the program, i.e., whether it's done or still in progress
    # ------------------------------
    def update_program_status(self):
        if len(self.scheduled_kernels) == 0 and len(self.yet_to_schedule_kernels) == 0:
            self.program_status = "done"
        elif len(self.scheduled_kernels) == 0:
            self.program_status = "idle"  # nothing scheduled yet
        elif len(self.yet_to_schedule_kernels) == 0:
            self.program_status = "all_kernels_scheduled"
        else:
            self.program_status = "in_progress"

    def next_throughput_trigger_time(self):
        throughput_achieved_time_list = []
        for krnl in self.all_kernels:
            if krnl.get_type() == "throughput_based" and krnl.status == "in_progress" \
                    and not krnl.get_task().is_task_dummy() and not krnl.operating_state == "execute":
                throughput_achieved_time_list.extend(krnl.firing_time_to_meet_throughput[krnl.operating_state])

        throughput_achieved_time_list_filtered = [el for el in throughput_achieved_time_list if el> self.clock_time]
        #time_sorted = sorted(throughput_achieved_time_list_filtered)
        return throughput_achieved_time_list_filtered

    def any_throughput_based_kernel(self):
        for krnl in self.all_kernels:
            if krnl.get_type()  == "throughput_based":
                return True
        return False

    # ------------------------------
    # Functionality:
    #   find the next tick time
    # ------------------------------
    def calc_new_tick_position(self):
        if config.scheduling_policy == "FRFS":
            new_clock_list = []
            if len(self.scheduled_kernels) > 0:
                new_clock_list.append(self.next_kernel_to_be_completed_time())

            if self.any_throughput_based_kernel():
                trigger_time = self.next_throughput_trigger_time()
                if len(trigger_time) > 0:
                    new_clock_list.append(min(trigger_time))

            if len(new_clock_list) == 0:
                return self.clock_time
            else:
                return min(new_clock_list)
            #new_clock = max(new_clock, min(throughput_achieved_time))
        """
        elif self.program_status == "in_progress":
            if self.program_status == "in_progress":
                if config.scheudling_policy == "time_based":
                    new_clock = min(self.next_kernel_to_be_scheduled_time(), self.next_kernel_to_be_completed_time())
            elif self.program_status == "all_kernels_scheduled":
                new_clock = self.next_kernel_to_be_completed_time()
            elif self.program_status == "idle":
                new_clock = self.next_kernel_to_be_scheduled_time()
            if self.program_status == "done":
                new_clock = self.clock_time
        else:
            raise Exception("scheduling policy:" + config.scheduling_policy + " is not supported")
        """

        #return new_clock

    # ------------------------------
    # Functionality:
    #   determine the various KPIs for each kernel.
    #   work-rate is how quickly each kernel can be done, which depends on it's bottleneck
    # ------------------------------
    def update_kernels_kpi_for_next_tick(self, design):
        # update each kernels's work-rate (bandwidth)
        _ = [kernel.update_block_att_work_rate(self.scheduled_kernels) for kernel in self.scheduled_kernels]
        # update each pipe cluster's paths (inpipe-outpipe) work-rate

        _ = [kernel.update_pipe_clusters_pathlet_work_rate() for kernel in self.scheduled_kernels]
        # update each pipe cluster's paths (inpipe-outpipe) latency. Note that latency update must run after path's work-rate

        # update as it depends on it
        # for fast simulation, ignore this
        #_ = [kernel.update_path_latency() for kernel in self.scheduled_kernels]
        #_ = [kernel.update_path_structural_latency(design) for kernel in self.scheduled_kernels]


    # ------------------------------
    # Functionality:
    #   how much work does each block do for each phase
    # ------------------------------
    def calc_design_work(self):
        for SOC_type, SOC_id in self.design.get_designs_SOCs():
            blocks_seen = []
            for kernel in self.scheduled_kernels:
                if kernel.SOC_type == SOC_type and kernel.SOC_id == SOC_id:
                    for block, work in kernel.block_phase_work_dict.items():
                        if block not in blocks_seen :
                            blocks_seen.append(block)
                        #if block in self.block_phase_work_dict.keys():
                        if self.phase_num in self.design.block_phase_work_dict[block].keys():
                            self.design.block_phase_work_dict[block][self.phase_num] += work[self.phase_num]
                        else:
                            self.design.block_phase_work_dict[block][self.phase_num] = work[self.phase_num]
            all_blocks = self.design.get_blocks()
            for block in all_blocks:
                if block in  blocks_seen:
                    continue
                self.design.block_phase_work_dict[block][self.phase_num] = 0

    # ------------------------------
    # Functionality:
    #   calculate the utilization of each block in the design
    # ------------------------------
    def calc_design_utilization(self):
        for SOC_type, SOC_id in self.design.get_designs_SOCs():
            for block,phase_work in self.design.block_phase_work_dict.items():
                if self.design.phase_latency_dict[self.phase_num] == 0:
                    work_rate = 0
                else:
                    work_rate = (self.design.block_phase_work_dict[block][self.phase_num])/self.design.phase_latency_dict[self.phase_num]
                self.design.block_phase_utilization_dict[block][self.phase_num] = work_rate/block.peak_work_rate

    # ------------------------------
    # Functionality:
    #   Aggregates the energy consumed for current phase over all the blocks
    # ------------------------------
    def calc_design_energy(self):
        for SOC_type, SOC_id in self.design.get_designs_SOCs():
            self.design.SOC_phase_energy_dict[(SOC_type, SOC_id)][self.phase_num] = \
                sum([kernel.stats.phase_energy_dict[self.phase_num] for kernel in self.scheduled_kernels
                    if kernel.SOC_type == SOC_type and kernel.SOC_id == SOC_id])
            if config.simulation_method == "power_knobs":
                # Add up the leakage energy to the total energy consumption
                # Please note the phase_leakage_energy_dict only counts for PE and IC energy (no mem included)
                # since memory cannot be cut-off; otherwise will lose its contents
                self.design.SOC_phase_energy_dict[(SOC_type, SOC_id)][self.phase_num] += \
                    sum([kernel.stats.phase_leakage_energy_dict[self.phase_num] for kernel in self.scheduled_kernels
                         if kernel.SOC_type == SOC_type and kernel.SOC_id == SOC_id])

                # Add the leakage power for memories
                self.design.SOC_phase_energy_dict[(SOC_type, SOC_id)][self.phase_num] += \
                    sum([block.get_leakage_power() * self.time_step_size for block in self.design.get_blocks()
                         if block.get_block_type_name() == "mem"])

    # ------------------------------
    # Functionality:
    #   step the simulator forward, by moving all the kernels forward in time
    # ------------------------------
    def step(self):
        # the time step of the previous phase
        self.time_step_size = self.clock_time - self.old_clock_time
        # add the time step (time spent in the phase) to the design phase duration dictionary
        self.design.phase_latency_dict[self.phase_num] = self.time_step_size

        # advance kernels
        before_time = time.time()
        self.step_kernels()
        self.task_update_time += (time.time() - before_time)

        before_time = time.time()
        # Aggregates the energy consumed for current phase over all the blocks
        self.calc_design_energy()   # needs be done after kernels have stepped, to aggregate their energy and divide
        self.calc_design_work()   # calculate how much work does each block do for this phase
        self.calc_design_utilization()
        self.phase_interval_calc_time += (time.time() - before_time)

        before_time = time.time()
        self.update_scheduled_kernel_list()  # if a kernel is done, schedule it out
        self.phase_scheduling_time += (time.time() - before_time)

        #self.schedule_kernels()  # schedule ready to be scheduled kernels

        self.schedule_kernels_token_based()
        self.old_clock_time = self.clock_time  # update clock

        # check if execution is completed or not!
        self.update_program_status()
        before_time = time.time()
        self.update_kernels_kpi_for_next_tick(self.design)  # update each kernels' work rate
        self.phase_interval_calc_time += (time.time() - before_time)

        self.phase_num += 1
        self.update_parallel_tasks()

        # return the new tick position
        before_time = time.time()
        new_tick_position = self.calc_new_tick_position()
        self.phase_interval_calc_time += (time.time() - before_time)

        return new_tick_position, self.program_status

    # ------------------------------
    # Functionality:
    #   call the simulator
    # ------------------------------
    def simulate(self, clock_time):
        self.tick(clock_time)
        return self.step()