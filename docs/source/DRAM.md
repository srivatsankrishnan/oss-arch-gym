# DRAM Simulator Documentation

## Overview

The DRAM simulator is an architecture simulation tool with a configurable parameter space. This document outlines the parameters and their sample values.

## Configuration Parameters

The configuration space consists of the following parameters along with their sample values:

| Parameter             | Sample Value      |
|-----------------------|-------------------|
| PagePolicy            | Open              |
| Scheduler             | Fifo              |
| SchedulerBuffer       | Bankwise          |
| RequestBufferSize     | 8                 |
| CmdMux                | Oldest            |
| RespQueue             | Fifo              |
| RefreshPolicy         | AllBank           |
| RefreshMaxPostponed   | 8                 |
| RefreshMaxPulledin    | 8                 |
| PowerDownPolicy       | NoPowerDown       |
| Arbiter               | Simple            |
| MaxActiveTransactions | 128               |

## Parameter Mappers

Here are the mapper functions for each parameter:

- `page_policy_mapper`: {0: "Open", 1: "OpenAdaptive", 2: "Closed", 3: "ClosedAdaptive"}
- `scheduler_mapper`: {0: "Fifo", 1: "FrFcfsGrp", 2: "FrFcfs"}
- `schedulerbuffer_mapper`: {0: "Bankwise", 1: "ReadWrite", 2: "Shared"}
- `request_buffer_size_mapper`: [1, 2, 4, 8, 16, 32, 64, 128]
- `respqueue_mapper`: {0: "Fifo", 1: "Reorder"}
- `refreshpolicy_mapper`: {0: "NoRefresh", 1: "AllBank"}
- `refreshmaxpostponed_mapper`: [1, 2, 4, 8]
- `refreshmaxpulledin_mapper`: [1, 2, 4, 8]
- `arbiter_mapper`: {0: "Simple", 1: "Fifo", 2: "Reorder"}
- `max_active_transactions_mapper`: [1, 2, 4, 8, 16, 32, 64, 128]

## Utilizing the Parameter Space

We utilize supported vizier algorithms to suggest parameters within this space. The training scripts for each algorithm integrated with the DRAM simulator can be found [here](www.apple.com).

## Conclusion

The DRAM simulator offers a configurable and versatile way to explore architecture scenarios using the defined parameter space. For more details on how to use the simulator and interpret results, refer to the provided training scripts and documentation.
