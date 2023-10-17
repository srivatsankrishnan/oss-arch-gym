# Architecture Gym (ArchGym)
### An OpenAI Gym Interface for Computer Architecture Research

Architecture Gym (ArchGym) is a systematic and standardized framework for ML-driven research tackling architectural design space exploration.
ArchGym currently supports five different ML-based search algorithms and three unique architecture simulators. The framework is built with the ability to be easily extended to support brand-new search algorithms and architecture simulators via a standardized interface.

![Alt text](./docs/ArchGym-animation.gif?raw=true "Title")

## Agents
We define “agent” as an encapsulation of the machine learning algorithm. An ML algorithm consists of “hyperparameters” and a guiding “policy”. 
We currently support the following agents:
- Ant Colony Optimization (ACO)
- Genetic Algorithm (GA)
- Bayesian Optimization (BO)
- Reinforcement Learning (RL)
- Random Walker (RW)
- Vizier Algorithms (WIP. Please stay tuned!)

## Environments (Simulators + Workloads)
Each environment is an encapsulation of the architecture cost model and the workload. The architecture cost model determines the cost of running the workload for a given set of architecture parameters. For example, the cost can be latency, throughput, area, or energy or any combination.
We currently support the following Gym Environments:
- DRAMGym     (DRAMSys Simulator + Memory Trace Workloads)
- TimeloopGym (Timeloop Simulator + CNN Workloads)
- FARSIGym    (FARSI Simulator + AR/VR Workloads)
- MaestroGym (as used in GAMMA paper + DNN Workloads)
- AstraSim (WIP. Please stay tuned)
- CFUPlayground (WIP. Please stay tuned)


## Paper

Checkout the [paper](https://dl.acm.org/doi/pdf/10.1145/3579371.3589049) for more information about ArchGym

```
@inproceedings{10.1145/3579371.3589049,
author = {Krishnan, Srivatsan and Yazdanbakhsh, Amir and Prakash, Shvetank and Jabbour, Jason and Uchendu, Ikechukwu and Ghosh, Susobhan and Boroujerdian, Behzad and Richins, Daniel and Tripathy, Devashree and Faust, Aleksandra and Janapa Reddi, Vijay},
title = {ArchGym: An Open-Source Gymnasium for Machine Learning Assisted Architecture Design},
year = {2023},
isbn = {9798400700958},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3579371.3589049},
doi = {10.1145/3579371.3589049},
booktitle = {Proceedings of the 50th Annual International Symposium on Computer Architecture},
articleno = {14},
numpages = {16},
keywords = {machine learning, reinforcement learning, machine learning for system, open source, reproducibility, bayesian optimization, baselines, machine learning for computer architecture},
location = {Orlando, FL, USA},
series = {ISCA '23}
}
```

Checkout the [Sigarch Blog](https://www.sigarch.org/architecture-2-0-why-computer-architects-need-a-data-centric-ai-gymnasium/) for the high level motivation and how ArchGym serves as a template for Architecture 2.0

Please stay tuned regarding update to this space. We look forward to your contribution in this space.

