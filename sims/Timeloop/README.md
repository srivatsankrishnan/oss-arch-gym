# Timeloop Simulator Documentation

## Overview

The Timeloop simulator is an architecture simulation tool for evaluating Deep Neural Network Accelerator designs. 

## Running the Simulator

To run the simulator, you need to build the docker image first and run the docker container. Multiple agents can be used to run the simulator.

1. Change directory to the Timeloop simulator
```
cd sims/Timeloop
```

2. Build the docker image
```
sudo ./build.sh
```

3. Run the docker container
```
docker run timeloop_4_archgym --algo=aco
```

