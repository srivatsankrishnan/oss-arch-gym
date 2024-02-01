#!/bin/bash

# git clone git@github.com:astra-sim/astrasim-archgym.git
cd astrasim_archgym_public
# git checkout v2.2.0
# git submodule update --init --recursive

cd astra-sim
# bash ./build/astra_analytical/build.sh

cd extern/graph_frontend/chakra
python3 -m pip uninstall chakra
python3 -m pip install . --user


