#!/bin/bash
echo "RUNNING ACME INSTALLATION SHELL SCRIPT"
if export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/arch-gym/lib/; then
    echo "Command succeeded"
else
    echo "Command failed"
fi

# Change directory to the "acme" folder
cd ../../acme

# Install required packages using pip
pip install .[tf,testing,envs,jax]