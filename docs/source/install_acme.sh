#!/bin/bash

# Set execute permissions for the script itself
chmod +x "$0"

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/$USERNAME/anaconda3/envs/arch-gym/lib/"

# Change directory to the "acme" folder
cd ../../acme

# Install required packages using pip
pip install .[jax,tf,testing,envs]
