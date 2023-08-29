#!/bin/bash

# Set execute permissions for the script itself
chmod +x "$0"

# Change directory to the "acme" folder
cd ../../acme

# Install required packages using pip
pip install .[jax,tf,testing,envs]
