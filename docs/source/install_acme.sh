#!/bin/bash

# Change directory to the "acme" folder
cd ../../acme

# Set execute permissions for the script itself
chmod +x "$0"

# Install required packages using pip
pip install .[jax,tf,testing,envs]