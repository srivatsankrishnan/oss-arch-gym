#!/bin/bash

# Change directory to the "acme" folder
cd ../../acme

# Install required packages using pip
pip install .[tf,testing,envs,jax]