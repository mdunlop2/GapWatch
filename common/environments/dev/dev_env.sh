#!/bin/bash

# Script for setting up the rapid development environment locally,
# no nvidia gpu available

# install opencv with opencl capabilities
apt-get install -y --no-install-recommends \
        ocl-icd-libopencl1 \
        clinfo &&

# setup the python environment

# Reboot
reboot