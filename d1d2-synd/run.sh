#!/bin/bash

# Make sure environment is set
#source env.sh

# Clean up
rm -f west.log

# Run w_run
w_init --bstate 'basis,1,108' --tstate 'target,50,-130'
# Note these states have the pcoord of:
# Basis: -86.38787085,43.94791794 --> Corresponds to C7eq of alanine dipeptide
# Target: 56.9945094,-164.70361737--> Corresponds to C7ax of alanine dipeptide (Closest center 119)
w_run > west.log

#w_run --work-manager processes "$@" &> west.log
