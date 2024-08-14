#!/bin/bash

# Make sure environment is set
#source env.sh

# Clean up
rm -f west-v00.log west-v00.h5

# Run w_run
#w_init --bstate 'basis,4,4,4,4,4,4' --tstate 'target,0,0,0,0,0,0'
w_init --bstate 'basis,4,4,4,4,4,4' --tstate 'target,5,5,3,50,-80,-80'
#w_init --bstate 'basis,4,4,4,4,4,4' 

#w_run > west.log
w_run > west-v00.log 

#w_run --work-manager processes "$@" &> west.log
