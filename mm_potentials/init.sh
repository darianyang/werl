#!/bin/bash

#rm -f west.h5 op_weights.txt
rm -f west.h5

# bstates are handled in odld.py but tstates must be accurately placed in bin
BSTATES="--bstate initial,-9.5,-9.5"
TSTATES="--tstate final,-1.5,-1.5"
#TSTATES="--tstate final,1000,1000"

#w_init $BSTATES $TSTATES "$@"
#w_init $BSTATES "$@"

#w_init $BSTATES $TSTATES --segs-per-state 50 "$@"
#w_init $BSTATES --segs-per-state 100 "$@"
#w_init $BSTATES --segs-per-state 80 "$@"
w_init $BSTATES $TSTATES --segs-per-state 80 "$@"

#BSTATES="--bstate-file bstates.txt"
#TSTATES="--tstate-file tstates.txt"
#w_init $BSTATES $TSTATES "$@"
#w_init $BSTATES "$@"
