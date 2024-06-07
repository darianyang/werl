#!/bin/bash

rm -f west.h5 op_weights.txt
#BSTATES="--bstate initial,-9.5,-9.5"
#TSTATES="--tstate final,-1.9,-1.9"

# fake coords since handled in prop and system
BSTATES="--bstate initial,1,1"
TSTATES="--tstate final,1,1"

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
