#!/bin/bash

rm -f west.h5
BSTATES="--bstate initial,1"
#TSTATES="--tstate final,1.9,1.9"
#TSTATES="--tstate final,0,-1.8"

#w_init $BSTATES $TSTATES "$@"
#w_init $BSTATES "$@"

#w_init $BSTATES $TSTATES --segs-per-state 40 "$@"
w_init $BSTATES --segs-per-state 40 "$@"
#w_init $BSTATES --segs-per-state 100 "$@"


#BSTATES="--bstate-file bstates.txt"
#TSTATES="--tstate-file tstates.txt"
#w_init $BSTATES $TSTATES "$@"
#w_init $BSTATES "$@"
