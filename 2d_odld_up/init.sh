#!/bin/bash

rm -f west.h5 binbounds.txt
#BSTATES="--bstate initial,1,9.5"
#TSTATES="--tstate final,1.9,1.9"

#w_init $BSTATES $TSTATES "$@"
#w_init $BSTATES "$@"

#w_init $BSTATES $TSTATES --segs-per-state 50 "$@"
#w_init $BSTATES --segs-per-state 50 "$@"

#BSTATES="--bstate-file bstates.txt"
#TSTATES="--tstate-file tstates.txt"
#w_init $BSTATES $TSTATES "$@"

rm -f west.h5 binbounds.txt
#BSTATES="--bstate-file bstates.txt"
BSTATES="--bstate start,1,0.1"
#TSTATES="--tstate final,2.51,2.51"
#w_init --segs-per-state 40 $BSTATES $TSTATES "$@"
#w_init $BSTATES $TSTATES "$@"
w_init $BSTATES "$@"
