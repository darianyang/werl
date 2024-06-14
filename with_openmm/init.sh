#!/bin/bash

rm -f west.h5
BSTATES="--bstate initial,-10,-6"
TSTATES="--tstate final,-15,5"

#w_init $BSTATES $TSTATES "$@"
#w_init $BSTATES "$@"

w_init $BSTATES $TSTATES --segs-per-state 100 "$@"
#w_init $BSTATES --segs-per-state 100 "$@"
#w_init $BSTATES --segs-per-state 80 "$@"

#BSTATES="--bstate-file bstates.txt"
#TSTATES="--tstate-file tstates.txt"
#w_init $BSTATES $TSTATES "$@"
#w_init $BSTATES "$@"
