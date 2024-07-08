#!/bin/bash

rm -f west.h5
BSTATES="--bstate initial,-15,-15"
TSTATES="--tstate final,15,15"

#w_init $BSTATES $TSTATES "$@"
#w_init $BSTATES "$@"

w_init $BSTATES $TSTATES --segs-per-state 80 "$@"
#w_init $BSTATES --segs-per-state 100 "$@"
#w_init $BSTATES --segs-per-state 80 "$@"

#BSTATES="--bstate-file bstates.txt"
#TSTATES="--tstate-file tstates.txt"
#w_init $BSTATES $TSTATES "$@"
#w_init $BSTATES "$@"
