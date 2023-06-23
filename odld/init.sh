#!/bin/bash

rm -f west.h5 binbounds.txt
BSTATES="--bstate initial,9.5"
TSTATES="--tstate final,1.9"

#w_init $BSTATES $TSTATES "$@"
#w_init $BSTATES "$@"

#w_init $BSTATES $TSTATES --segs-per-state 50 "$@"
w_init $BSTATES --segs-per-state 50 "$@"
