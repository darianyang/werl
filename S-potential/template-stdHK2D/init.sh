#!/bin/bash

BSTATES="--bstate initial,-15,-15"
TSTATES="--tstate final,15,15"

w_init $BSTATES $TSTATES --segs-per-state 80 "$@"
