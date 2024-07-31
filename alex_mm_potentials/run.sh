#!/bin/bash

rm -f west.log
w_run "$@" &> west_LCAS.log
#w_run "$@" &> west_TSLC.log
#w_run "$@" &> west_REAP.log
#w_run "$@" &> west_FAST.log
#w_run "$@" &> west.log
