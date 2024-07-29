#!/bin/bash

rm -f west.log
#w_run --work-manager serial "$@" &> west.log
#w_run "$@" &> west.log
w_run --work-manager threads "$@" &> west.log
