#!/bin/bash

rm -f west.log
#w_run --work-manager serial "$@" &> west.log
#w_run "$@" &> west.log
#w_run --work-manager threads "$@" &> west.log
#w_run --work-manager processes "$@" &> bigdata/west-LCAS.log
#w_run --work-manager processes "$@" &> bigdata/west-REAP.log
#w_run --work-manager processes "$@" &> bigdata/west-stdHK.log
#w_run --work-manager processes "$@" &> bigdata/west-stdMD.log
w_run --work-manager processes "$@" &> bigdata/west-stdHK1D.log
