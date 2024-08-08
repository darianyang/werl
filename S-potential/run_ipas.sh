#!/bin/bash

SYSTEMS=(template-stdMD-test)
#SYSTEMS=(template-stdHK1D template-stdHK2D template-LCAS template-REAP)

# loop each sys and run kinetics calc
for SYS in ${SYSTEMS[@]} ; do
    cd $SYS
    for V in {00..04} ; do
        cd $V
        w_ipa -ao -W west-v${V}.h5 &&
        cd ../
    done
    cd ..
done
