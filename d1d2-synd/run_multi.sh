#!/bin/bash

#SYSTEMS=(template-stdMD)
#SYSTEMS=(template-stdHK template-LCAS template-REAP)
#SYSTEMS=(template-LCAS template-REAP)
SYSTEMS=(template-REAP)
ITERS=2000

# loop each sys
for SYS in ${SYSTEMS[@]} ; do
    cd $SYS
    # create dir using template files
    for V in {00..04} ; do
        # copy over master files
        #cp -v ../west.cfg .
        mkdir $V
        cp * $V/
        # replace text with V
        cd $V
        sed -i "s/MAXTOTALITERS/${ITERS}/" west.cfg
        sed -i "s/REPLACEWITHVERSION/v${V}/g" *
        # run WE within V
        #./init.sh &&
        ./run.sh &&
        w_ipa -ao -W west-v${V}.h5 &&
        cd ../
    done
    cd ../
done
