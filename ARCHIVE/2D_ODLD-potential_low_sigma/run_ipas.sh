#!/bin/bash

#SYSTEMS=(template-FAST)
SYSTEMS=(template-stdMD template-stdHK1D template-stdHK2D template-LCAS template-REAP template-TSLC template-FAST)

# loop each sys and run kinetics calc
for SYS in ${SYSTEMS[@]} ; do
    cd $SYS
    for V in {00..09} ; do
        cd $V
        w_ipa -ao -W west-REPLACEWITHVERSION.h5 &&
        cd ../
    done
    cd ..
done
