#!/bin/bash

#SYSTEMS=(template-FAST template-LCAS template-REAP template-TSLC template-stdMD template-stdHK2D)
#SYSTEMS=(template-stdMD template-stdHK2D)
SYSTEMS=(template-stdHK1D_finer)

# loop each sys
for SYS in ${SYSTEMS[@]} ; do
    cd $SYS
    # create dir using template files
    for V in {00..09} ; do
        mkdir $V
        cp *.* $V/
        # replace text with V
        cd $V
        sed -i "s/REPLACEWITHVERSION/v${V}/g" *
        # run WE within V
        ./init.sh &&
        ./run.sh &&
        cd ../
    done
    cd ..
done
