#!/bin/bash

#SYSTEMS=(template-FAST template-LCAS template-REAP template-TSLC template-stdHK1D template-stdHK2D template-stdMD)
#SYSTEMS=(template-stdMD template-stdHK2D)
SYSTEMS=(template-stdHK2D-test)

# loop each sys
for SYS in ${SYSTEMS[@]} ; do
    cd $SYS
    # create dir using template files
    for V in {0..4} ; do
        mkdir $V
        cp * $V/
        # replace text with V
        cd $V
        #sed -i "s/REPLACEWITHVERSION/v${V}/g" *
        # run WE within V
        ./init.sh &&
        ./run.sh &&
        w_ipa -ao -W west-REPLACEWITHVERSION.h5 &&
        cd ../
    done
    cd ..
done
