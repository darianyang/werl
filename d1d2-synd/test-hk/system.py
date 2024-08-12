from __future__ import print_function, division
import numpy as np
from westpa.core.systems import WESTSystem
from westpa.core.binning import RectilinearBinMapper
from westpa.core.binning import VectorizingFuncBinMapper

import logging
log = logging.getLogger('westpa.rc')

# number of walkers per bin
#bintargetcount = 80
#bintargetcount = 8
bintargetcount = 8
# number of dimensions
numberofdim = 6
# n_bins per dim
#binsperdim = [1, 1]
# length of the pcoord
pcoordlength = 2

def classify_into_bin(bins, value):
    """
    Assigns value to a bin index from bins list.
    """
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            # Bin numbers start from 1   
            return i + 1  
    # Return None if the value doesn't fall within any bin
    return None  

def vec_function_map(coords):
    """
    Functional mapper for assigning bins based on pcoords.
    Evaluated once for each unmasked coordinate tuple provided (single frame).

    Parameters
    ----------
    coords : ndarray
        An array with pcoord and weight info.

    Returns
    ------
    bin_assignment : int
    """
    # put everything in bin unless it's in the target state (bin 0)
    oa1, oa2, tt, c2, m1x, m2x = coords
    #print(coords)

    # set tstate conditions
    if oa1 < 10 and oa2 < 10 and tt < 5 and c2 > 40 and -95 < m1x < -40 and -95 < m2x < -40:
        print("TS", oa1, oa2, tt, c2)
        return 0
    # otherwise assign to a bin (here, based on tt_dist)
    else:
        return classify_into_bin([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, np.inf], tt)
    
class System(WESTSystem):
    def initialize(self):
        # pcoord variables
        self.pcoord_ndim = numberofdim
        self.pcoord_len = pcoordlength
        self.pcoord_dtype = np.float32 
        
        # TODO: add third pcoord of noise

        # rectilinear bin mapper
        #self.bin_mapper = RectilinearBinMapper([['-inf', 'inf'],['-inf', 'inf']])
        # using 1D pcoord for the 2D DOF particle space
        #self.bin_mapper = RectilinearBinMapper([['-inf', -20, -15, -10, -5, 0, 5, 10, 20, 'inf'],
        #                                        ['-inf', 10, 'inf']])
        # rough estimate of the 2D landscape
        #self.bin_mapper = RectilinearBinMapper([[0, 10, 'inf'],
        #                                        [0, 10, 'inf'],
        #                                        [0, 5, 'inf'],
        #                                        [0, 50, 'inf'],
        #                                        ['-inf', -95, -45, 'inf'],
        #                                        ['-inf', -95, -45, 'inf']])
        # FuncBinMapper(function, nbins)
        # 2 bins: one for 'binless' resampling and one for target state
        self.bin_mapper = VectorizingFuncBinMapper(vec_function_map, 12)
        
        # target counts (walkers per bin)
        self.bin_target_counts = np.empty((self.bin_mapper.nbins,), np.int_)
        self.bin_target_counts[...] = bintargetcount
