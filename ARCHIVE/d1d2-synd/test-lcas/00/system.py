from __future__ import print_function, division
import numpy as np
from westpa.core.systems import WESTSystem
from westpa.core.binning import RectilinearBinMapper
from westpa.core.binning import VectorizingFuncBinMapper

import logging
log = logging.getLogger('westpa.rc')

# number of walkers per bin
bintargetcount = 80
#bintargetcount = 8
#bintargetcount = 4
# number of dimensions
numberofdim = 6
# n_bins per dim
#binsperdim = [1, 1]
# length of the pcoord
pcoordlength = 2

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
    # put everything in bin 0 unless it's in the target state (bin 1)
    oa1, oa2, tt, c2, m1x, m2x = coords
    #print(coords)

    # set tstate conditions
    if oa1 < 10 and oa2 < 10 and tt < 5 and c2 > 40 and -95 < m1x < -40 and -95 < m2x < -40:
        print("TS", oa1, oa2, tt, c2)
        return 1
    # otherwise assign to binless bin
    else:
        #print("0-state", oa1, oa2, tt, c2)
        return 0
    
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
        # self.bin_mapper = RectilinearBinMapper([['-inf', -20, -10, 0, 10, 20, 'inf'],
        #                                         ['-inf', -20, -10, 0, 10, 20, 'inf']])
        # optimal 2D bins with knowledge of the system/potential
        # self.bin_mapper = RectilinearBinMapper([['-inf', -16, -6, -3, 0, 10, 'inf'],
        #                                         ['-inf', -16, -5, 0, 2, 4, 10, 'inf']])
        # FuncBinMapper(function, nbins)
        # 2 bins: one for 'binless' resampling and one for target state
        self.bin_mapper = VectorizingFuncBinMapper(vec_function_map, 2)
        
        # target counts (walkers per bin)
        self.bin_target_counts = np.empty((self.bin_mapper.nbins,), np.int_)
        self.bin_target_counts[...] = bintargetcount
