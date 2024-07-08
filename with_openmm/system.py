from __future__ import print_function, division
import numpy as np
from westpa.core.systems import WESTSystem
from westpa.core.binning import RectilinearBinMapper
from westpa.core.binning import VectorizingFuncBinMapper

import logging
log = logging.getLogger('westpa.rc')

# number of walkers per bin
bintargetcount = 80
#bintargetcount = 5
# number of dimensions
numberofdim = 2
# n_bins per dim
binsperdim = [1, 1]
# length of the pcoord
pcoordlength = 21

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
	# for Z-potential: tstate is X<-10 & 0<Y<6
	# put everything in bin 0 unless it's in the target state (bin 1)
	x, y = coords

	# set tstate conditions
	#if x < -10 and 0 < y < 6:
	if x > 10 and y > 10:
		return 1
	# otherwise assign to binless bin
	else:
		return 0
    
class System(WESTSystem):
	def initialize(self):
		# pcoord variables
		self.pcoord_ndim = numberofdim
		self.pcoord_len = pcoordlength
		self.pcoord_dtype = np.float32 
		
        # rectilinear bin mapper
		#self.bin_mapper = RectilinearBinMapper([['-inf', 'inf'],['-inf', 'inf']])
		#self.bin_mapper = RectilinearBinMapper([['-inf', -20, -10, 0, 10, 20, 'inf'],
        #                                        ['-inf', 0, 6, 'inf']])
		# FuncBinMapper(function, nbins)
		# 2 bins: one for 'binless' resampling and one for target state
		self.bin_mapper = VectorizingFuncBinMapper(vec_function_map, 2)
		
        # target counts (walkers per bin)
		self.bin_target_counts = np.empty((self.bin_mapper.nbins,), np.int_)
		self.bin_target_counts[...] = bintargetcount
