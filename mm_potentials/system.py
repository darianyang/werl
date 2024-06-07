from __future__ import print_function, division
import numpy as np
from westpa.core.systems import WESTSystem
from westpa.core.binning import RectilinearBinMapper
from westpa.core.binning import FuncBinMapper
from westpa.core.binning import RecursiveBinMapper

import logging
log = logging.getLogger('westpa.rc')

# number of walkers per bin
bintargetcount = 80
# number of dimensions
numberofdim = 2
# n_bins per dim
binsperdim = [1, 1]
# length of the pcoord
pcoordlength = 21

# tstate boundary per dim
#targetstate=[-2,-2]
targetstate=[-8,-8]

def function_map(coords, mask, output):
	# test: bin 0 if sum of xy is < 1000, and bin 1 if > 1000
    # output[np.abs(coords[:,0] + coords[:,1]) < 1000] = 0
    # output[np.abs(coords[:,0] + coords[:,1]) >= 1000] = 1
	
	# for 2D ODLD: tstate is < 2 in x and y
	# so put everything in bin 0 unless it's in the target state (bin 1)
	output[(coords[:,0] < targetstate[0]) & (coords[:,0] < targetstate[0])] = 0
	output[(coords[:,0] >= targetstate[1]) & (coords[:,0] >= targetstate[1])] = 1
	
    
class System(WESTSystem):
	def initialize(self):
		# pcoord variables
		self.pcoord_ndim = numberofdim
		self.pcoord_len = pcoordlength
		self.pcoord_dtype = np.float32 
		
        # rectilinear bin mapper
		#self.bin_mapper = RectilinearBinMapper([['-inf', 'inf'],['-inf', 'inf']])
		# FuncBinMapper(function, nbins)
		# 2 bins: one for functional bin mapper and one for target state
		self.bin_mapper = FuncBinMapper(function_map, 2)
		
        # target counts (walkers per bin)
		self.bin_target_counts = np.empty((self.bin_mapper.nbins,), np.int_)
		self.bin_target_counts[...] = bintargetcount