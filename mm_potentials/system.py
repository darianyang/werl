from __future__ import print_function, division
import numpy as np
from westpa.core.systems import WESTSystem
from westpa.core.binning import RectilinearBinMapper
from westpa.core.binning import FuncBinMapper
from westpa.core.binning import RecursiveBinMapper

import logging
log = logging.getLogger('westpa.rc')

#THESE ARE THE FOUR THINGS YOU SHOULD CHANGE
bintargetcount=80 #number of walkers per bin
numberofdim=2  # number of dimensions
binsperdim=[1,1]   # You will have prod(binsperdim)+numberofdim*(2+2*splitIsolated)+activetarget bins total
pcoordlength=21 # length of the pcoord

targetstate=[None,None]    #enter boundaries for target state or None if there is no target state in that dimension
activetarget=0		#if no target state make this zero

#########
def function_map(coords, mask, output):
	# test: bin 0 if sum of xy is < 1000, and bin 1 if > 1000
    output[np.abs(coords[:,0] + coords[:,1]) < 1000] = 0
    output[np.abs(coords[:,0] + coords[:,1]) >= 1000] = 1
    
class System(WESTSystem):
	def initialize(self):
		# pcoord variables
		self.pcoord_ndim = numberofdim
		self.pcoord_len = pcoordlength
		self.pcoord_dtype = np.float32 
		
        # bin mapper: FuncBinMapper(function, nbins)
		#self.bin_mapper = FuncBinMapper(function_map, np.prod(binsperdim)+numberofdim*(2+2*splitIsolated)+activetarget)
		#self.bin_mapper = RectilinearBinMapper([['-inf', 'inf'],['-inf', 'inf']])
		#self.bin_mapper = FuncBinMapper(function_map, np.prod(binsperdim)+numberofdim)
		self.bin_mapper = FuncBinMapper(function_map, 2)
		
        # target counts (walkers per bin)
		self.bin_target_counts = np.empty((self.bin_mapper.nbins,), np.int_)
		self.bin_target_counts[...] = bintargetcount