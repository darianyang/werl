from __future__ import print_function, division
import numpy
from westpa.core.propagators import WESTPropagator
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
	pass

class System(WESTSystem):
	def initialize(self):
		# pcoord variables
		self.pcoord_ndim = numberofdim
		self.pcoord_len = pcoordlength
		self.pcoord_dtype = numpy.float32 
		
        # bin mapper
		#self.bin_mapper = FuncBinMapper(function_map, prod(binsperdim)+numberofdim*(2+2*splitIsolated)+activetarget)
		self.bin_mapper = RectilinearBinMapper([['-inf', 'inf'],['-inf', 'inf']])
		
        # target counts (walkers per bin)
		self.bin_target_counts = numpy.empty((self.bin_mapper.nbins,), numpy.int_)
		self.bin_target_counts[...] = bintargetcount