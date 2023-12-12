from __future__ import print_function, division
import logging

#from sympy import diff, exp, symbols, lambdify
from potentials import *

import numpy as np
from numpy.random import normal as random_normal

from westpa.core.propagators import WESTPropagator

PI = np.pi
log = logging.getLogger(__name__)

##########################
### SET POTENTIAL HERE ###
##########################
potential = we_odld_2d
#potential = L_potential
##########################
##########################
##########################

class ODLDPropagator(WESTPropagator):

    ###########################
    ### SET PARAMETERS HERE ###
    ###########################
    # initial XY position
    #xy_position = [0, 0]
    #xy_position = [1, 1]
    #xy_position = [9.5, 9.5]
    xy_position = [-9.5, -9.5]
    #xy_position = [1.8, 1.8]
    # pcoord params
    coord_len = 5
    coord_dtype = np.float32
    coord_ndim = 2
    # Implement a reflecting boundary at this xy value
    # (or None, for no reflection)
    # reflect_at_x0 = -0.5
    # reflect_at_x = 2.5
    # reflect_at_y0 = -0.5
    # reflect_at_y = 2.5

    # reflect_at_x0 = 0
    # reflect_at_x = 10
    # reflect_at_y0 = 0
    # reflect_at_y = 10

    reflect_at_x0 = -10
    reflect_at_x = 0
    reflect_at_y0 = -10
    reflect_at_y = 0

    # reflect_at_x0 = None
    # reflect_at_x = None
    # reflect_at_y0 = None
    # reflect_at_y = None

    # friction coefficient
    sigma = 0.001 ** (0.5)  
    ###########################
    ###########################
    ###########################

    def __init__(self, rc=None):
        super().__init__(rc)
        self.initial_pcoord = np.array(self.xy_position, dtype=self.coord_dtype)

    ##################################################################
    ### adjust for different potential function unpacking settings ###
    ##################################################################
    def _calc_gradient(self, x, y):
        grad = potential(x, y)
        return grad

    def get_pcoord(self, state):
        """Get the progress coordinate of the given basis or initial state."""
        state.pcoord = self.initial_pcoord.copy()

    def gen_istate(self, basis_state, initial_state):
        initial_state.pcoord = self.initial_pcoord.copy()
        initial_state.istate_status = initial_state.ISTATE_STATUS_PREPARED
        return initial_state

    def _reflect(self, coord_array, min_bound, max_bound):
        # Variable wrangling...
        if min_bound is None:
            min_bound = -np.inf
        if max_bound is None:
            max_bound = np.inf

        # Loop through each coord and reflect the same amount at that direction.
        for idx, coord in enumerate(coord_array):
            if coord < min_bound:
                coord_array[idx] += 2 * (min_bound - coord)
            elif coord > max_bound:
                coord_array[idx] += 2 * (max_bound - coord)

        return coord_array

    def propagate(self, segments):
        # log.info(segments)
        # Create empty array for coords
        n_segs = len(segments)
        coords = np.empty(
            (n_segs, self.coord_len, self.coord_ndim), dtype=self.coord_dtype
        )

        # Update each one with corresponding pcoord
        for iseg, segment in enumerate(segments):
            coords[iseg, 0] = segment.pcoord[0]

        gradfactor = self.sigma * self.sigma / 2
        
        all_displacements = np.zeros(
            (n_segs, self.coord_len, self.coord_ndim), dtype=self.coord_dtype
        )

        for istep in range(1, self.coord_len):
            xi = coords[:, istep - 1, 0]
            yi = coords[:, istep - 1, 1]
            # log.info(f'{xi}, {yi}')

            #print(xi, yi)

            all_displacements[:, istep, 0] = x_displacements = random_normal(
                scale=self.sigma, size=(n_segs,)
            )
            all_displacements[:, istep, 1] = y_displacements = random_normal(
                scale=self.sigma, size=(n_segs,)
            )

            ### ADJUST GRAD ###
            #log.info(f'XY vars: {xi}, {yi}')
            grad = self._calc_gradient(xi, yi)
            # gradfactor = sigma**2 / 2 = 0.00005
            newx = xi - (gradfactor * grad) + x_displacements
            newy = yi - (gradfactor * grad) + y_displacements
   
            # Update coords, return reflected coords
            coords[:, istep, 0] = self._reflect(newx, self.reflect_at_x0, self.reflect_at_x)
            coords[:, istep, 1] = self._reflect(newy, self.reflect_at_y0, self.reflect_at_y)

        for iseg, segment in enumerate(segments):
            segment.pcoord[...] = coords[iseg, :]
            segment.data["displacement"] = all_displacements[iseg]
            segment.status = segment.SEG_STATUS_COMPLETE

        return segments
