from __future__ import print_function, division
import logging
from sympy import diff, exp, symbols, lambdify

import numpy as np
from numpy.random import normal as random_normal

import westpa

from westpa.core.binning import RectilinearBinMapper
from westpa.core.propagators import WESTPropagator
from westpa.core.systems import WESTSystem

from westpa.core.propagators import WESTPropagator
from westpa.core.systems import WESTSystem
from westpa.core.binning import RectilinearBinMapper
from westpa.core.binning import RecursiveBinMapper

PI = np.pi
log = logging.getLogger(__name__)


class ODLDPropagator(WESTPropagator):
    def _calc_gradient(self):
        A, B, C, D, E, x0, y0 = self.A, self.B, self.C, self.D, self.E, self.x0, self.y0

        x, y = symbols('x y')
        
        logU1 = -A * ((x-.25)**2) - A * ((y-.75)**2) - 2 * B * (x-.25) * (y-.75)
        dxU1 = diff(exp(logU1), x)
        dyU1 = diff(exp(logU1), y)
        
        logU2 = -C * (x**2) * ((1-x)**2) * (y**2) * ((1-y)**2)
        dxU2 = diff(exp(logU2), x)
        dyU2 = diff(exp(logU2), y)
        
        logU3 = -D * (x**2) - D * (y**2) + 2 * E * x * y
        dxU3 = diff(exp(logU3), x)
        dyU3 = diff(exp(logU3), y)

        gradx = (dxU1 + dxU2 + 0.5 * dxU3)
        grady = (dyU1 + dyU2 + 0.5 * dyU3)

        return lambdify([x, y], gradx, "numpy"), lambdify([x, y], grady, "numpy")

    def __init__(self, rc=None):
        super().__init__(rc)

        self.coord_len = 5
        self.coord_dtype = np.float32
        self.coord_ndim = 2
		
        self.initial_pcoord = np.array([0.1,0.5], dtype=self.coord_dtype)

        self.sigma = 0.0001 ** (0.5)  # friction coefficient

        self.A = 50.5
        self.B = 49.5
        self.C = 10000
        self.D = 51
        self.E = 49
        self.x0 = 1
        self.y0 = 1

        self.grad_x, self.grad_y = self._calc_gradient()

        # Implement a reflecting boundary at this x value
        # (or None, for no reflection)
        self.reflect_at_x0 = None
        self.reflect_at_x = None
        self.reflect_at_y0 = None
        self.reflect_at_y = None

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
                coord_array[idx] -= 2 * (min_bound - coord)

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

        sigma = self.sigma
        gradfactor = self.sigma * self.sigma / 2
        coord_len = self.coord_len
        
        all_displacements = np.zeros(
            (n_segs, self.coord_len, self.coord_ndim), dtype=self.coord_dtype
        )

        for istep in range(0, coord_len):
            xi = coords[:, istep - 1, 0]
            yi = coords[:, istep - 1, 1]
            # log.info(f'{xi}, {yi}')

            #print(xi, yi)

            all_displacements[:, istep, 0] = x_displacements = random_normal(
                scale=sigma, size=(n_segs,)
            )
            all_displacements[:, istep, 1] = y_displacements = random_normal(
                scale=sigma, size=(n_segs,)
            )

            # log.info(f'{xi}, {yi}')
            newx = xi - (gradfactor * self.grad_x(xi, yi)) + x_displacements
            newy = yi - (gradfactor * self.grad_y(xi, yi)) + y_displacements
   
            # Update coords, return reflected coords
            coords[:, istep, 0] = self._reflect(newx, self.reflect_at_x0, self.reflect_at_x)
            coords[:, istep, 1] = self._reflect(newy, self.reflect_at_y0, self.reflect_at_y)

        for iseg, segment in enumerate(segments):
            segment.pcoord[...] = coords[iseg, :]
            segment.data["displacement"] = all_displacements[iseg]
            segment.status = segment.SEG_STATUS_COMPLETE

        return segments