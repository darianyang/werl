from __future__ import print_function, division
import logging

#from sympy import diff, exp, symbols, lambdify
#from potentials import *

import numpy as np
#from numpy.random import normal as random_normal
import openmm as mm
from westpa.core.propagators import WESTPropagator

import potentials
four_wells_symmetric = potentials.gaussian_bivariate_string().replace('-80', '-25', 1)

PI = np.pi
log = logging.getLogger(__name__)

# with metastable well
# Z_potential_string = '''\
# (x^4 + y^4)/20480 - 3 * exp(-0.01 * (x+5)^2 - 0.2 * (y+5)^2) - \
# 13 * exp(-0.01 * (x-5)^2 - 0.2 * (y-5)^2) + \
# 5 * exp(-0.2 * (x+3*(y-3)^2))/(1+exp(-x-3)) + \
# 5 * exp(-0.2 * (x+3*(y+3)^2))/(1+exp(x-3)) + \
# 23 * exp(-0.01 *(x^2 + y^2))'''

# # with barrier
# Z_potential_string = '''\
# (x^4 + y^4)/20480 - 3 * exp(-0.01 * (x+5)^2 - 0.2 * (y+5)^2) - \
# -30 * exp(-0.01 * (x-5)^2 - 0.2 * (y-5)^2) + \
# 5 * exp(-0.2 * (x+3*(y-3)^2))/(1+exp(-x-3)) + \
# 5 * exp(-0.2 * (x+3*(y+3)^2))/(1+exp(x-3)) + \
# 23 * exp(-0.01 *(x^2 + y^2))'''

# updated with well and barrier
S_potential_string = '''\
(x^4 + y^4)/20480 - 3 * exp(-0.01 * (x+5)^2 - 0.2 * (y+5)^2) - \
-60 * exp(-0.003 * (x-5)^2 - 0.2 * (y-5)^2) + \
-35 * exp(-0.075 * (x+3*(y-3)^2))/(1+exp(-x-3)) + \
50 * exp(-0.2 * (x+3*(y+3)^2))/(1+exp(x-3)) + \
16 * exp(-0.01 *(x^2 + y^2))'''

class ODLDPropagator(WESTPropagator):

    ###########################
    ### SET PARAMETERS HERE ###
    ###########################
    # initial XY position
    #initial_position = [0, 0, 0]
    #initial_position = [0.25, 1, 0]
    #initial_position = [0, 1, 0]
    #initial_position = [0, -10, 0]
    #initial_position = [-10, -6, 0]
    #initial_position = [-10, -6]
    initial_position = [-15, -15]
    # pcoord params
    coord_len = 21
    coord_dtype = np.float32
    coord_ndim = 2

    # optionally set a potential, otherwise default 2 well XY
    #potential = four_wells_symmetric
    #potential = L_potential_string
    #potential = Z_potential_string
    potential = S_potential_string

    def __init__(self, rc=None):
        super().__init__(rc)
        self.initial_pcoord = np.array(self.initial_position, dtype=self.coord_dtype)

    def get_pcoord(self, state):
        """Get the progress coordinate of the given basis or initial state."""
        state.pcoord = self.initial_pcoord.copy()

    def gen_istate(self, basis_state, initial_state):
        initial_state.pcoord = self.initial_pcoord.copy()
        initial_state.istate_status = initial_state.ISTATE_STATUS_PREPARED
        return initial_state

    # TODO: import particle sim?
    def particle_sim(self, n_steps=20, potential='4*(x-1)^2*(x+1)^2+y^2',
                     initial_position=[0, 0, 0], T=300, gamma=1, step_size=0.02,
                     save_steps=10, save_traj=None, random_seed=None):
        '''
        Parameters
        ----------
        n_steps : int
            Number of steps for the trajectory. Default 5000.
        potential : str
            Input for OpenMM CustomExternalForce, the potential for particle
            simulation. Default uses a 2D ODLD double well potential with a 
            4 kT barrier height ('4*(x-1)^2*(x+1)^2+y^2').
        initial_position : list of [x,y,z] floats
            Initial particle positions.
        T : double
            Temperature of the ODLD system in Kelvin, default 500K.
        gamma : double
            Friction coefficient which couples the system to the heat bath (ps^-1).
            Default 1.0 ps^-1, increase to increase viscosity / collision frequency.
        step_size : double
            The step size with which to integrate the system (in ps).
            Default 0.02 ps.
        save_steps : int
            Integrator steps, default 1. Increase to save less frequently.
        save_traj : str
            Default None and no trajectory saving, otherwise input a path to the
            save the trajectory array.
        random_seed : int
            Random seed for selecting velocities. Default None for auto seed.

        Returns
        -------
        x : 3d array
            Where X and Y positions are the first 2 columns.
            The third column is Z positions if the potential is 3D.
        '''
        # initialize system and add particle with a unit mass (1)
        system = mm.System()
        system.addParticle(1)
        
        # defines the potential
        if hasattr(self, "potential"):
            force = mm.CustomExternalForce(self.potential)
        else:
            force = mm.CustomExternalForce(potential)

        force.addParticle(0, [])
        system.addForce(force)
        
        # Langevin integrator with custom temperature, friction, and step size
        integrator = mm.LangevinIntegrator(T, gamma, step_size)
        context = mm.Context(system, integrator)
        
        # initialize positions and velocities
        context.setPositions([initial_position])
        if random_seed:
            context.setVelocitiesToTemperature(T, random_seed)
        else:
            context.setVelocitiesToTemperature(T)

        # save coordinates
        x = np.zeros((n_steps, 3))
        for i in range(n_steps):
            x[i] = context.getState(getPositions=True).getPositions(asNumpy=True)._value
            integrator.step(save_steps)
        
        # optionally save trajectory as text file
        if save_traj is not None:
            np.savetxt(save_traj, x)
        return x

    def propagate(self, segments):
        log.info(segments)
        # Create empty array for coords
        n_segs = len(segments)
        coords = np.empty(
            (n_segs, self.coord_len, self.coord_ndim), dtype=self.coord_dtype
        )

        # Update each one with corresponding pcoord
        for iseg, segment in enumerate(segments):
            coords[iseg, 0] = segment.pcoord[0]

            # calc new coords for each segment with initial pcoord position start
            # some initial position manipulation for 2d xy traj output
            #print(np.concatenate((segment.pcoord[0], np.array([0]))))
            xyz_position = np.concatenate((segment.pcoord[0], np.array([0])))
            #print(segment.pcoord[0].dtype)
            #x = self.particle_sim(initial_position=segment.pcoord[0])
            x = self.particle_sim(initial_position=xyz_position)
            # fill out the rest of the pcoord array with xy positions
            coords[iseg, 1:] = x[:,:self.coord_ndim]

            # fill out new coords
            segment.pcoord[...] = coords[iseg, :]
            #segment.data["displacement"] = all_displacements[iseg]
            segment.status = segment.SEG_STATUS_COMPLETE

        return segments
