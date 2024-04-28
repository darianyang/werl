import numpy as np
import openmm as mm
import argparse

# TODO: add option to select a potential by name and import other potentials
class ParticleSimulator:
    '''
    ODLD simulations in OpenMM.
    '''
    def __init__(self, n_steps=5000, potential='4*(x-1)^2*(x+1)^2+y^2',
                 initial_position=[0, 0, 0], T=500, gamma=1.0, step_size=0.02,
                 save_steps=1, save_traj=None, random_seed=None):
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
        '''
        self.n_steps = n_steps
        self.potential = potential
        self.initial_position = initial_position
        self.T = T
        self.gamma = gamma
        self.step_size = step_size
        self.save_steps = save_steps
        self.save_traj = save_traj
        self.random_seed = random_seed

    def run_simulation(self):
        '''
        Simulate a single particle in OpenMM.

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
        force = mm.CustomExternalForce(self.potential)
        force.addParticle(0, [])
        system.addForce(force)
        
        # Langevin integrator with custom temperature, friction, and step size
        integrator = mm.LangevinIntegrator(self.T, self.gamma, self.step_size)
        context = mm.Context(system, integrator)
        
        # initialize positions and velocities
        context.setPositions([self.initial_position])
        context.setVelocitiesToTemperature(self.T, randomSeed=self.random_seed)
        
        # save coordinates
        x = np.zeros((self.n_steps, 3))
        for i in range(self.n_steps):
            x[i] = context.getState(getPositions=True).getPositions(asNumpy=True)._value
            integrator.step(self.save_steps)
        
        # optionally save trajectory as text file
        if self.save_traj is not None:
            np.savetxt(self.save_traj, x)
        return x

def parse_arguments():
    parser = argparse.ArgumentParser(description='Simulate a single particle in OpenMM.')
    parser.add_argument('--n-steps', 
                        type=int, default=5000, store='n_steps',
                        help='Number of steps for the trajectory (default: 5000)')
    parser.add_argument('--potential', 
                        type=str, default='4*(x-1)^2*(x+1)^2 + y^2', store='potential',
                        help='Input for OpenMM CustomExternalForce, the potential for ' 
                             'particle simulation (default: "4*(x-1)^2*(x+1)^2 + y^2")')
    parser.add_argument('--initial-position', 
                        type=float, nargs='+', default=[0, 0, 0], store='initial_position',
                        help='Initial particle positions as list of three floats (default: [0, 0, 0])')
    parser.add_argument('--T', 
                        type=float, default=500, store='T',
                        help='Temperature of the ODLD system in Kelvin (default: 500K)')
    parser.add_argument('--gamma', 
                        type=float, default=1.0, store='gamma',
                        help='Friction coefficient which couples the system to the heat bath (ps^-1)' 
                             '(default: 1.0 ps^-1)')
    parser.add_argument('--step-size', 
                        type=float, default=0.02, store='step_size',
                        help='The step size for integrating the system (in ps) (default: 0.02 ps)')
    parser.add_argument('--save-steps', 
                        type=float, default=1, store='save_steps',
                        help='Integrator steps before save, default 1. Increase to save less frequently.')
    parser.add_argument('--save-traj',
                        type=str, default=None, store='save_traj',
                        help='Path to save the trajectory array (default: None)')
    parser.add_argument('--random-seed',
                        type=int, default=None, store='random_seed',
                        help='Path to save the trajectory array (default: None)')
    
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_arguments()
    ParticleSimulator(**vars(args)).run_simulation()