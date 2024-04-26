import numpy as np
import openmm as mm
import argparse

#import matplotlib.pyplot as plt

def run_simulation(n_steps=5000, potential='4*(x-1)^2*(x+1)^2 + y^2',
                   initial_position=[0, 0, 0], T=500, gamma=1.0, stepsize=0.02,
                   savetraj=None):
    """
    Simulate a single particle in OpenMM.

    Parameters
    ----------
    n_steps : int
        Number of steps for the trajectory. Default 5000.
    potential : str
        Input for OpenMM CustomExternalForce, the potential for particle
        simulation. Default uses a 2D ODLD double well potential with a 
        4 kT barrier height ('4*(x-1)^2*(x+1)^2 + y^2').
    initial_position : list of [x,y,z] floats
        Initial particle positions.
    T : double
        Temperature of the ODLD system in Kelvin, default 500K.
    gamma : double
        Friction coefficient which couples the system to the heat bath (ps^-1).
        Default 1.0 ps^-1, increase to increase viscosity / collision frequency.
    stepsize : double
        The step size with which to integrate the system (in ps).
        Default 0.02 ps.
    savetraj : str
        Default None and no trajectory saving, otherwise input a path to the
        save the trajectory array.
        
    Returns
    -------
    x : 3d array
        Where X and Y positions are the first 2 columns.
        The third column is Z positions if the potential is 3D.
    """
    # initialize system and add particle with a unit mass (1)
    system = mm.System()
    system.addParticle(1)
    
    # defines the potential
    force = mm.CustomExternalForce(potential)
    force.addParticle(0, [])
    system.addForce(force)
    
    # Langevin integrator with custom temperature, friction, and step size
    integrator = mm.LangevinIntegrator(T, gamma, stepsize)
    context = mm.Context(system, integrator)
    
    # initialize positions and velocities
    context.setPositions([initial_position])
    context.setVelocitiesToTemperature(T)
    
    # save coordinates
    x = np.zeros((n_steps, 3))
    for i in range(n_steps):
        x[i] = context.getState(getPositions=True).getPositions(asNumpy=True)._value
        integrator.step(1)
    
    # optionally save trajectory as text file
    if savetraj is not None:
        np.savetxt(savetraj, x)
    return x

def parse_arguments():
    parser = argparse.ArgumentParser(description='Simulate a single particle in OpenMM.')
    parser.add_argument('--n_steps', type=int, default=5000, 
                        help='Number of steps for the trajectory (default: 5000)')
    parser.add_argument('--potential', type=str, default='4*(x-1)^2*(x+1)^2 + y^2', 
                        help='Input for OpenMM CustomExternalForce, the potential for ' 
                             'particle simulation (default: "4*(x-1)^2*(x+1)^2 + y^2")')
    parser.add_argument('--initial_position', type=float, nargs='+', default=[0, 0, 0], 
                        help='Initial particle positions as list of three floats (default: [0, 0, 0])')
    parser.add_argument('--T', type=float, default=500, 
                        help='Temperature of the ODLD system in Kelvin (default: 500K)')
    parser.add_argument('--gamma', type=float, default=1.0, 
                        help='Friction coefficient which couples the system to the heat bath (ps^-1)' 
                             '(default: 1.0 ps^-1)')
    parser.add_argument('--stepsize', type=float, default=0.02, 
                        help='The step size for integrating the system (in ps) (default: 0.02 ps)')
    parser.add_argument('--savetraj', type=str, default=None, 
                        help='Path to save the trajectory array (default: None)')
    
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_arguments()
    run_simulation(**vars(args))