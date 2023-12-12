
from utils import setup_simulation, run_trajectory
from potentials import L_potential, we_odld_2d, four_wells_asymmetric
import openmm as mm
import numpy as np
import matplotlib.pyplot as plt
import mdap

traj = run_trajectory(n_steps=5000, potential=four_wells_asymmetric, initial_position=[0.2, 1, 0])

np.savetxt("test_traj5000.txt", traj)

#traj = np.loadtxt("test_traj5000.txt")

# TODO: I thought mdap and wedap could also handle ndarray input? 
# an error if I use an array directly
mpdist = mdap.MD_Plot(data_type="pdist", Xname="test_traj5000.txt", Xindex=0, 
                      Yname="test_traj5000.txt", Yindex=1, xlim=(0,2), ylim=(0,2))
mpdist.plot()
plt.show()

