
#import matplotlib.pyplot as plt
from potentials import *
from utils import plot_potential

# from MA-REAP
#plot_potential(two_wells_complex_func, (0,100), (0,100))
plot_potential(four_wells_symmetric_func, (0,2), (0,2), granularity=0.01)
#plot_potential(four_wells_asymmetric_func, (0,2), (0,2), granularity=0.01)

# from REAP
#plot_potential(I_potential, (-2, 2), (0, 4), granularity=0.01)
#plot_potential(L_potential, (-0.5, 1.5), (-0.5, 1.5), granularity=0.01)
#plot_potential(O_potential, (-1.5, 1.5), (-1.5, 1.5), granularity=0.01)

# from TSLC
#plot_potential(ring_potential, (-3, 3), (-3, 3), granularity=0.01)

# WE 2D ODLDs
#plot_potential(we_odld_2d, (0, 10), (0, 10), granularity=0.01, vmax=20)
#plot_potential(we_odld_2d_new, (0, 1), (0, 1), granularity=0.01)

# from Gideon Simpson Julia landscapes code
# using a single x var and Python unpacking
# TODO: convert other potentials to be consistent or just unpack when using these
#plot_potential(EntropicSwitch, (-3, 3), (-3, 3), granularity=0.01, single_param=True, vmax=10)
#plot_potential(SymmetricTwoChannel, (-3, 3), (-3, 3), granularity=0.01, single_param=True, vmax=4)
#plot_potential(Muller, (-3, 3), (-3, 3), granularity=0.01, single_param=True, vmax=10)
#plot_potential(Rosenbrock, (-2, 2), (-2, 2), granularity=0.01, single_param=True, vmax=100)
#plot_potential(Zpotential, (-10, 10), (-10, 10), granularity=0.01, single_param=True, vmax=10)
#plot_potential(EntropicBox, (-1, 1), (-1, 1), granularity=0.01, single_param=True)