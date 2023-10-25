
#import matplotlib.pyplot as plt
from potentials import *
from utils import plot_potential

#plot_potential(two_wells_complex_func, (0,100), (0,100))
#plot_potential(four_wells_symmetric_func, (0,2), (0,2), granularity=0.01)
#plot_potential(four_wells_asymmetric_func, (0,2), (0,2), granularity=0.01)
#plot_potential(I_potential, (-2, 2), (0, 4), granularity=0.01)
#plot_potential(L_potential, (-0.5, 1.5), (-0.5, 1.5), granularity=0.01)
#plot_potential(O_potential, (-1.5, 1.5), (-1.5, 1.5), granularity=0.01)
#plot_potential(ring_potential, (-3, 3), (-3, 3), granularity=0.01)
#plot_potential(we_odld_2d, (0, 10), (0, 10), granularity=0.01, vmax=20)
plot_potential(we_odld_2d_new, (0, 1), (0, 1), granularity=0.01)
