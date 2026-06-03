import numpy as np
from sympy import diff, exp, symbols, lambdify
import math

# # Potential for TSLC
# def circular_potential_func(v, r=2, c=-250, a=-10):
#     '''
#     Computes circular potential value at point v = (x, y ,z).
#     It is assumed the circle is centered at the origin.

#     Args
#     -----------------
#     v (array-like): 3d coordinates of point in cartesian coordinates.
#     r (float): circle radius.
#     c (float): energy minimum at the circle. Should be negative for the circle to be a stable equilibrium line.
#     a (float): exponential scaling factor (adjusts how quickly the energy increases far form the circle). Should be negative.

#     Returns
#     -----------------
#     potential (float): potential value at v.
#     '''
#     x, y, z = v
#     circle_radius = r
#     d_squared = z ** 2 + np.square(np.sqrt(x ** 2 + y ** 2) - circle_radius)
#     potential = c * np.exp(a * d_squared)

#     return potential


def circular_potential_string(r=2, c=-250, a=-10):
    d_squared = "z^2 + (sqrt(x^2 + y^2) - {circle_radius})^2".format(circle_radius=r)
    circular_potential = "{c}*exp({a}*({d}))".format(c=c, a=a, d=d_squared)

    return circular_potential


circular_potential = circular_potential_string()

# from TSLC paper
def ring_potential(x, y):
    #Potential Parameters:
    a=10

    b=250

    rad=2 #rad is radius of circle

    #distance to circle
    d = np.square(np.sqrt(np.square(x)+np.square(y)) - rad)
    V = -b*np.exp(-d*a)

    return V

