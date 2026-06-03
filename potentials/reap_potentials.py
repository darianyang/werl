import numpy as np
from sympy import diff, exp, symbols, lambdify
import math

# potentials from REAP paper

def I_potential(x, y):
    # parameters in Mueller potential

    aa = [-1.5, -10, -1.5] # inverse radius in x
    bb = [0, 0, 0] # radius in xy
    cc = [-20, -1, -20] # inverse radius in y
    AA = [-80, -80, -80] # strength

    XX = [0, 0, 0] # center_x
    YY = [0.5, 2, 3.5] # center_y

    V1 = AA[0]*np.exp(aa[0] * np.square(x-XX[0]) + bb[0] * (x-XX[0]) * (y-YY[0]) +cc[0]*np.square(y-YY[0]))

    for j in range(1,3):
        V1 =  V1 + AA[j]*np.exp(aa[j]*np.square(x-XX[j]) + bb[j]*(x-XX[j])*(y-YY[j]) + cc[j]*np.square(y-YY[j]))

    return V1

def L_potential(x, y):
    # parameters in Mueller potential

    aa = [-2, -20, -20, -20, -20] # inverse radius in x
    bb = [0, 0, 0, 0, 0] # radius in xy
    cc = [-20, -20, -2, -20, -20] # inverse radius in y
    AA = 30*[-200, -120, -200, -80, -80] # strength

    XX = [1, 0, 0, 0, 0.4] # center_x
    YY = [0, 0, 1, 0.4, 0] # center_y

    V1 = AA[0]*np.exp(aa[0] * np.square(x-XX[0]) + bb[0] * (x-XX[0]) * (y-YY[0]) +cc[0]*np.square(y-YY[0]))
    for j in range(1,5):
            V1 =  V1 + AA[j]*np.exp(aa[j]*np.square(x-XX[j]) + bb[j]*(x-XX[j])*(y-YY[j]) + cc[j]*np.square(y-YY[j]))

    # shift to a scale similar to standard ODLD potential
    #V1 = (V1 / 5) + 15
    #V1 += 1
    # make upper limit huge just like ODLD
    #V1[V1 >= 0] = 200000

    #return -V1 - 100

    return V1

def O_potential(x, y):
    # parameters in Mueller potential

    aa = [-1] # inverse radius in x
    bb = [0] # radius in xy
    cc = [-1] # inverse radius in y
    AA = 3*[-200] # strength

    XX = [0] # center_x
    YY = [0] # center_y

    V1 = AA[0]*np.exp(aa[0] * np.square(x-XX[0]) + bb[0] * (x-XX[0]) * (y-YY[0]) +cc[0]*np.square(y-YY[0]))
    return V1

