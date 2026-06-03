import numpy as np
from sympy import diff, exp, symbols, lambdify
import math

### converted from Julia
### from: https://github.com/gideonsimpson/TestLandscapes.jl/blob/master/src/potentials2D.jl

def EntropicSwitch(x):
    """
    EntropicSwitch - Entropically switching potential with three local minima. It is symmetric about x=0
    
    Parameters:
    x (array-like): Position x in R²
    
    Returns:
    float: The value of the potential at the given position x.
    """
    return (3 * np.exp(-x[0]**2 - (x[1]-1/3)**2)
     - 3 * np.exp(-x[0]**2 - (x[1]-5/3)**2)
     - 5 * np.exp(-(x[0]-1)**2 - x[1]**2)
     - 5 * np.exp(-(x[0]+1)**2 - x[1]**2)
     + 1/5 * x[0]**4 + 1/5 * (x[1]-1/3)**4)

def SymmetricTwoChannel(x):
    """
    SymmetricTwoChannel - Double well potential in 2D with two, symmetric channels joining them.
    
    Parameters:
    x (array-like): Position x in R²
    
    Returns:
    float: The value of the potential at the given position x.
    """
    return 1/6 * (4 * (1-x[0]**2-x[1]**2)**2 + 2 *(x[0]**2-2)**2
    + ((x[0]+x[1])**2 - 1 )**2 + ((x[0]-x[1])**2 - 1 )**2)

def Muller(x):
    """
    Muller - The Muller potential with three distinct minima and highly asymmetric.
    
    Parameters:
    x (array-like): Position x in R²
    
    Returns:
    float: The value of the potential at the given position x.
    """
    aa = (-1, -1, -6.5, 0.7)
    bb = (0., 0., 11., 0.6)
    cc = (-10., -10., -6.5, 0.7)
    AA = (-200., -100., -170., 15.)
    XX = (1., 0., -0.5, -1.)
    YY = (0., 0.5, 1.5, 1.)

    return ( AA[0]*np.exp(aa[0]*(x[0]-XX[0])**2+bb[0]*(x[0]-XX[0])*(x[1]-YY[0])+cc[0]*(x[1]-YY[0])**2)
             + AA[1]*np.exp(aa[1]*(x[0]-XX[1])**2+bb[1]*(x[0]-XX[1])*(x[1]-YY[1])+cc[1]*(x[1]-YY[1])**2)
             + AA[2]*np.exp(aa[2]*(x[0]-XX[2])**2+bb[2]*(x[0]-XX[2])*(x[1]-YY[2])+cc[2]*(x[1]-YY[2])**2)
             + AA[3]*np.exp(aa[3]*(x[0]-XX[3])**2+bb[3]*(x[0]-XX[3])*(x[1]-YY[3])+cc[3]*(x[1]-YY[3])**2))

def Rosenbrock(x, a=1.0, b=100.0):
    """
    Rosenbrock - Banana-shaped Rosenbrock potentials with global minimum located at (a, a²).
    
    Parameters:
    x (array-like): Position x in R²
    a (float, optional): Rosenbrock parameter. Default is 1.0.
    b (float, optional): Rosenbrock parameter. Default is 100.0.
    
    Returns:
    float: The value of the potential at the given position x.
    """
    return (a-x[0])**2 + b * (x[1]-x[0]**2)**2

def Zpotential(x):
    """
    Zpotential - Z-shaped potential.
    
    Parameters:
    x (array-like): Position x in R²
    
    Returns:
    float: The value of the potential at the given position x.
    """
    # original coefficients: 3, 5, 5, 3
    return (x[0]**4 + x[1]**4)/20480 - 3 * np.exp(-0.01 * (x[0]+5)**2 - 0.2 * (x[1]+5)**2) - \
            3 * np.exp(-0.01 * (x[0]-5)**2 - 0.2 * (x[1]-5)**2) + \
            5 * np.exp(-0.2 * (x[0]+3*(x[1]-3)**2))/(1+np.exp(-x[0]-3)) + \
            5 * np.exp(-0.2 * (x[0]+3*(x[1]+3)**2))/(1+np.exp(x[0]-3)) + \
            3 * np.exp(-0.01 *(x[0]**2 + x[1]**2))
def Zpotential_barrier(x):
    """
    Zpotential - Z-shaped potential.
    
    Parameters:
    x (array-like): Position x in R²
    
    Returns:
    float: The value of the potential at the given position x.
    """
    # original coefficients: 3, 5, 5, 3
    return (x[0]**4 + x[1]**4)/20480 - 3 * np.exp(-0.01 * (x[0]+5)**2 - 0.2 * (x[1]+5)**2) - \
            -60 * np.exp(-0.003 * (x[0]-5)**2 - 0.2 * (x[1]-5)**2) + \
            -35 * np.exp(-0.075 * (x[0]+3*(x[1]-3)**2))/(1+np.exp(-x[0]-3)) + \
            50 * np.exp(-0.2 * (x[0]+3*(x[1]+3)**2))/(1+np.exp(x[0]-3)) + \
            16 * np.exp(-0.01 *(x[0]**2 + x[1]**2))
            # originally smaller barrier near middle
            #3 * np.exp(-0.01 *(x[0]**2 + x[1]**2))

def EntropicBox(x):
    """
    EntropicBox - A potential concentrated in [0,1]² with internal entropic barriers. Formulated by D. Aristoff (Colorado State).
    
    Parameters:
    x (array-like): Position x in R²
    
    Returns:
    float: The value of the potential at the given position x.
    """
    c = (50.5, 49.5, 10**5, 51, 49)
    return np.exp(-(c[0]*(x[0]-0.25)**2+c[0]*(x[1]-0.75)**2+2*c[1]*(x[0]-0.25)*(x[1]-0.75))) + np.exp(-c[2]*(x[0]**2*(1-x[0])**2*x[1]**2*(1-x[1])**2)) + 0.5*np.exp(-(c[3]*x[0]**2+c[3]*x[1]**2-2*c[4]*x[0]*x[1]))

