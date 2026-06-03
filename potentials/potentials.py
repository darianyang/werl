import numpy as np
from sympy import diff, exp, symbols, lambdify
import math

# Potential definitions

########################################## EXAMPLES (NOT USED) ##########################################
def two_wells_func(x, y):
    return 2 * (x - 1) ** 2 * (x + 1) ** 2 + y ** 2


two_wells = '2*(x-1)^2*(x+1)^2 + y^2'


def two_wells_complex_func(x, y):
    return -3 * np.exp(-(x - 1) ** 2 - y ** 2) - 3 * np.exp(-(x + 1) ** 2 - y ** 2) + 15 * np.exp(
        -0.32 * (x ** 2 + y ** 2 + 20 * (x + y) ** 2)) + 0.0512 * (x ** 4 + y ** 4) + 0.4 * np.exp(-2 - 4 * y)


two_wells_complex = '-3*exp(-(x-1)^2-y^2)-3*exp(-(x+1)^2-y^2)+15*exp(-0.32*(x^2+y^2+20*(x+y)^2))+0.0512*(x^4+y^4)+0.4*exp(-2-4*y)'


#########################################################################################################

def we_odld_2d_energy(x, y):
    A = 2
    B = 2
    #B = 10
    C = 0.5
    # for -10 to 0
    # x0 = 0
    # y0 = 0
    # for 0 to 10
    x0 = 1
    y0 = 1

    twopi_by_A = 2 * np.pi / A
    half_B = B / 2

    xarg = twopi_by_A * (x - x0)
    yarg = twopi_by_A * (y - y0)

    eCx = np.exp(C * x)
    eCx_less_one = eCx - 1.0
    eCy = np.exp(C * y)
    eCy_less_one = eCy - 1.0

    potentialx = -half_B / eCx_less_one * np.cos(xarg)
    potentialy = -half_B / eCy_less_one * np.cos(yarg)
    potentialx -= np.min(potentialx)
    potentialy -= np.min(potentialy)
    return potentialx + potentialy

def we_odld_2d_grad(x, y):
    A = 2
    B = 2
    #B = 10
    C = 0.5
    # for -10 to 0
    x0 = 0
    y0 = 0
    # for 0 to 10
    # x0 = 1
    # y0 = 1
    PI = np.pi

    twopi_by_A = 2 * PI / A
    half_B = B / 2

    xarg = twopi_by_A * (x - x0)
    yarg = twopi_by_A * (y - y0)

    eCx = np.exp(C * x)
    eCx_less_one = eCx - 1.0
    eCy = np.exp(C * y)
    eCy_less_one = eCy - 1.0

    gradx = (
        half_B
        / (eCx_less_one * eCx_less_one)
        * (twopi_by_A * eCx_less_one * np.sin(xarg) + C * eCx * np.cos(xarg))
    )

    grady = (
        half_B
        / (eCy_less_one * eCy_less_one)
        * (twopi_by_A * eCy_less_one * np.sin(yarg) + C * eCy * np.cos(yarg))
    )

    #return -(gradx + grady)
    return gradx, grady
    #return gradx + grady

def we_odld_2d_new_energy(x, y):
    """
    Just the energy.
    """
    A = 50.5
    B = 49.5
    #C = 10000
    C = 10**5
    D = 51
    E = 49

    #x, y = symbols('x y')
    
    logU1 = -A * ((x-.25)**2) - A * ((y-.75)**2) - 2 * B * (x-.25) * (y-.75)
    
    logU2 = -C * (x**2) * ((1-x)**2) * (y**2) * ((1-y)**2)
    
    logU3 = -D * (x**2) - D * (y**2) + 2 * E * x * y

    # use math instead of np to be compatible with sympy differentiation later
    return np.exp(logU1) + np.exp(logU2) + (0.5 * np.exp(logU3))

def we_odld_2d_new_grad(x, y):
    def calc_gradient():
        A = 50.5
        B = 49.5
        #C = 10000
        C = 10**5
        D = 51
        E = 49

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
    grad_x, grad_y = calc_gradient()
    return grad_x(x, y) + grad_y(x, y)

def Spotential(x):
    """
    Spotential - S-shaped potential (previously referred to as Zpotential_barrier)
    
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

# This is the same potential in a different format and as a str
def Spotential_str(x, y):
    """
    Spotential - S-shaped potential.
    
    Parameters:
    x (array-like): Position x in R²
    
    Returns:
    float: The value of the potential at the given position x.
    """
    # original coefficients: 3, 5, 5, 3
    return '''(x^4 + y^4)/20480 - 3 * exp(-0.01 * (x+5)^2 - 0.2 * (y+5)^2) - \
           -60 * exp(-0.003 * (x-5)^2 - 0.2 * (y-5)^2) + \
           -35 * exp(-0.075 * (x+3*(y-3)^2))/(1+exp(-x-3)) + \
           50 * exp(-0.2 * (x+3*(y+3)^2))/(1+exp(x-3)) + \
           16 * exp(-0.01 *(x^2 + y^2))'''


