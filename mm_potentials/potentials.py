import numpy as np


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

# Potentials for MA REAP

def gaussian_bivariate(c, mean, std, x, y):
    return c * np.exp(-0.5 * ((x - mean[0]) ** 2 / std[0] + (y - mean[1]) ** 2 / std[1]))


def four_wells_asymmetric_func(x, y):
    # Four corners
    cs = [-80, -25, -25, -25]
    means = [(0.2, 1),
             (1, 0.2),
             (1.8, 1),
             (1, 1.8),
             ]

    stds = [(0.008, 0.008),
            (0.008, 0.008),
            (0.008, 0.008),
            (0.008, 0.008),
            ]

    # Bridges
    cs += [-50, -50, -50, -50, -50, -50, -50, -50]

    means += [(0.4, 1),
              (0.8, 1),
              (1.2, 1),
              (1.6, 1),
              (1, 0.4),
              (1, 0.8),
              (1, 1.2),
              (1, 1.6),
              ]

    stds += [(0.04, 0.02),
             (0.04, 0.02),
             (0.04, 0.02),
             (0.04, 0.02),
             (0.02, 0.04),
             (0.02, 0.04),
             (0.02, 0.04),
             (0.02, 0.04),
             ]

    # Center compensation
    cs += [53]

    means += [(1, 1),
              ]

    stds += [(0.02, 0.02),
             ]

    potential = 0
    for c, mean, std in zip(cs, means, stds):
        potential += gaussian_bivariate(c, mean, std, x, y)

    return potential


def gaussian_bivariate_string():
    # Four corners
    cs = [-80, -25, -25, -25]
    means = [(0.2, 1),
             (1, 0.2),
             (1.8, 1),
             (1, 1.8),
             ]

    stds = [(0.008, 0.008),
            (0.008, 0.008),
            (0.008, 0.008),
            (0.008, 0.008),
            ]

    # Bridges
    cs += [-50, -50, -50, -50, -50, -50, -50, -50]

    means += [(0.4, 1),
              (0.8, 1),
              (1.2, 1),
              (1.6, 1),
              (1, 0.4),
              (1, 0.8),
              (1, 1.2),
              (1, 1.6),
              ]

    stds += [(0.04, 0.02),
             (0.04, 0.02),
             (0.04, 0.02),
             (0.04, 0.02),
             (0.02, 0.04),
             (0.02, 0.04),
             (0.02, 0.04),
             (0.02, 0.04),
             ]

    # Center compensation
    cs += ['+53']

    means += [(1, 1),
              ]

    stds += [(0.02, 0.02),
             ]

    base_string = '{c}*exp(-0.5*((x-{mean_x})^2/{std_x}+(y-{mean_y})^2/{std_y}))'

    string = ''
    for c, mean, std in zip(cs, means, stds):
        string += base_string.format(c=str(c), mean_x=str(mean[0]), mean_y=str(mean[1]), std_x=str(std[0]),
                                     std_y=str(std[1]))

    return string

four_wells_asymmetric = gaussian_bivariate_string()


def four_wells_symmetric_func(x, y):
    # Four corners
    cs = [-25, -25, -25, -25]
    means = [(0.2, 1),
             (1, 0.2),
             (1.8, 1),
             (1, 1.8),
             ]

    stds = [(0.008, 0.008),
            (0.008, 0.008),
            (0.008, 0.008),
            (0.008, 0.008),
            ]

    # Bridges
    cs += [-50, -50, -50, -50, -50, -50, -50, -50]

    means += [(0.4, 1),
              (0.8, 1),
              (1.2, 1),
              (1.6, 1),
              (1, 0.4),
              (1, 0.8),
              (1, 1.2),
              (1, 1.6),
              ]

    stds += [(0.04, 0.02),
             (0.04, 0.02),
             (0.04, 0.02),
             (0.04, 0.02),
             (0.02, 0.04),
             (0.02, 0.04),
             (0.02, 0.04),
             (0.02, 0.04),
             ]

    # Center compensation
    cs += [60]

    means += [(1, 1),
              ]

    stds += [(0.02, 0.02),
             ]

    potential = 0
    for c, mean, std in zip(cs, means, stds):
        potential += gaussian_bivariate(c, mean, std, x, y)

    return potential


four_wells_symmetric = gaussian_bivariate_string().replace('-80', '-25', 1)


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

    return -V1 - 100

    #return V1

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

def we_odld_2d(x, y):
    sigma = 0.001 ** (0.5)  # friction coefficient

    A = 2
    B = 10
    C = 0.5
    x0 = 0
    y0 = 0
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

    return -(gradx + grady)

from sympy import diff, exp, symbols, lambdify

def we_odld_2d_new(x, y):
    def calc_gradient():
        A = 50.5
        B = 49.5
        C = 10000
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
    return (x[0]**4 + x[1]**4)/20480 - 3 * np.exp(-0.01 * (x[0]+5)**2 - 0.2 * (x[1]+5)**2) - 3 * np.exp(-0.01 * (x[0]-5)**2 - 0.2 * (x[1]-5)**2) + 5 * np.exp(-0.2 * (x[0]+3*(x[1]-3)**2))/(1+np.exp(-x[0]-3)) + 5 * np.exp(-0.2 * (x[0]+3*(x[1]+3)**2))/(1+np.exp(x[0]-3)) + 3 * np.exp(-0.01 *(x[0]**2 + x[1]**2))

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
