
import matplotlib.pyplot as plt
from potentials import *
import sympy as sp

def calculate_gradient(func, x, y, epsilon=1e-6):
    '''
    Use finite differencing to approx the gradient of each energy potential.

    Finite differencing is a numerical technique used to approximate the derivative
    of a function at a particular point. The basic idea is to estimate the rate of 
    change of a function by considering small changes in its input. 

    Parameters
    ----------
    func : Python function
        Potential energy function.
    x : array
        X position(s) of the particle.
    y : array
        Y position(s) of the particle.
    epsilon : float
        The step size, used to perturb the input values (x, y) in both positive 
        and negative directions to approximate the partial derivatives with 
        respect to x and y. Default 1e-6.
    '''
    grad_x = (func(x + epsilon, y) - func(x - epsilon, y)) / (2 * epsilon)
    grad_y = (func(x, y + epsilon) - func(x, y - epsilon)) / (2 * epsilon)
    return grad_x, grad_y

# TODO: calc using sympy and compare, may be better to save and use the analytical solution
def calculate_gradient_sympy(func, x, y):
    '''
    Parameters
    ----------
    func : Python function
        Potential energy function.
    x : array
        X position(s) of the particle.
    y : array
        Y position(s) of the particle.
    '''
    # Define symbolic variables
    x_sym, y_sym = sp.symbols('x y')

    # Define the energy landscape function symbolically
    energy_landscape_sym = func(x_sym, y_sym)

    # Calculate the gradient symbolically
    gradient_sym = [sp.diff(energy_landscape_sym, var) for var in (x_sym, y_sym)]

    # Define a function to evaluate the symbolic expression
    calculate_gradient_lambda = sp.lambdify((x_sym, y_sym), gradient_sym, 'numpy')

    # Evaluate the gradient at the given positions
    gradients = calculate_gradient_lambda(x, y)
    
    return gradients

def plot_potential(potential, xlim=None, ylim=None, granularity=1, vmax=None, vmin=None,
                   single_param=False, calc_grad=False):
    '''
    Plots potential based on analytic function.
    '''
    x = np.arange(*xlim, granularity)
    y = np.arange(*ylim, granularity)
    X, Y = np.meshgrid(x, y)  # grid of point

    # calc gradient if needed
    if calc_grad:
        gradX, gradY = calculate_gradient(potential, X, Y)
        Z = gradX + gradY
        #Z = calculate_gradient_sympy(potential, X, Y)

    # Python unpacking for multi dimensional output using single input
    elif single_param:
        Z = potential([X, Y])  # evaluation of the function on the grid
    else:
        Z = potential(X, Y)  # evaluation of the function on the grid

    #im = plt.imshow(Z, cmap=plt.cm.jet, vmax=vmax, #interpolation='None',
    #                extent=[xlim[0], xlim[1], ylim[0], ylim[1]])  # drawing the function

    # note that to accurately get the potential needs to be plotted as negative
    # TODO: not certain why yet
    #im = plt.pcolormesh(x, y, np.abs(Z), cmap=plt.cm.jet, vmax=vmax, vmin=vmin)
    im = plt.pcolormesh(x, y, Z, cmap=plt.cm.jet, vmax=vmax, vmin=vmin)
    #im = plt.pcolormesh(x, y, -np.log((np.min(np.abs(Z))+np.abs(Z))), cmap=plt.cm.jet, vmax=vmax, vmin=vmin)
    #im = plt.pcolormesh(x, y, -np.log(Z/np.max(Z)), cmap=plt.cm.jet, vmax=vmax, vmin=vmin)
    #plt.xlim(xlim)
    #plt.ylim(ylim)

    # adding the Contour lines with labels
    # cset = plt.contour(Z, [-2, -1, 0, 1, 2, 3], linewidths=2, cmap=plt.cm.binary, extent=[xlim[0], xlim[1], ylim[1], ylim[0]])
    # plt.clabel(cset, inline=True,fmt='%1.1f', fontsize=10)
    plt.colorbar(im)  # adding the colobar on the right
    plt.show()

# from MA-REAP
#plot_potential(two_wells_complex_func, (0,100), (0,100))
#plot_potential(four_wells_symmetric_func, (0,2), (0,2), granularity=0.01)
#plot_potential(four_wells_asymmetric_func, (0,2), (0,2), granularity=0.01)

# from REAP
#plot_potential(I_potential, (-2, 2), (0, 4), granularity=0.01)
#plot_potential(L_potential, (-1, 3), (-1, 3), granularity=0.01)
#plot_potential(L_potential, (-1, 3), (-1, 3), granularity=0.01, calc_grad=True)
#plot_potential(O_potential, (-1.5, 1.5), (-1.5, 1.5), granularity=0.01)

# from TSLC
#plot_potential(ring_potential, (-3, 3), (-3, 3), granularity=0.01)

# WE 2D ODLDs
#plot_potential(we_odld_2d_energy, (0, 10), (0, 10), granularity=0.01)
#plot_potential(we_odld_2d, (-10, 0), (-10, 0), granularity=0.01, vmax=15)
#plot_potential(we_odld_2d_energy, (-10, 0), (-10, 0), granularity=0.01, vmax=15)
#plot_potential(we_odld_2d, (0, 10), (0, 10), granularity=0.01, vmin=-15)
#plot_potential(we_odld_2d_new_grad, (0, 1), (0, 1), granularity=0.01)
plot_potential(we_odld_2d_new_energy, (0, 1), (0, 1), granularity=0.01, calc_grad=True, vmin=-10, vmax=10)

# from Gideon Simpson Julia landscapes code
# using a single x var and Python unpacking
# TODO: convert other potentials to be consistent or just unpack when using these
#plot_potential(EntropicSwitch, (-3, 3), (-3, 3), granularity=0.01, single_param=True, vmax=10)
#plot_potential(SymmetricTwoChannel, (-3, 3), (-3, 3), granularity=0.01, single_param=True, vmax=4)
#plot_potential(Muller, (-3, 3), (-3, 3), granularity=0.01, single_param=True, vmax=10)
#plot_potential(Rosenbrock, (-2, 2), (-2, 2), granularity=0.01, single_param=True, vmax=100)
#plot_potential(Zpotential, (-10, 10), (-10, 10), granularity=0.1, single_param=True, vmax=10)
#plot_potential(EntropicBox, (-1, 1), (-1, 1), granularity=0.01, single_param=True)
