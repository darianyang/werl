
import matplotlib.pyplot as plt
from potentials import *
import sympy as sp
import numpy as np

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
                   single_param=False, calc_grad=False, cmap='jet'):
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

    # note that to accurately get the potential needs to be plotted as negative
    # TODO: not certain why yet
    try:
        im = plt.pcolormesh(x, y, Z, cmap=cmap, vmax=vmax, vmin=vmin)
    except:
        print("Array passed as tuple of arrays, combined as dist")
        Z = np.sqrt(Z[0]**2 + Z[1]**2)
        im = plt.pcolormesh(x, y, Z, cmap=cmap, vmax=vmax, vmin=vmin)
    
    plt.xlim(xlim)
    plt.ylim(ylim)

    # adding the Contour lines with labels
    # cset = plt.contour(Z, [-2, -1, 0, 1, 2, 3], linewidths=2, cmap=plt.cm.binary, extent=[xlim[0], xlim[1], ylim[1], ylim[0]])
    # plt.clabel(cset, inline=True,fmt='%1.1f', fontsize=10)
    plt.colorbar(im)  # adding the colobar on the right

# WE 2D ODLDs

# Plotting each potential used for resampler evaluation

# S-potential
#plt.figure()
#plot_potential(Spotential, (-30, 30), (-30, 30), granularity=0.1, single_param=True, vmin=0, vmax=25, cmap='viridis')
#plt.savefig("S-potential.pdf")

# 2D ODLD 
plt.figure()
#plot_potential(we_odld_2d_energy, (-10, 0), (-10, 0), granularity=0.01, vmax=10, calc_grad=True)
plot_potential(we_odld_2d_grad, (-10, 0), (-10, 0), granularity=0.01, vmin=-8, vmax=8, cmap='viridis')
plt.savefig("ODLD_2d_grad.pdf")


