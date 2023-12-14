import numpy as np
import matplotlib.pyplot as plt

def calculate_gradient_1d(func, x, epsilon=1e-6):
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
    epsilon : float
        The step size, used to perturb the input values (x, y) in both positive 
        and negative directions to approximate the partial derivatives with 
        respect to x and y. Default 1e-6.
    '''
    grad = (func(x + epsilon) - func(x - epsilon)) / (2 * epsilon)
    return grad

def odld_1d_grad(x, A=2, B=10, C=0.5, x0=0):
    twopi_by_A = 2 * np.pi / A
    half_B = B / 2

    xarg = twopi_by_A * (x - x0)

    eCx = np.exp(C * x)
    eCx_less_one = eCx - 1.0

    grad = (
        half_B
        / (eCx_less_one * eCx_less_one)
        * (twopi_by_A * eCx_less_one * np.sin(xarg) + C * eCx * np.cos(xarg))
    )
    
    return grad

def odld_1d_potential(x, A=2, B=10, C=0.5, x0=0):
    twopi_by_A = 2 * np.pi / A
    half_B = B / 2

    xarg = twopi_by_A * (x - x0)

    eCx = np.exp(C * x)
    eCx_less_one = eCx - 1.0

    potential = -half_B / eCx_less_one * np.cos(xarg)
    
    return potential

x = np.arange(-10, 10, 0.1)
plt.plot(x, odld_1d_grad(x), linewidth=3, color="k")
plt.plot(x, odld_1d_potential(x), linewidth=3)
plt.plot(x, calculate_gradient_1d(odld_1d_potential, x), linestyle='dotted', color="red", linewidth=3)
plt.ylim(-30, 30)
plt.show()