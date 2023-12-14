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

def potential_energy_1d(x, A=2, B=10, C=0.5, x0=0):
    twopi_by_A = 2 * np.pi / A
    half_B = B / 2

    xarg = twopi_by_A * (x - x0)
    eCx = np.exp(C * x)
    eCx_less_one = eCx - 1.0

    potential = half_B * (eCx_less_one**2 - np.cos(xarg))
    return potential

x_values = np.linspace(-5, 5, 50)
#x_values = np.arange(0, 10, 0.1)
potential_values_1d = potential_energy_1d(x_values)
plt.plot(potential_values_1d)
plt.plot(calculate_gradient_1d(potential_energy_1d, x_values))
plt.show()

# def potential_energy_2d(x, y, A=2, B=10, C=0.5, x0=0, y0=0):
#     return potential_energy_1d(x, A, B, C, x0) + potential_energy_1d(y, A, B, C, y0)

# # Example usage
# x_values = np.linspace(-5, 5, 100)
# y_values = np.linspace(-5, 5, 100)
# X, Y = np.meshgrid(x_values, y_values)
# potential_values_2d = potential_energy_2d(X, Y)
# plt.pcolormesh(potential_energy_2d, x_values, y_values)
# plt.show()