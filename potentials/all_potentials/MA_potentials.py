import numpy as np
from sympy import diff, exp, symbols, lambdify
import math

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


