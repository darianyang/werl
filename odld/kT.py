"""
Calculate pmin from input kT. 
"""

import numpy as np
import sys

# first arg
kT = float(sys.argv[1])
pmax = 0.1

# kT = -ln(Pmin/Pmax)
# Pmin = -e^kT * Pmax

pmin = np.exp(kT) * pmax

print(pmin)
