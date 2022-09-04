"""
Test 2 of fem_1d_v0.py
Convergence of FEM solution with mesh refinement
"""

import numpy as np
import matplotlib.pyplot as plt
import fem_1d_v0 as fem0

n_elts = np.array([4, 9, 16, 25, 36, 49, 64, 81, 100])
errs_L2 = np.array([fem0.compute_L2_error(n_elt) for n_elt in n_elts])

plt.loglog(n_elts, errs_L2, 'ko-', lw=2)
plt.loglog(n_elts, (1/n_elts)**2, 'ro--', lw=2)
plt.xlabel('Number of elements')
plt.ylabel('L2 error')
plt.show()
