"""
Test 2 of fem_1d_modular.py
Convergence of FEM solution with mesh refinement
"""

import numpy as np
import matplotlib.pyplot as plt
import fem_1d_modular as fem


n_quad = 3
n_test = 101

n_elts = np.array([4, 9, 16, 25, 36, 49, 64, 81, 100])
errs_L2 = []
for n_elt in n_elts:
    nodes, elements, dbc = fem.create_mesh_1d_uniform(n_elt)
    uh = fem.solve_bvp(nodes, elements, dbc, n_quad)
    errs_L2.append(fem.compute_L2_error(nodes, elements, uh, n_test))
errs_L2 = np.array(errs_L2)

plt.loglog(1/n_elts, errs_L2, 'ko-', lw=2, label='FEM')
plt.loglog(1/n_elts, (1/n_elts)**2, 'ro--', lw=2, label='h^2')
plt.xlabel('Mesh size (h)')
plt.ylabel('L2 error')
plt.legend()
plt.show()
