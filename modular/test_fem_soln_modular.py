"""
Test 1 of fem_1d_modular.py
Comparison of FEM and exact solutions
"""

import fem_1d_modular as fem

n_elt = 10
n_quad = 3

nodes, elements, dbc = fem.create_mesh_1d_uniform(n_elt)
uh = fem.solve_bvp(nodes, elements, dbc, n_quad)
fem.plot_fem_soln(nodes, elements, uh)
