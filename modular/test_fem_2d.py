#!/usr/bin/env python3

"""
Test 2d fem code for Poisson equation

Amuthan Ramabathiran
September 2022
"""

import fem_2d_modular as fem


n_side = 20
n_quad = 2
nodes, elements, dbc = fem.create_mesh_unit_square(n_side=n_side)
uh = fem.solve_bvp(nodes, elements, dbc, n_quad)
err_L2 = fem.compute_L2_error_centers(nodes, elements, uh)
print(f'L2 error = {err_L2}')
fem.plot_fem_soln(nodes, uh, n_plot=21)
