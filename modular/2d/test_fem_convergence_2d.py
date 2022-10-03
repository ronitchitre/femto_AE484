#!/usr/bin/env python3

"""
Test convergence of fem solution for 2d Poisson equation

Amuthan Ramabathiran
September 2022
"""

import fem_2d_modular as fem
import numpy as np
import matplotlib.pyplot as plt

pi = np.pi


def f(x, y):
    return 8*pi*pi*np.sin(2*pi*x)*np.sin(2*pi*y)


def u_exact(x, y):
    return np.sin(2*pi*x)*np.sin(2*pi*y)


def in_dbc(x, y):
    tol = 1e-6
    if (   abs(x) <= tol or abs(1 - x) <= tol
        or abs(y) <= tol or abs(1 - y) <= tol ):
        return True, 0.0
    else:
        return False, None


n_quad = 3
n_elts = np.array([4, 10, 20, 40, 100, 200])
errs_L2 = []

for n_elt in n_elts:
    nodes, elements, bdy = fem.create_mesh_unit_square(n_elt)
    dbc = fem.apply_bc(nodes, elements, bdy, in_dbc)
    nbc = None

    uh = fem.solve_bvp(nodes, elements, dbc, nbc, n_quad, f)

    err_L2 = fem.compute_L2_error_centers(nodes, elements, uh, u_exact)
    errs_L2.append(err_L2)
    print(f'{n_elt}: {err_L2}')

errs_L2 = np.array(errs_L2)

plt.loglog(np.sqrt(2)/n_elts, errs_L2, 'ko-', lw=2, label='FEM')
plt.loglog(np.sqrt(2)/n_elts, (1/n_elts)**2, 'ro--', lw=2, label='h^2')
plt.xlabel('Mesh size (h)')
plt.ylabel('L2 error')
plt.legend()
plt.show()
