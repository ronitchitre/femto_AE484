"""
A very simple implementation of the finite element method in 1D
Solve -u'' = f on (0,1) for specfied f
Boundary conditions: u(0) = u(1) = 0
Piecewise-linear finite element approximation over a uniform mesh is used.
Most of the details are hard-coded.
"""

import numpy as np
import matplotlib.pyplot as plt

pi = np.pi


def f(x):
    return 4*pi*pi*np.sin(2*pi*x)


def u_exact(x):
    return np.sin(2*pi*x)


def compute_stiffness(n_elt):
    N = n_elt - 1
    h = 1/n_elt

    K = np.zeros((N, N))
    for i in range(N):
        K[i, i] = 2/h
        if i < (N-1):
            K[i, (i+1)] = -1/h
        if i > 0:
            K[i, (i-1)] = -1/h

    return K


def compute_load_vector(n_elt):
    N = n_elt - 1
    h = 1/n_elt

    F = np.zeros(N)
    for i in range(N):
        xi = (i + 1)*h
        F[i] = h*f(xi)

    return F


def solve_bvp(n_elt):
    K = compute_stiffness(n_elt)
    F = compute_load_vector(n_elt)
    U = np.linalg.solve(K, F)
    return U


def plot_fem_soln(n_elt):
    U = solve_bvp(n_elt)

    h = 1/n_elt
    N = n_elt - 1
    x_nodes = np.array(
        [i*h for i in range(N + 2)]
    )
    u_nodes = np.concatenate((
        np.array([0.0]),
        U,
        np.array([0.0])
    ))

    xs = np.linspace(0, 1, 40)
    us_exact = u_exact(xs)

    plt.plot(xs, us_exact, '-', color='gray', lw=6)
    plt.plot(x_nodes, u_nodes, 'bo-', lw=2, markersize=3, label='FEM')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.show()


def fe_interpolate(U, x):
    N = len(U)
    h = 1/(N + 1)

    u_nodes = np.concatenate((
        np.array([0.0]),
        U,
        np.array([0.0])
    ))

    elt_id = int(x/h)
    if elt_id < 0:
        elt_id = 0
    if elt_id > N:
        elt_id = N
    uh = u_nodes[elt_id] + (u_nodes[elt_id + 1] -
                            u_nodes[elt_id])*(x - elt_id*h)/h
    return uh


def compute_L2_error(n_elt, n_quad=100):
    U = solve_bvp(n_elt)
    xs = np.linspace(0, 1, (n_quad + 1))
    us_exact = u_exact(xs)
    us_fem = np.array([fe_interpolate(U, x) for x in xs])
    err_L2 = np.mean((us_exact - us_fem)**2)
    return np.sqrt(err_L2)
