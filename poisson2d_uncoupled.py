#!/usr/bin/env python3

"""
Test 2d object oriented fem code for Poisson equation

Amuthan Ramabathiran
September 2022
"""

import femtolib as femto
import mesher
import elements
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

pi = np.pi


def f(i_field, xs):
    if i_field == 0:
        if xs.ndim > 1:
            x = xs[:, 0]
            y = xs[:, 1]
            return 8*pi*pi*np.sin(2*pi*x)*np.sin(2*pi*y)
        else:
            x = xs[0]
            y = xs[1]
            return 8*pi*pi*np.sin(2*pi*x)*np.sin(2*pi*y)
    elif i_field == 1:
        if xs.ndim > 1:
            x = xs[:, 0]
            y = xs[:, 1]
            return 8*pi*pi*np.sin(2*pi*x)*np.cos(2*pi*y)
        else:
            x = xs[0]
            y = xs[1]
            return 8*pi*pi*np.sin(2*pi*x)*np.cos(2*pi*y)


def u_exact(i_field, x, y):
    if i_field == 0:
        return np.sin(2*pi*x)*np.sin(2*pi*y)
    elif i_field == 1:
        return np.sin(2*pi*x)*np.cos(2*pi*y)


def in_dbc(i_field, x, y):
    tol = 1e-6
    if i_field == 0:
        if (   abs(x) <= tol or abs(1 - x) <= tol
            or abs(y) <= tol or abs(1 - y) <= tol ):
            return True, 0.0
        else:
            return False, None
    elif i_field == 1:
        if abs(x) <= tol or abs(1 - x) <= tol:
            return True, 0.0
        else:
            return False, None


class Poisson2D(femto.Model):
    def __init__(self, fields=None, source=None, exact=None):
        super().__init__(fields, exact)
        self.source = source

    def stiffness_kernel(self, i, j, x, u, v, grad_u, grad_v):
        if i == 0 and j == 0:
            return np.dot(grad_u, grad_v)
        elif i == 1 and j == 1:
            return np.dot(grad_u, grad_v)
        else:
            return 0.0

    def load_kernel(self, i, x, v, grad_v):
        if i == 0:
            return self.source(i, x)*v
        elif i == 1:
            return self.source(i, x)*v

    def plot(self, i_field=0, n_plot=10):
        if i_field == 0 or i_field == 1:
            u = self.fields[i_field]

            if self.exact is not None:
                xs = np.linspace(0, 1, n_plot)
                ys = np.linspace(0, 1, n_plot)
                xs, ys = np.meshgrid(xs, ys)
                us_exact = self.exact(i_field, xs, ys)

            plt.figure(figsize=(6, 6))
            ax = plt.axes(projection='3d')

            surf = ax.plot_trisurf(u.mesh.nodes[:, 0], u.mesh.nodes[:, 1],
                                   u.dof, color='red', label='FEM')
            # The next two lines may not be necessary
            surf._facecolors2d = surf._facecolors
            surf._edgecolors2d = surf._edgecolors

            if self.exact is not None:
                surf = ax.plot_wireframe(xs, ys, us_exact,
                                         linewidth=1, color='black',
                                         label='Exact')
                # The next two lines may not be necessary
                surf._facecolors2d = surf._facecolors
                surf._edgecolors2d = surf._edgecolors

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u')
            ax.legend()
            plt.show()


if __name__ == '__main__':
    n_side = 20
    quad_order = 2
    n_plot = 41

    mesh = mesher.UnitSquareTri(n_side=n_side)
    ref_triangle = elements.TriangleP1(quad_order=quad_order)

    n_dof = len(mesh.nodes)
    u1h = femto.FunctionSpace(mesh, ref_triangle, n_dof, idx=0)
    u2h = femto.FunctionSpace(mesh, ref_triangle, n_dof, idx=1)

    fields = [u1h, u2h]
    model = Poisson2D(fields, source=f, exact=u_exact)
    model.dirichlet = in_dbc

    model.solve()
    model.plot(i_field=0, n_plot=n_plot)
    model.plot(i_field=1, n_plot=n_plot)
