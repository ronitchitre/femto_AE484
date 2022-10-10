#!/usr/bin/env python3

"""
Poisson equation on unstructured triangular mesh.

Amuthan Ramabathiran
October 2022
"""

import femtolib as femto
import mesher
import elements
import numpy as np
import triangle as tr
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def f(i, xs):
    if i == 0:
        if xs.ndim == 1:
            return 1.0
        else:
            return np.ones(len(xs))


def create_mesh(plot=True):
    vertices = [[0, 0], [1, 0], [2, 1], [0.5, 1]]
    t = tr.triangulate({'vertices': vertices}, 'qa0.005')

    if plot:
        ax = plt.axes()
        tr.plot(ax, **t)

    return t


def in_dbc(i, x, y):
    tol = 1e-6
    if i == 0:
        if abs(y) <= tol or abs(1 - y) <= tol:
            return True, 0.0
        else:
            return False, None


class Poisson(femto.Model):
    def __init__(self, fields=None, source=None, exact=None):
        super().__init__(fields, exact)
        self.source = source

    def stiffness_kernel(self, i, j, x, u, v, grad_u, grad_v):
        if i == 0 and j == 0:
            return np.dot(grad_u, grad_v)

    def load_kernel(self, i, x, v, grad_v):
        if i == 0:
            return self.source(i, x)*v

    def plot(self, i_field=0, n_plot=10):
        if i_field == 0:
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
    quad_order = 2
    n_plot = 41

    t_dict = create_mesh(plot=True)
    mesh = mesher.TriangleMesh(t_dict)
    ref_triangle = elements.TriangleP1(quad_order=quad_order)

    n_dof = len(mesh.nodes)
    uh = femto.FunctionSpace(mesh, ref_triangle, n_dof, idx=0)

    fields = [uh]
    model = Poisson(fields, source=f, exact=None)
    model.dirichlet = in_dbc

    model.solve()
    model.plot(i_field=0, n_plot=n_plot)
