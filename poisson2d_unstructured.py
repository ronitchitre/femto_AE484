#!/usr/bin/env python3

"""
Poisson equation on unstructured triangular mesh.

Amuthan Ramabathiran
October 2022
"""

import femtolib as femto
import mesher
import elements
import function_spaces

import numpy as np
import triangle as tr
import matplotlib.pyplot as plt


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
    def __init__(self, mesh=None, fields=None, exact=None):
        super().__init__(mesh, fields, exact)
        
        self.dirichlet = in_dbc
        self.neumann = None
        
    def source(self, idx, xs):
        if idx == 0:
            if xs.ndim == 1:
                return 1.0
            else:
                return np.ones(len(xs))

    def stiffness_kernel(self, i, j, x, u, v, grad_u, grad_v):
        if i == 0 and j == 0:
            return np.dot(grad_u, grad_v)

    def load_kernel(self, i, x, v, grad_v):
        if i == 0:
            return self.source(i, x)*v
            

if __name__ == '__main__':
    quad_order = 2
    n_plot = 41
    
    ref_triangle = elements.TriangleP1(quad_order=quad_order)

    t_dict = create_mesh(plot=True)
    mesh = mesher.TriMesh(*mesher.triangle_to_femto(t_dict), ref_triangle)

    uh = function_spaces.P1(mesh, ref_triangle, idx=0)
    fields = [uh]
    
    model = Poisson(mesh, fields, exact=None)
    model.solve()
    
    uh.plot(mesh)
