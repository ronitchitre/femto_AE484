#!/usr/bin/env python3

"""
Test 2d object oriented fem code for Poisson equation

Amuthan Ramabathiran
September 2022
"""

import femtolib as femto
import mesher
import elements
import function_spaces
import numpy as np

pi = np.pi


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
    def __init__(self, mesh=None, fields=None, exact=None):
        super().__init__(mesh, fields, exact)
        
        self.dirichlet = in_dbc
        self.neumann = None
        
    def source(self, i_field, xs):
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
            

if __name__ == '__main__':
    n_side = 20
    quad_type = 'triangle'
    # quad_type = 'quadrilateral'
    quad_order = 2
    n_plot = 41

    if quad_type == 'triangle':
        ref_elt = elements.TriangleP1(quad_order=quad_order)
        mesh = mesher.UnitSquareTri(nx=n_side, ny=n_side, reference=ref_elt)
        u1h = function_spaces.P1(mesh, ref_elt, idx=0)
        u2h = function_spaces.P1(mesh, ref_elt, idx=1)
    elif quad_type == 'quadrilateral':
        ref_elt = elements.QuadP1(quad_order=quad_order)
        mesh = mesher.UnitSquareQuad(nx=n_side, ny=n_side, reference=ref_elt)
        u1h = function_spaces.Q1(mesh, ref_elt, idx=0)
        u2h = function_spaces.Q1(mesh, ref_elt, idx=1)
    fields = [u1h, u2h]
    
    model = Poisson2D(mesh, fields, exact=u_exact)
    model.solve()
    
    u1h.plot(mesh)
    u2h.plot(mesh)
