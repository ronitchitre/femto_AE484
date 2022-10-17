#!/usr/bin/env python3

"""
Steady state heat transfer in a domain with holes

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


pi = np.pi


def add_hole(x, y, r, n_bdy, vertices, segments):
    i = np.arange(n_bdy)
    theta = 2*pi*i/n_bdy
    pts = np.stack([x + r*np.cos(theta), y + r*np.sin(theta)], axis=1)
    segs = np.stack([i, i + 1], axis=1) % n_bdy

    vertices = np.vstack([vertices, pts])
    segments = np.vstack([segments, segs + segments.shape[0]])

    return vertices, segments


def create_mesh(a=0.005, n_bdy=10, r=0.1, plot=True):
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)

    center_holes = np.array([
        [0.5, 0.1],
        [0.2, 0.4], [0.8, 0.4],
        [0.4, 0.6], [0.6, 0.6]
    ])
    # center_holes = np.array([[0.5, 0.5]])

    for c in center_holes:
        vertices, segments = add_hole(
            c[0], c[1], r, n_bdy, vertices, segments)

    mesh_data = dict(
        vertices=vertices, segments=segments, holes=center_holes
    )
    t = tr.triangulate(mesh_data, f'qpa{a}')

    if plot:
        tr.compare(plt, mesh_data, t)

    return t

def in_dirichlet(i, x, y):
    tol = 1e-6
    center_holes = np.array([
        [0.5, 0.1],
        [0.2, 0.4], [0.8, 0.4],
        [0.4, 0.6], [0.6, 0.6]
    ])
    # center_holes = np.array([[0.5, 0.5]])
    r = 0.05

    on_hole = False
    for c in center_holes:
        if np.abs((x - c[0])**2 + (y - c[1])**2 - r**2) < tol:
            on_hole = True
            break

    if on_hole:
        return on_hole, 1.0
    elif abs(y) < tol:
        return True, 0.75
    else:
        return on_hole, None


def in_neumann(i, x, y):
    tol = 1e-6
    if i == 0:
        if (   abs(x) <= tol or abs(1 - x) <= tol
            or abs(1 - y) <= tol ):
            if ( abs(1 - y) <= tol and
                ((x >= 0.1 and x <= 0.3) or (x >= 0.7 and x <= 0.9)) ):
                return True, -0.5
            else:
                return True, 0.0
        else:
            return False, None


class HeatConduction(femto.Model):
    def __init__(self, mesh=None, fields=None, exact=None):
        super().__init__(mesh, fields, exact)
        
        self.dirichlet = in_dirichlet
        self.neumann = in_neumann
        
    def source(self, i, xs):
        if i == 0:
            if xs.ndim == 1:
                return 0.0
            else:
                return np.zeros(len(xs))

    def stiffness_kernel(self, i, j, x, u, v, grad_u, grad_v):
        if i == 0 and j == 0:
            return np.dot(grad_u, grad_v)

    def load_kernel(self, i, x, v, grad_v):
        if i == 0:
            return self.source(i, x)*v


if __name__ == '__main__':
    quad_order = 2
    
    ref_triangle = elements.TriangleP1(quad_order=quad_order)
    t_dict = create_mesh(a=0.002, n_bdy=8, r=0.05, plot=True)
    mesh = mesher.TriMesh(*mesher.triangle_to_femto(t_dict), ref_triangle)

    uh = function_spaces.P1(mesh, ref_triangle, idx=0)
    fields = [uh]
    
    model = HeatConduction(mesh, fields, exact=None)
    model.solve()
    
    uh.plot(mesh)
