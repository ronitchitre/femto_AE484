#!/usr/bin/env python3

"""
Functions to create meshes.

Amuthan A. Ramabathiran
October 2022
"""

import numpy as np
import femtolib as femto


def _in_triangle(x, y, x1, x2, x3, y1, y2, y3):
    J = np.zeros((2, 2))
    J[0, 0] = x2 - x1
    J[0, 1] = x3 - x1
    J[1, 0] = y2 - y1
    J[1, 1] = y3 - y1
    l1, l2 = np.linalg.inv(J) @ np.array([x - x1, x - y1])
    if l1 >= 0 and l2 >= 0 and (l1 + l2) <= 1:
        return True
    else:
        return False


def _find_element_triangle(x, nodes, elements):
    elt_id = -1
    for iE, elt in enumerate(elements):
        coords = nodes[elt]
        x1, x2, x3 = coords[:, 0]
        y1, y2, y3 = coords[:, 1]
        if _in_triangle(x[0], x[1], 
                        x1, x2, x3, 
                        y1, y2, y3):
            elt_id = iE
            break
    return elt_id


class UnitSquareTri(femto.Mesh):
    def __init__(self, nx=4, ny=4):
        dim = 2
        self.nx = nx
        self.ny = ny
        hx = 1.0/nx
        hy = 1.0/ny
        
        nodes = []
        node_count = 0
        bdy = []
        tol = 1e-6

        for j in range(ny + 1):
            y = j*hy
            for i in range(nx + 1):
                x = i*hx
                nodes.append([x, y])

                if (   abs(x) <= tol
                    or abs(1 - x) <= tol
                    or abs(y) <= tol
                    or abs(1 - y) <= tol ):
                    bdy.append(node_count)

                node_count += 1

        nodes = np.array(nodes)
        boundary = np.array(bdy, dtype=int)
        elements = []

        for j in range(ny):
            for i in range(nx):
                elements.append([
                    j*(nx + 1) + i,
                    j*(nx + 1) + i + 1,
                    (j + 1)*(nx + 1) + i
                ])

                elements.append([
                    j*(nx + 1) + i + 1,
                    (j + 1)*(nx + 1) + i + 1,
                    (j + 1)*(nx + 1) + i
                ])

        elements = np.array(elements, dtype=int)

        super().__init__(dim, nodes, elements, bdy)

    def find_element(self, x):
        return _find_element_triangle(x, self.nodes, self.elements)
        

def _find_element_quad(x, nodes, elements):
    elt_id = -1
    for iE, elt in enumerate(elements):
        coords = nodes[elt]
        x1, x2, x3, x4 = coords[:, 0]
        y1, y2, y3, y4 = coords[:, 1]
        t1 = _in_triangle(x[0], x[1],
                          x1, x2, x4,
                          y1, y2, y4)
        t2 = _in_triangle(x[0], x[1],
                          x3, x4, x2,
                          y3, y4, y2)
        if t1 or t2:
            elt_id = iE
            break
    return elt_id

        
class UnitSquareQuad(femto.Mesh):
    def __init__(self, nx=4, ny=4):
        dim = 2
        self.nx = nx
        self.ny = ny
        hx = 1.0/nx
        hy = 1.0/ny
        
        nodes = []
        node_count = 0
        bdy = []
        tol = 1e-6

        for j in range(ny + 1):
            y = j*hy
            for i in range(nx + 1):
                x = i*hx
                nodes.append([x, y])

                if (   abs(x) <= tol
                    or abs(1 - x) <= tol
                    or abs(y) <= tol
                    or abs(1 - y) <= tol ):
                    bdy.append(node_count)

                node_count += 1

        nodes = np.array(nodes)
        boundary = np.array(bdy, dtype=int)
        elements = []

        for j in range(ny):
            for i in range(nx):
                elements.append([
                    j*(nx + 1) + i,
                    j*(nx + 1) + i + 1,
                    (j + 1)*(nx + 1) + i + 1,
                    (j + 1)*(nx + 1) + i
                ])

        elements = np.array(elements, dtype=int)

        super().__init__(dim, nodes, elements, bdy)

    def find_element(self, x):
        return _find_element_quad(x, self.nodes, self.elements)


class TriangleMesh(femto.Mesh):
    def __init__(self, t_dict=None):
        dim = 2

        nodes = t_dict['vertices']
        elements = t_dict['triangles']

        boundary = []
        markers = t_dict['vertex_markers']
        for i in range(len(nodes)):
            if markers[i, 0] == 1:
                boundary.append(i)
        boundary = np.array(boundary, dtype=int)

        super().__init__(dim, nodes, elements, boundary)

    def find_element(self, x):
        return _find_element_triangle(x, self.nodes, self.elements)
