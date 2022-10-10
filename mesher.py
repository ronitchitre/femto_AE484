#!/usr/bin/env python3

"""
Functions to create meshes.

Amuthan A. Ramabathiran
October 2022
"""

import numpy as np
import femtolib as femto


def _find_element_triangle(x, nodes, elements):
    elt_id = -1
    for iE, elt in enumerate(elements):
        coords = nodes[elt]
        x1, x2, x3 = coords[:, 0]
        y1, y2, y3 = coords[:, 1]
        J = np.zeros((2, 2))
        J[0, 0] = x2 - x1
        J[0, 1] = x3 - x1
        J[1, 0] = y2 - y1
        J[1, 1] = y3 - y1
        l1, l2 = np.linalg.inv(J) @ np.array([x[0] - x1, x[1] - y1])
        if l1 >= 0 and l2 >= 0 and (l1 + l2) <= 1:
            elt_id = iE
            break
    return elt_id


class UnitSquareTri(femto.Mesh):
    def __init__(self, n_side=4):
        self.n_side = n_side
        dim = 2
        h = 1.0/n_side
        nodes = []
        node_count = 0
        bdy = []
        tol = 1e-6

        for j in range(n_side + 1):
            y = j*h
            for i in range(n_side + 1):
                x = i*h
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

        for j in range(n_side):
            for i in range(n_side):
                elements.append([
                    j*(n_side + 1) + i,
                    j*(n_side + 1) + i + 1,
                    (j + 1)*(n_side + 1) + i
                ])

                elements.append([
                    j*(n_side + 1) + i + 1,
                    (j + 1)*(n_side + 1) + i + 1,
                    (j + 1)*(n_side + 1) + i
                ])

        elements = np.array(elements, dtype=int)

        super().__init__(dim, nodes, elements, bdy)

    def find_element(self, x):
        return _find_element_triangle(x, self.nodes, self.elements)


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
