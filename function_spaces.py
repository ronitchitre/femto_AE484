#!/usr/bin/env python3

"""
Implementation of common elements.

Amuthan A. Ramabathiran
October 2022
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.tri as tri
import femtolib as femto


class P1(femto.FunctionSpace):
    def __init__(self, mesh, reference, idx=0):
        n_dof = len(mesh.nodes)
        super().__init__(reference, n_dof, idx)
        
    def eval(self, mesh, x, D=None):
        elt_id = mesh.find_element(x)
        xi = mesh.get_ref_coords(elt_id, x)
        uh = 0.0
        if D is None:
            for i in range(self.reference.n_dof):
                uh += ( self.reference.phi(i, *xi)
                       *self.dof[mesh.elements[elt_id][i]] )
        else:
            for i in range(self.reference.n_dof):
                uh += ( self.reference.d_phi(i, D, *xi)
                       *self.dof[mesh.elements[elt_id][i]] )
        return uh
        
    def plot(self, mesh, plot_type='contour'):
        if plot_type == 'surface':
            plt.figure(figsize=(6, 6))
            ax = plt.axes(projection='3d')
            surf = ax.plot_trisurf(mesh.nodes[:, 0], mesh.nodes[:, 1],
                                   self.dof, color='red')
            # The next two lines may not be necessary
            surf._facecolors2d = surf._facecolors
            surf._edgecolors2d = surf._edgecolors

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel(f'Field {self.idx}')
            ax.legend()
            
        elif plot_type == 'contour':
            fig, ax = plt.subplots(1, 1)
            triangulation = tri.Triangulation(mesh.nodes[:, 0],
                                              mesh.nodes[:, 1])
            ax.tricontour(triangulation, self.dof,
                          levels=40, linewidths=0.5, colors='k')
            surf = ax.tricontourf(triangulation, self.dof,
                                  levels=40, cmap="RdBu_r")

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            fig.colorbar(surf, ax=ax)
            plt.subplots_adjust(hspace=0.5)
            
        plt.show()
        
        
class Q1(femto.FunctionSpace):
    def __init__(self, mesh, reference, idx=0):
        n_dof = len(mesh.nodes)
        super().__init__(reference, n_dof, idx)
        
    def eval(self, mesh, x, D=None):
        raise NotImplementedError()
        
    def plot(self, mesh):
        triangulation = tri.Triangulation(mesh.nodes[:, 0],
                                          mesh.nodes[:, 1])

        plt.figure(figsize=(6, 6))
        ax = plt.axes(projection='3d')

        surf = ax.plot_trisurf(triangulation, self.dof, color='red')
        # The next two lines may not be necessary
        surf._facecolors2d = surf._facecolors
        surf._edgecolors2d = surf._edgecolors

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel(f'Field {self.idx}')
        ax.legend()
        plt.show()