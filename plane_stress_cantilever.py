#!/usr/bin/env python3

"""
Plane stress linear elasticity

Amuthan Ramabathiran
October 2022
"""

import femtolib as femto
import mesher
import elements
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits import mplot3d

from mesher import UnitSquareTri, UnitSquareQuad

pi = np.pi


def b(i, xs):
    if i == 0:
        if xs.ndim == 1:
            return 0.0
        else:
            return np.zeros(len(xs))
    elif i == 1:
        B = 1e-3
        if xs.ndim == 1:
            return B
        else:
            return B*np.ones(len(xs))


def in_dirichlet(i, x, y):
    L = 20
    W_half = 1
    tol = 1e-6
    if abs(x) < tol:
        return True, 0.0
    else:
        return False, None


def in_neumann(i, x, y):
    L = 20
    W_half = 1
    tol = 1e-6
    if i == 1:
        if not in_dirichlet(i, x, y)[0]:
            if abs(L - x) < tol:
                return True, 0.0
            else:
                return True, 0.0
        else:
            return False, None
    elif i == 0:
        if not in_dirichlet(i, x, y)[0]:
            return True, 0.0
        else:
            return False, None



class PlaneStress(femto.Model):
    def __init__(self, fields=None,
                 lamda=1.0, mu=1.0, body_force=None,
                 exact=None):
        super().__init__(fields, exact)
        self.lamda = lamda
        self.mu = mu
        self.b = body_force

    def stiffness_kernel(self, i, j, x, u, v, grad_u, grad_v):
        if i == 0 and j == 0:
            return ( (self.lamda + 2*self.mu)*(grad_u[0]*grad_v[0])
                    + 2*self.mu*grad_u[1]*grad_v[1] )
        elif i == 0 and j == 1:
            return ( self.lamda*grad_u[0]*grad_v[1]
                    + 2*self.mu*grad_u[1]*grad_v[0] )
        elif i == 1 and j == 0:
            return ( self.lamda*grad_u[1]*grad_v[0]
                    + 2*self.mu*grad_u[0]*grad_v[1] )
        elif i == 1 and j == 1:
            return ( (self.lamda + 2*self.mu)*grad_u[1]*grad_v[1]
                    + 2*self.mu*grad_u[0]*grad_v[0] )

    def load_kernel(self, i, x, v, grad_v):
        return self.b(i, x)*v

    def plot(self, i_field=0):
        if i_field == 0 or i_field == 1:
            u = self.fields[i_field]
            triangulation = tri.Triangulation(u.mesh.nodes[:, 0],
                                              u.mesh.nodes[:, 1])

            fig, ax = plt.subplots(1, 1)
            ax.tricontour(triangulation, u.dof,
                          levels=65, linewidths=0.5, colors='k')
            surf = ax.tricontourf(triangulation, u.dof,
                                  levels=65, cmap="RdBu_r")

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            fig.colorbar(surf, ax=ax)
            plt.subplots_adjust(hspace=0.5)
            plt.show()


if __name__ == '__main__':
    L = 25.0
    W_half = 0.5
    nx = 25
    ny = 2
    quad_order = 2
    E = 1e5
    nu = 0.3
    lamda = E*nu/((1 + nu)*(1 - 2*nu))
    mu = E/(2*(1 + nu))

    mesh = UnitSquareTri(nx=nx, ny=ny)
    # mesh = UnitSquareQuad(nx=nx, ny=ny)
    mesh.nodes[:, 0] *= L
    mesh.nodes[:, 1] = W_half*(2*mesh.nodes[:, 1] - 1.0)
    ref_elt = elements.TriangleP1(quad_order=quad_order)
    # ref_elt = elements.QuadP1(quad_order=quad_order)

    n_dof = len(mesh.nodes)
    uh = femto.FunctionSpace(mesh, ref_elt, n_dof, idx=0)
    vh = femto.FunctionSpace(mesh, ref_elt, n_dof, idx=1)

    fields = [uh, vh]
    model = PlaneStress(fields, lamda=lamda, mu=mu, body_force=b, exact=None)
    model.dirichlet = in_dirichlet
    model.neumann = in_neumann

    model.solve()

    model.plot(i_field=0)
    model.plot(i_field=1)

    # print(f'Tip displacement = ({uh.eval([L, 0.0])}, {vh.eval([L, 0.0])})')
