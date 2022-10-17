#!/usr/bin/env python3

"""
Plane stress linear elasticity

Amuthan Ramabathiran
October 2022
"""

import femtolib as femto
import mesher
import elements
import function_spaces
import numpy as np

pi = np.pi


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
    def __init__(self, mesh=None, fields=None, 
                 lamda=1.0, mu=1.0, exact=None):
        super().__init__(mesh, fields, exact)
        
        self.lamda = lamda
        self.mu = mu
        
        self.dirichlet = in_dirichlet
        self.neumann = in_neumann
    
    def b(self, i, xs):
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

    def stiffness_kernel(self, i, j, x, u, v, grad_u, grad_v):
        if i == 0 and j == 0:
            return ( (self.lamda + 2*self.mu)*(grad_u[0]*grad_v[0])
                    + self.mu*grad_u[1]*grad_v[1] )
        elif i == 0 and j == 1:
            return ( self.lamda*grad_u[0]*grad_v[1]
                    + self.mu*grad_u[1]*grad_v[0] )
        elif i == 1 and j == 0:
            return ( self.lamda*grad_u[1]*grad_v[0]
                    + self.mu*grad_u[0]*grad_v[1] )
        elif i == 1 and j == 1:
            return ( (self.lamda + 2*self.mu)*grad_u[1]*grad_v[1]
                    + self.mu*grad_u[0]*grad_v[0] )

    def load_kernel(self, i, x, v, grad_v):
        return self.b(i, x)*v


if __name__ == '__main__':
    L = 25.0
    W_half = 0.5
    nx = 250
    ny = 20
    
    quad_type = 'triangle'
    # quad_type = 'quadrilateral'
    quad_order = 2
    
    E = 1e5
    nu = 0.3
    lamda = E*nu/((1 + nu)*(1 - 2*nu))
    mu = E/(2*(1 + nu))
    
    if quad_type == 'triangle':
        ref_elt = elements.TriangleP1(quad_order=quad_order)
        mesh = mesher.UnitSquareTri(nx=nx, ny=ny, reference=ref_elt)
        mesh.nodes[:, 0] *= L
        mesh.nodes[:, 1] = W_half*(2*mesh.nodes[:, 1] - 1.0)
        uh = function_spaces.P1(mesh, ref_elt, idx=0)
        vh = function_spaces.P1(mesh, ref_elt, idx=1)
    elif quad_type == 'quadrilateral':
        ref_elt = elements.QuadP1(quad_order=quad_order)
        mesh = mesher.UnitSquareQuad(nx=nx, ny=ny, reference=ref_elt)
        mesh.nodes[:, 0] *= L
        mesh.nodes[:, 1] = W_half*(2*mesh.nodes[:, 1] - 1.0)
        uh = function_spaces.Q1(mesh, ref_elt, idx=0)
        vh = function_spaces.Q1(mesh, ref_elt, idx=1)
    fields = [uh, vh]
    
    model = PlaneStress(mesh, fields, lamda=lamda, mu=mu, exact=None)
    model.solve()

    uh.plot(mesh)
    vh.plot(mesh)

    u_tip = uh.eval(mesh, [L, 0.0])
    v_tip = vh.eval(mesh, [L, 0.0])
    print(f'Tip displacement = ({u_tip}, {v_tip})')
