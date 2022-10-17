#!/usr/bin/env python3

"""
Femto: Object Oriented Finite Element Library

Amuthan A. Ramabathiran
October 2022
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
from numba import njit


class FiniteElement:
    def __init__(self, dim=1, n_dof=2, 
                 quad_order=1, quad_type=None, affine=True):
        self.dim = dim
        self.n_dof = n_dof
        self.quad_order = quad_order
        self.quad_type = quad_type
        self.affine = affine
        
        # These are set by self.init_quadrature()
        self.n_quad = 0
        self.qpts = None
        self.qwts = None
        
        # These are set by self.init_Gauss()
        self.phi_q = None
        self.d_phi_q = None
        
        self.init_quadrature()
        self.init_Gauss()

    def init_quadrature(self):
        raise NotImplementedError()

    def integrate(self, g):
        intgl = 0.0
        for i in range(self.n_quad):
            intgl += self.qwts[i] * g(self.qpts[i])
        return intgl
        
    def integrate_gauss(self, gs):
        return np.sum(self.qwts * gs)

    def phi(self, idx_phi, *xi):
        raise NotImplementedError()

    def d_phi(self, idx_phi, idx_x, *xi):
        raise NotImplementedError()
        
    def init_Gauss(self):
        self.phi_q = np.zeros((self.n_dof, self.n_quad))
        for i in range(self.n_dof):
            for j in range(self.n_quad):
                self.phi_q[i, j] = self.phi(i, *self.qpts[j])
        
        self.d_phi_q = np.zeros((self.n_dof, self.dim, self.n_quad))
        for i in range(self.n_dof):
            for j in range(self.dim):
                for k in range(self.n_quad):
                    self.d_phi_q[i, j, k] = self.d_phi(i, j, *self.qpts[k])


class Mesh:
    def __init__(self, dim=1, 
                 nodes=None, elements=None, boundary=None,
                 reference=None):
        self.dim = dim
        self.nodes = nodes
        self.elements = elements
        self.boundary = boundary
        self.reference = reference
        
        self.n_nodes = len(nodes)
        self.n_elements = len(elements)

    def find_element(self, x):
        raise NotImplementedError()
        
    def get_coords(self, elt_id, xi):
        nodes = self.nodes[self.elements[elt_id]]
        x = np.zeros(self.dim)
        for i in range(self.dim):
            for j in range(self.reference.n_dof):
                x[i] += self.reference.phi(j, *xi)*nodes[j, i]
        return x

    def Jacobian(self, elt_id, xi):
        nodes = self.nodes[self.elements[elt_id]] 
        J = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                J[i, j] = 0.0
                for k in range(self.reference.n_dof):
                    J[i, j] += nodes[k, i]*self.reference.d_phi(k, j, *xi)
        return J

    def get_ref_coords(self, elt_id, x):
        raise NotImplementedError()


class FunctionSpace:
    def __init__(self, reference, n_dof, idx=0):
        self.idx = idx
        self.reference = reference
        self.n_dof = n_dof
        self.dof = np.array([np.nan for _ in range(n_dof)])
        self.dbc = None
        self.n_solve = n_dof
        self.n_dirichlet = 0

    def eval(self, mesh, x, D=None):
        raise NotImplementedError()
        
    def plot(self, mesh):
        raise NotImplementedError()


class Model:
    def __init__(self, mesh, fields, exact=None):
        self.mesh = mesh
        self.exact = exact
        
        self.fields = fields
        self.n_fields = len(fields)

        self.n_dof = [field.n_dof for field in fields]
        cum_dof = [0 for _ in range(self.n_fields)]
        for i in range(1, self.n_fields):
            cum_dof[i] = cum_dof[i-1] + fields[i-1].n_dof
        self.cum_dof = cum_dof
        self.n_tot = np.sum(self.n_dof)

        self.n_dof_elt = [field.reference.n_dof for field in fields]
        cum_dof_elt = [0 for _ in range(self.n_fields)]
        for i in range(1, self.n_fields):
            cum_dof_elt[i] = cum_dof_elt[i-1] + fields[i-1].reference.n_dof
        self.cum_dof_elt = cum_dof_elt
        self.n_tot_elt = np.sum(self.n_dof_elt)

        self.dirichlet = None
        self.neumann = None

        self.node_idx = None
        self.node_idx_inv = None

        self.stiffness = None
        self.load = None

    def apply_dirichlet_bc(self):
        if self.dirichlet is not None:
            for i, u in enumerate(self.fields):
                in_bc = lambda x: self.dirichlet(i, *x)

                bc = []
                for j in self.mesh.boundary:
                    on_bdy, val = in_bc(self.mesh.nodes[j])
                    if on_bdy:
                        bc.append([j, val])

                for j, val in bc:
                    u.dof[j] = val

                u.dbc = bc
                u.n_dirichlet = len(bc)
                u.n_solve = u.n_dof - u.n_dirichlet

    def apply_neumann_bc(self):
        '''
        self.renumber() needs to be called before this method is called!
        '''
        if self.neumann is not None:
            for i, u in enumerate(self.fields):
                in_bc = lambda x: self.neumann(i, *x)

                bc = []
                for j in self.mesh.boundary:
                    on_bdy, val = in_bc(self.mesh.nodes[j])
                    if on_bdy:
                        bc.append([j, val])

                for node, val in bc:
                    ii = self.cum_dof[i] + node
                    self.load[self.node_idx[ii]] += val

    def _compute_inverse_node_indices(self):
        node_idx_inv = np.zeros_like(self.node_idx, dtype=int)

        idx_count = 0
        for n in self.node_idx:
            node_idx_inv[n] = idx_count
            idx_count += 1

        self.node_idx_inv = node_idx_inv

    def renumber(self):
        '''
        self.appy_dirichlet_bc() needs to be called before calling this!
        '''
        node_idx = np.zeros(self.n_tot, dtype=int)
        node_count = 0

        for i_field, field in enumerate(self.fields):
            for i, u in enumerate(field.dof):
                if np.isnan(u):
                    node_idx[self.cum_dof[i_field] + i] = node_count
                    node_count += 1

        for i_field, field in enumerate(self.fields):
            for i, u in enumerate(field.dof):
                if not np.isnan(u):
                    node_idx[self.cum_dof[i_field] + i] = node_count
                    node_count += 1

        self.node_idx = node_idx
        self._compute_inverse_node_indices()

    def stiffness_kernel(self, i, j, x, u, v, grad_u, grad_v):
        raise NotImplementedError()

    def load_kernel(self, i, x, v, grad_v):
        raise NotImplementedError()
        
    def reference_stiffness_matrix(self, elt_id):
        ke = np.zeros((self.n_tot_elt, self.n_tot_elt))
        
        affine = self.mesh.reference.affine
        if affine:
            # J = self.mesh.Jacobian(elt_id, self.mesh.reference.qpts[0])
            J = self.mesh.Jacobian(elt_id, None)
            vol = np.abs(np.linalg.det(J))
            J = np.linalg.inv(J).transpose()

        for i_field, field_i in enumerate(self.fields):
            n_dof_elt_i = field_i.reference.n_dof
            n_quad = field_i.reference.n_quad
            xis = field_i.reference.qpts
            
            if not affine:
                Js = []
                vols = []
            xs = []
            
            for k in range(n_quad):
                if not affine:
                    J = self.mesh.Jacobian(elt_id, xis[k])
                    vol = np.abs(np.linalg.det(J))
                    J = np.linalg.inv(J).transpose()
                    Js.append(J)
                    vols.append(vol)
                x = self.mesh.get_coords(elt_id, xis[k])
                xs.append(x)
            
            for i in range(n_dof_elt_i):
                ii = self.cum_dof_elt[i_field] + i
                
                for j_field, field_j in enumerate(self.fields):
                    n_dof_elt_j = field_j.reference.n_dof
                
                    for j in range(n_dof_elt_j):
                        jj = self.cum_dof_elt[j_field] + j
                        gs = np.zeros(n_quad)
                
                        for k in range(n_quad):
                            fi = field_i.reference.phi_q[i, k]
                            fj = field_j.reference.phi_q[j, k] 
                            if affine:
                                dfi = J @ field_i.reference.d_phi_q[i, :, k]
                                dfj = J @ field_j.reference.d_phi_q[j, :, k]
                                gs[k] = self.stiffness_kernel(
                                    i_field, j_field, xs[k], fi, fj, dfi, dfj
                                ) * vol
                            else:
                                dfi = Js[k] @ field_i.reference.d_phi_q[i, :, k]
                                dfj = Js[k] @ field_j.reference.d_phi_q[j, :, k]
                                gs[k] = self.stiffness_kernel(
                                    i_field, j_field, xs[k], fi, fj, dfi, dfj
                                ) * vols[k]
                        
                        ke[ii, jj] = field_i.reference.integrate_gauss(gs)
                        
        return ke
           
    def reference_load_vector(self, elt_id):
        fe = np.zeros(self.n_tot_elt)
        
        affine = self.mesh.reference.affine
        if affine:
            # J = self.mesh.Jacobian(elt_id, self.mesh.reference.qpts[0])
            J = self.mesh.Jacobian(elt_id, None)
            vol = np.abs(np.linalg.det(J))
            J = np.linalg.inv(J).transpose()

        for i_field, field in enumerate(self.fields):
            n_dof_elt = field.reference.n_dof
            n_quad = field.reference.n_quad
            xis = field.reference.qpts
            
            for i in range(n_dof_elt):
                ii = self.cum_dof_elt[i_field] + i
                gs = np.zeros(n_quad)
                
                for k in range(n_quad):
                    if not affine:
                        J = self.mesh.Jacobian(elt_id, xis[k])
                        vol = np.abs(np.linalg.det(J))
                        J = np.linalg.inv(J).transpose()
                    x = self.mesh.get_coords(elt_id, xis[k])
                    fi = field.reference.phi_q[i, k]
                    dfi = J @ field.reference.d_phi_q[i, :, k]
                    gs[k] = self.load_kernel(i_field, x, fi, dfi)*vol

                fe[ii] = field.reference.integrate_gauss(gs)
                
        return fe
  
    def assemble_stiffness(self):
        II = []
        JJ = []
        V = []

        for i_field in range(self.n_fields):
            for i_elt, elt in enumerate(self.mesh.elements):
                ke = self.reference_stiffness_matrix(i_elt)

                for i in range(self.n_dof_elt[i_field]):
                    ii = self.cum_dof[i_field] + elt[i]
                    
                    for j_field in range(self.n_fields):
                        for j in range(self.n_dof_elt[j_field]):
                            jj = self.cum_dof[j_field] + elt[j]
                            
                            II.append(self.node_idx[ii])
                            JJ.append(self.node_idx[jj])
                            V.append(ke[self.cum_dof_elt[i_field] + i,
                                        self.cum_dof_elt[j_field] + j])

        K = coo_matrix((V, (II, JJ)), shape=(self.n_tot, self.n_tot))
        self.stiffness = K
  
    def assemble_load(self):
        F = np.zeros(self.n_tot)

        for i_field in range(self.n_fields):
            for i_elt, elt in enumerate(self.mesh.elements):
                fe = self.reference_load_vector(i_elt)

                for i in range(self.n_dof_elt[i_field]):
                    ii = self.cum_dof[i_field] + elt[i]
                    F[self.node_idx[ii]] += fe[self.cum_dof_elt[i_field] + i]

        self.load = F

    def solve(self):
        self.apply_dirichlet_bc()
        self.renumber()

        self.assemble_stiffness()
        self.stiffness = self.stiffness.tocsr()

        self.assemble_load()
        self.apply_neumann_bc()

        U_dbc = []
        for i, u in enumerate(self.fields):
            for _, val in u.dbc:
                U_dbc.append(val)
        U_dbc = np.array(U_dbc)

        N = 0
        for u in self.fields:
            N += u.n_solve

        U = spsolve(self.stiffness[:N, :N],
                    self.load[:N] - self.stiffness[:N, N:] @ U_dbc)

        solve_count = 0
        for i_field, u in enumerate(self.fields):
            for i in range(u.n_solve):
                ii = self.node_idx_inv[solve_count] - self.cum_dof[i_field]
                u.dof[ii] = U[solve_count]
                solve_count += 1

    def plot(self, i_field, n_plot):
        raise NotImplementedError()
