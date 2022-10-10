#!/usr/bin/env python3

"""
Femto: Object Oriented Finite Element Library

Amuthan A. Ramabathiran
October 2022
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix


class Mesh:
    def __init__(self, dim=1, nodes=None, elements=None, boundary=None):
        self.dim = dim
        self.nodes = nodes
        self.elements = elements
        self.boundary = boundary

    def find_element(self, x):
        raise NotImplementedError()


class FiniteElement:
    def __init__(self):
        self.n_dof = 0
        self.quad_order = 0
        self.quad_type = None
        self.n_quad = 0
        self.qpts = None
        self.qwts = None

    def init_quadrature(self):
        raise NotImplementedError()

    def integrate(self, g):
        intgl = 0.0
        for i in range(self.n_quad):
            intgl += self.qwts[i] * g(self.qpts[i])
        return intgl

    def phi(self, idx_phi, *xi):
        raise NotImplementedError()

    def d_phi(self, idx_phi, idx_x, *xi):
        raise NotImplementedError()

    def get_coords(self, xi, nodes):
        raise NotImplementedError()

    def Jacobian(self, nodes):
        raise NotImplementedError()

    def get_ref_coords(self, x, nodes):
        raise NotImplementedError()


class FunctionSpace:
    def __init__(self, mesh, reference, n_dof, idx=0):
        self.idx = idx
        self.mesh = mesh
        self.reference = reference
        self.n_dof = n_dof
        self.dof = np.array([np.nan for _ in range(n_dof)])
        self.dbc = None
        self.n_solve = n_dof
        self.n_dirichlet = 0

    def eval(self, x, D=None):
        elt_id = self.mesh.find_element(x)
        nodes = self.mesh.nodes[self.mesh.elements[elt_id]]
        xi = self.reference.get_ref_coords(nodes)
        uh = 0.0
        if D is None:
            for i in range(self.reference.n_dof):
                uh += ( self.reference.phi(i, *xi)
                       *self.dof[self.reference.elements[elt_id][i]] )
        else:
            for i in range(self.reference.n_dof):
                uh += ( self.reference.d_phi(i, D, *xi)
                       *self.dof[self.reference.elements[elt_id][i]] )
        return uh


class Model:
    def __init__(self, fields, exact=None):
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
                for j in u.mesh.boundary:
                    on_bdy, val = in_bc(u.mesh.nodes[j])
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
                for j in u.mesh.boundary:
                    on_bdy, val = in_bc(u.mesh.nodes[j])
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

    def reference_stiffness_matrix(self, nodes):
        ke = np.zeros((self.n_tot_elt, self.n_tot_elt))

        for i_field, field_i in enumerate(self.fields):
            J_i = field_i.reference.Jacobian(nodes)
            vol_i = np.abs(np.linalg.det(J_i))
            J_i = np.linalg.inv(J_i).transpose()

            for i in range(self.n_dof_elt[i_field]):
                ii = self.cum_dof_elt[i_field] + i

                for j_field, field_j in enumerate(self.fields):
                    J_j = field_j.reference.Jacobian(nodes)
                    J_j = np.linalg.inv(J_j).transpose()

                    for j in range(self.n_dof_elt[j_field]):
                        def g(xi):
                            x = field_i.reference.get_coords(xi, nodes)
                            fi = field_i.reference.phi(i, *xi)
                            fj = field_j.reference.phi(j, *xi)
                            dfi = J_i @ np.array(
                                [field_i.reference.d_phi(i, dim, *xi)
                                 for dim in range(field_i.mesh.dim)]
                            )
                            dfj = J_j @ np.array(
                                [field_j.reference.d_phi(j, dim, *xi)
                                 for dim in range(field_j.mesh.dim)]
                            )
                            return self.stiffness_kernel(
                                i_field, j_field, x, fi, fj, dfi, dfj
                            )

                        jj = self.cum_dof_elt[j_field] + j
                        ke[ii, jj] = field_i.reference.integrate(g)

            ke *= vol_i

        return ke

    def reference_load_vector(self, nodes):
        fe = np.zeros(self.n_tot_elt)

        for i_field, field in enumerate(self.fields):
            J = field.reference.Jacobian(nodes)
            vol = np.abs(np.linalg.det(J))
            J = np.linalg.inv(J).transpose()

            for i in range(self.n_dof_elt[i_field]):
                def g(xi):
                    x = field.reference.get_coords(xi, nodes)
                    fi = field.reference.phi(i, *xi)
                    dfi = J @ np.array(
                        [field.reference.d_phi(i, dim, *xi)
                         for dim in range(field.mesh.dim)]
                    )
                    return self.load_kernel(i_field, x, fi, dfi)

                ii = self.cum_dof_elt[i_field] + i
                fe[ii] = field.reference.integrate(g)

            fe *= vol

        return fe

    def assemble_stiffness(self):
        II = []
        JJ = []
        V = []

        for i_field, field_i in enumerate(self.fields):
            for elt in field_i.mesh.elements:
                nodes = field_i.mesh.nodes[elt]
                ke = self.reference_stiffness_matrix(nodes)

                for i in range(self.n_dof_elt[i_field]):
                    for j_field in range(self.n_fields):
                        for j in range(self.n_dof_elt[j_field]):
                            ii = self.cum_dof[i_field] + elt[i]
                            jj = self.cum_dof[j_field] + elt[j]
                            II.append(self.node_idx[ii])
                            JJ.append(self.node_idx[jj])
                            V.append(ke[self.cum_dof_elt[i_field] + i,
                                        self.cum_dof_elt[j_field] + j])

        K = coo_matrix((V, (II, JJ)), shape=(self.n_tot, self.n_tot))
        self.stiffness = K

    def assemble_load(self):
        F = np.zeros(self.n_tot)

        for i_field, field in enumerate(self.fields):
            for elt in field.mesh.elements:
                nodes = field.mesh.nodes[elt]
                fe = self.reference_load_vector(nodes)

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
