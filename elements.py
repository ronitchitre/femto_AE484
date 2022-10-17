#!/usr/bin/env python3

"""
Implementation of common elements.

Amuthan A. Ramabathiran
October 2022
"""

import numpy as np
import femtolib as femto


class TriangleP1(femto.FiniteElement):
    def __init__(self, quad_order=2, quad_type='area', affine=True):
        dim = 2
        n_dof = 3 
        super().__init__(dim, n_dof, quad_order, quad_type, affine)

    def _get_GL_pts_wts_1d(self, n_quad):
        if n_quad == 1:
            pts = np.array([0.0])
            wts = np.array([2.0])
        elif n_quad == 2:
            xi = 1.0/np.sqrt(3)
            pts = np.array([-xi, xi])
            wts = np.array([1.0, 1.0])
        elif n_quad == 3:
            xi = np.sqrt(3/5)
            pts = np.array([-xi, 0, xi])
            wts = np.array([5/9, 8/9, 5/9])
        elif n_quad == 4:
            xi_1 = np.sqrt((3/7) - (2/7)*np.sqrt(6/5))
            xi_2 = np.sqrt((3/7) + (2/7)*np.sqrt(6/5))
            w1 = (18 + np.sqrt(30))/36
            w2 = (18 - np.sqrt(30))/36
            pts = np.array([-xi_2, -xi_1, xi_1, xi_2])
            wts = np.array([w2, w1, w1, w2])
        elif n_quad == 5:
            xi_1 = np.sqrt(5 - 2*np.sqrt(10/7))/3
            xi_2 = np.sqrt(5 + 2*np.sqrt(10/7))/3
            w1 = (322 + 13*np.sqrt(70))/900
            w2 = (322 - 13*np.sqrt(70))/900
            pts = np.array([-xi_2, -xi_1, 0, xi_1, xi_2])
            wts = np.array([w2, w1, 128/225, w1, w2])
        else:
            raise Exception("Invalid quadrature order!")

        return pts, wts

    def init_quadrature(self):
        if self.quad_type == 'area':
            if self.quad_order == 1:
                pts = np.array([[1/3, 1/3]])
                wts = np.array([1.0])
            elif self.quad_order == 2:
                pts = np.array([[1/6, 1/6],
                                [2/3, 1/6],
                                [1/6, 2/3]])
                wts = np.array([1/3, 1/3, 1/3])
            elif self.quad_order == 3:
                pts = np.array([[1/3, 1/3],
                                [1/5, 1/5],
                                [3/5, 1/5],
                                [1/5, 3/5]])
                wts = np.array([-27/48, 25/48, 25/48, 25/48])
            else:
                raise Exception("Invalid quadrature order!")
        elif self.quad_type == 'duffy':
            pts_x, wts_x = self._get_GL_pts_wts_1d(self.quad_order)
            pts_x = (1 + pts_x)/2
            wts_x = 2*wts_x
            pts = np.zeros((self.quad_order**2, 2))
            wts = np.zeros(self.quad_order**2)
            for j in range(self.quad_order):
                for i in range(self.quad_order):
                    pts[j*self.quad_order + i, 0] = pts_x[i]
                    pts[j*self.quad_order + i, 1] = pts_x[j]*(1 - pts_x[i])
                    wts[j*self.quad_order + i] = (wts_x[i]*wts_x[j]
                                                 *(1 - pts_x[i]))
        else:
            raise Exception(
                "Invalid quadrature type: use either 'area' or 'duffy'"
            )

        self.n_quad = len(wts)
        self.qpts = pts
        self.qwts = wts

    def phi(self, idx_phi, xi, eta):
        if idx_phi == 0:
            return (1 - xi - eta)
        elif idx_phi == 1:
            return xi
        elif idx_phi == 2:
            return eta
        else:
            raise Exception("Invalid shape function index")

    def d_phi(self, idx_phi, idx_x, xi, eta):
        if idx_phi == 0:
            if idx_x == 0:
                return -1.0
            elif idx_x == 1:
                return -1.0
            else:
                raise Exception("Invalid coordinate index")
        elif idx_phi == 1:
            if idx_x == 0:
                return 1.0
            elif idx_x == 1:
                return 0.0
            else:
                raise Exception("Invalid coordinate index")
        elif idx_phi == 2:
            if idx_x == 0:
                return 0.0
            elif idx_x == 1:
                return 1.0
            else:
                raise Exception("Invalid coordinate index")
        else:
            raise Exception("Invalid shape function index")


class QuadP1(femto.FiniteElement):
    def __init__(self, quad_order=2, quad_type='gauss', affine=True):
        dim = 2
        n_dof = 4
        super().__init__(dim, n_dof, quad_order, quad_type, affine)

    def _get_GL_pts_wts_1d(self, n_quad):
        if n_quad == 1:
            pts = np.array([0.0])
            wts = np.array([2.0])
        elif n_quad == 2:
            xi = 1.0/np.sqrt(3)
            pts = np.array([-xi, xi])
            wts = np.array([1.0, 1.0])
        elif n_quad == 3:
            xi = np.sqrt(3/5)
            pts = np.array([-xi, 0, xi])
            wts = np.array([5/9, 8/9, 5/9])
        elif n_quad == 4:
            xi_1 = np.sqrt((3/7) - (2/7)*np.sqrt(6/5))
            xi_2 = np.sqrt((3/7) + (2/7)*np.sqrt(6/5))
            w1 = (18 + np.sqrt(30))/36
            w2 = (18 - np.sqrt(30))/36
            pts = np.array([-xi_2, -xi_1, xi_1, xi_2])
            wts = np.array([w2, w1, w1, w2])
        elif n_quad == 5:
            xi_1 = np.sqrt(5 - 2*np.sqrt(10/7))/3
            xi_2 = np.sqrt(5 + 2*np.sqrt(10/7))/3
            w1 = (322 + 13*np.sqrt(70))/900
            w2 = (322 - 13*np.sqrt(70))/900
            pts = np.array([-xi_2, -xi_1, 0, xi_1, xi_2])
            wts = np.array([w2, w1, 128/225, w1, w2])
        else:
            raise Exception("Invalid quadrature order!")

        return pts, wts

    def init_quadrature(self):
        if self.quad_type == 'gauss':
            pts_x, wts_x = self._get_GL_pts_wts_1d(self.quad_order)
            pts = np.zeros((self.quad_order**2, 2))
            wts = np.zeros(self.quad_order**2)
            for j in range(self.quad_order):
                for i in range(self.quad_order):
                    pts[j*self.quad_order + i, 0] = pts_x[i]
                    pts[j*self.quad_order + i, 1] = pts_x[j]
                    wts[j*self.quad_order + i] = wts_x[i]*wts_x[j]
        else:
            raise Exception(
                "Invalid quadrature type: use either 'area' or 'duffy'"
            )

        self.n_quad = len(wts)
        self.qpts = pts
        self.qwts = wts

    def phi(self, idx_phi, xi, eta):
        if idx_phi == 0:
            return (1 - xi)*(1 - eta)/4
        elif idx_phi == 1:
            return (1 + xi)*(1 - eta)/4
        elif idx_phi == 2:
            return (1 + xi)*(1 + eta)/4
        elif idx_phi == 3:
            return (1 - xi)*(1 + eta)/4
        else:
            raise Exception("Invalid shape function index")

    def d_phi(self, idx_phi, idx_x, xi, eta):
        if idx_phi == 0:
            if idx_x == 0:
                return -(1 - eta)/4
            elif idx_x == 1:
                return -(1 - xi)/4
            else:
                raise Exception("Invalid coordinate index")
        elif idx_phi == 1:
            if idx_x == 0:
                return (1 - eta)/4
            elif idx_x == 1:
                return -(1 + xi)/4
            else:
                raise Exception("Invalid coordinate index")
        elif idx_phi == 2:
            if idx_x == 0:
                return (1 + eta)/4
            elif idx_x == 1:
                return (1 + xi)/4
            else:
                raise Exception("Invalid coordinate index")
        elif idx_phi == 3:
            if idx_x == 0:
                return -(1 + eta)/4
            elif idx_x == 1:
                return (1 - xi)/4
            else:
                raise Exception("Invalid coordinate index")
        else:
            raise Exception("Invalid shape function index")
