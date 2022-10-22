import unittest
import femtolib
import numpy as np


class TestFiniteElement(unittest.TestCase):

    def setUp(self):
        class TestElement(femtolib.FiniteElement):
            def __init__(self, quad_order=2, quad_type=2, affine=True):
                dim = 1
                n_dof = 2
                super().__init__(dim, n_dof, quad_order, quad_type, affine)

            def init_quadrature(self):
                if self.quad_type == 1:
                    pts = np.array([0.0])
                    wts = np.array([2.0])
                elif self.quad_type == 2:
                    xi = 1.0 / np.sqrt(3)
                    pts = np.array([[-xi], [xi]])
                    wts = np.array([[1.0], [1.0]])
                elif self.quad_type == 3:
                    xi = np.sqrt(3 / 5)
                    pts = np.array([-xi, 0, xi])
                    wts = np.array([5 / 9, 8 / 9, 5 / 9])
                elif self.quad_type == 4:
                    xi_1 = np.sqrt((3 / 7) - (2 / 7) * np.sqrt(6 / 5))
                    xi_2 = np.sqrt((3 / 7) + (2 / 7) * np.sqrt(6 / 5))
                    w1 = (18 + np.sqrt(30)) / 36
                    w2 = (18 - np.sqrt(30)) / 36
                    pts = np.array([-xi_2, -xi_1, xi_1, xi_2])
                    wts = np.array([w2, w1, w1, w2])
                elif self.quad_type == 5:
                    xi_1 = np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 3
                    xi_2 = np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 3
                    w1 = (322 + 13 * np.sqrt(70)) / 900
                    w2 = (322 - 13 * np.sqrt(70)) / 900
                    pts = np.array([-xi_2, -xi_1, 0, xi_1, xi_2])
                    wts = np.array([w2, w1, 128 / 225, w1, w2])
                else:
                    raise Exception("Invalid quadrature order!")

                self.qpts = pts
                self.qwts = wts
                self.n_quad = len(wts)

            def phi(self, i, *xi):
                if i == 0:
                    return np.array([1 - xi[0]])
                if i == 1:
                    return np.array([xi[0]])

            def d_phi(self, i, *xi):
                if i == 0:
                    return np.array([-1])
                if i == 1:
                    return np.array([1])

        self.test_element1 = TestElement()

    def test_integrate(self):
        g1 = lambda x: np.cos(x)
        reference_val1 = 0
        pts = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
        wts = np.array([1.0, 1.0])
        for i in range(len(wts)):
            reference_val1 += wts[i] * g1(pts[i])
        self.assertEqual(reference_val1, self.test_element1.integrate(g1))

    def test_integrate_gauss(self):
        ref_gs = 2
        ref_val = 0
        wts = np.array([1.0, 1.0])
        for i in wts:
            ref_val += ref_gs * i
        self.assertEqual(ref_val, self.test_element1.integrate_gauss(ref_gs))

    def test_init_Gauss(self):
        ref_val1 = np.array([[
            1 + 1 / np.sqrt(3), 1 - 1 / np.sqrt(3)],
            [-1 / np.sqrt(3), 1 / np.sqrt(3)]
        ])
        ref_val2 = np.array([[[-1, -1]], [[1, 1]]])
        test1 = self.test_element1.phi_q
        test2 = self.test_element1.d_phi_q
        self.assertTrue((ref_val1 == test1).all())
        self.assertTrue((ref_val2 == test2).all())


class TestMesh(unittest.TestCase):

    def setUp(self):

        def _in_segment(x, x1, x2):
            return x1 <= x <= x2

        class TestElement(femtolib.FiniteElement):
            def __init__(self, quad_order=2, quad_type=2, affine=True):
                dim = 1
                n_dof = 2
                super().__init__(dim, n_dof, quad_order, quad_type, affine)

            def init_quadrature(self):
                if self.quad_type == 1:
                    pts = np.array([0.0])
                    wts = np.array([2.0])
                elif self.quad_type == 2:
                    xi = 1.0 / np.sqrt(3)
                    pts = np.array([[-xi], [xi]])
                    wts = np.array([[1.0], [1.0]])

                self.qpts = pts
                self.qwts = wts
                self.n_quad = len(wts)

            def phi(self, i, *xi):
                if i == 0:
                    return np.array([1 - xi[0]])
                if i == 1:
                    return np.array([xi[0]])

            def d_phi(self, i, *xi):
                if i == 0:
                    return np.array([-1])
                if i == 1:
                    return np.array([1])

        self.test_element1 = TestElement()

        class TestClassMesh(femtolib.Mesh):
            def __init__(self, nodes, elements, boundary, reference):
                dim = 1
                super().__init__(dim, nodes, elements, boundary, reference)

            def find_element(self, x):
                elt_id = -1
                for iE, elt in enumerate(self.elements):
                    x1, x2 = self.nodes[elt]
                    if _in_segment(x, x1, x2):
                        elt_id = iE
                return elt_id

            def get_ref_coords(self, elt_id, x):
                coords = self.nodes[self.elements[elt_id]]
                return (x - coords[0]) / (coords[1] - coords[0])

        test_element = TestElement()
        nodes = np.array([[i] for i in np.linspace(0, 1, 5)])
        elements = np.zeros((nodes.shape[0] - 1, 2), dtype=int)
        boundary = np.array([0, 1])
        for i in range(nodes.shape[0] - 1):
            elements[i] = np.array([int(i), int(i + 1)])
        self.test_mesh = TestClassMesh(nodes, elements, boundary, test_element)


    def test_get_coords(self):
        ref_val1 = 0.125
        test_x1 = self.test_mesh.get_coords(0, [0.5])
        ref_val2 = 0.9375
        test_x2 = self.test_mesh.get_coords(3, [0.75])
        self.assertEqual(ref_val1, test_x1)
        self.assertEqual(ref_val2, test_x2)

    def test_jacobian(self):
        ref_val1 = 0.25
        ref_val2 = 0.25
        test_val1 = self.test_mesh.Jacobian(0, [0.5])
        test_val2 = self.test_mesh.Jacobian(2, [0.7])
        self.assertEqual(ref_val1, test_val1)
        self.assertEqual(ref_val2, test_val2)


class TestModel(unittest.TestCase):
    def setUp(self):

        def _in_segment(x, x1, x2):
            return x1 <= x <= x2

        class TestElement(femtolib.FiniteElement):
            def __init__(self, quad_order=2, quad_type=2, affine=True):
                dim = 1
                n_dof = 2
                super().__init__(dim, n_dof, quad_order, quad_type, affine)

            def init_quadrature(self):
                if self.quad_type == 1:
                    pts = np.array([0.0])
                    wts = np.array([2.0])
                elif self.quad_type == 2:
                    xi = 1.0 / np.sqrt(3)
                    pts = np.array([[-xi], [xi]])
                    wts = np.array([[1.0], [1.0]])

                self.qpts = pts
                self.qwts = wts
                self.n_quad = len(wts)

            def phi(self, i, *xi):
                if i == 0:
                    return np.array([1 - xi[0]])
                if i == 1:
                    return np.array([xi[0]])

            def d_phi(self, i, *xi):
                if i == 0:
                    return np.array([-1])
                if i == 1:
                    return np.array([1])

        self.test_element = TestElement()

        class TestClassMesh(femtolib.Mesh):
            def __init__(self, nodes, elements, boundary, reference):
                dim = 1
                super().__init__(dim, nodes, elements, boundary, reference)

            def find_element(self, x):
                elt_id = -1
                for iE, elt in enumerate(self.elements):
                    x1, x2 = self.nodes[elt]
                    if _in_segment(x, x1, x2):
                        elt_id = iE
                return elt_id

            def get_ref_coords(self, elt_id, x):
                coords = self.nodes[self.elements[elt_id]]
                return (x - coords[0]) / (coords[1] - coords[0])

            def Jacobian(self, elt_id, xi):
                coords = self.nodes[self.elements[elt_id]]
                h = coords[1] - coords[0]
                return np.array([[h]])

        nodes = np.array([[i] for i in np.linspace(0, 1, 5)])
        elements = np.zeros((nodes.shape[0] - 1, 2), dtype=int)
        for i in range(nodes.shape[0] - 1):
            elements[i] = np.array([int(i), int(i + 1)])
        boundary = np.array([elements[0][0], elements[-1][1]])
        self.test_mesh = TestClassMesh(nodes, elements, boundary, self.test_element)

        def in_dirichlet(i, x):
            if x == 0:
                return True, 0.0
            else:
                return False, None

        def in_neumann(i, x):
            if x == 1:
                return True, 2 * np.pi
            else:
                return False, None

        class TestModel(femtolib.Model):
            def __init__(self, mesh=None, fields=None, exact=None):
                super().__init__(mesh, fields, exact)

                self.dirichlet = in_dirichlet
                self.neumann = in_neumann

            def stiffness_kernel(self, i, j, x, u, v, grad_u, grad_v):
                return np.dot(grad_u, grad_v)

            def load_kernel(self, i, x, v, grad_v):
                return 4 * np.pi * np.pi * np.sin(2 * np.pi * x) * v

        test_uh = femtolib.FunctionSpace(reference=self.test_element, n_dof=len(nodes))
        test_field = [test_uh]
        self.testmodel = TestModel(mesh=self.test_mesh, fields=test_field)

    def test_apply_dirichlet_bc(self):
        self.testmodel.apply_dirichlet_bc()
        ref_dof = np.array([0, np.nan, np.nan, np.nan, np.nan])
        test_dof = self.testmodel.fields[0].dof
        self.assertTrue(((ref_dof == test_dof) | (np.isnan(ref_dof) & np.isnan(test_dof))).all())

    def test_renumber(self):
        self.testmodel.apply_dirichlet_bc()
        self.testmodel.renumber()
        ref_node_idx = np.array([4, 0, 1, 2, 3])
        ref_node_idx_inv = np.array([1, 2, 3, 4, 0])
        self.assertTrue((ref_node_idx == self.testmodel.node_idx).all())
        self.assertTrue((ref_node_idx_inv == self.testmodel.node_idx_inv).all())

    def test_reference_stiffness_matrix(self):
        self.testmodel.apply_dirichlet_bc()
        self.testmodel.renumber()
        ref_val = np.array([[16, -16], [-16, 16]])
        test_val = self.testmodel.reference_stiffness_matrix(1)
        for i in range(test_val.shape[0]):
            for j in range(test_val.shape[1]):
                self.assertAlmostEqual(ref_val[i, j], test_val[i, j])

    def test_reference_load_vector(self):
        self.testmodel.apply_dirichlet_bc()
        self.testmodel.renumber()
        ref_val = np.array([-17.95160322, 17.95160322])
        test_val = self.testmodel.reference_load_vector(0)
        for i in range(test_val.shape[0]):
            self.assertAlmostEqual(ref_val[i], test_val[i])

    def test_assemble_stiffness(self):
        self.testmodel.apply_dirichlet_bc()
        self.testmodel.renumber()
        self.testmodel.assemble_stiffness()
        ref_val = np.array([[32, -16, 0, 0, -16],
                            [-16, 32, -16, 0, 0],
                            [0, -16, 32, -16, 0],
                            [0, 0, -16, 16, 0],
                            [-16, 0, 0, 0, 16]])
        test_val = self.testmodel.stiffness.toarray()
        for i in range(test_val.shape[0]):
            for j in range(test_val.shape[1]):
                self.assertAlmostEqual(ref_val[i, j], test_val[i, j])

    def test_assemble_load(self):
        self.testmodel.apply_dirichlet_bc()
        self.testmodel.renumber()
        self.testmodel.assemble_stiffness()
        self.testmodel.assemble_load()
        ref_val = np.array([42.2778294,  17.9516032, -42.2778294, 0, -17.9516032])
        for i in range(self.testmodel.load.shape[0]):
            self.assertAlmostEqual(ref_val[i], self.testmodel.load[i])

    def test_apply_neumann_bc(self):
        self.testmodel.apply_dirichlet_bc()
        self.testmodel.renumber()
        self.testmodel.assemble_stiffness()
        self.testmodel.assemble_load()
        self.testmodel.apply_neumann_bc()
        ref_val = np.array([42.27782944, 17.95160322, -42.27782944, 6.28318531, -17.95160322])
        for i in range(self.testmodel.load.shape[0]):
            self.assertAlmostEqual(ref_val[i], self.testmodel.load[i])

    def test_solve(self):
        self.testmodel.solve()
        ref_val = np.array([0, 1.51467428, 0.38698423, -1.86268103, -1.46998195])
        for i in range(self.testmodel.fields[0].dof.shape[0]):
            self.assertAlmostEqual(ref_val[i], self.testmodel.fields[0].dof[i])


if __name__ == "__main__":
    unittest.main()
