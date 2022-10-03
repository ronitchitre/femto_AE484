import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix

pi = np.pi


def create_mesh_1d_uniform(n_elt):
    N = n_elt - 1
    nodes = np.linspace(0, 1, N+2)
    elements = []
    for i in range(n_elt):
        elements.append([i, i+1])
    dbc = [[0, 0.0], [N+1, 0.0]]
    return nodes, elements, dbc


def create_dof(nodes, dbc):
    n_nodes = np.shape(nodes)[0]
    dof = np.array([np.nan for _ in range(n_nodes)])
    for bc in dbc:
        dof[bc[0]] = bc[1]
    return dof


def _renumber_node_indices(dof):
    n_idx = np.zeros(np.shape(dof)[0], dtype=int)
    node_count = 0

    for i, d in enumerate(dof):
        if np.isnan(d):
            n_idx[i] = node_count
            node_count += 1

    for i, d in enumerate(dof):
        if not np.isnan(d):
            n_idx[i] = node_count
            node_count += 1

    return n_idx


def _compute_inverse_node_indices(node_idx):
    node_idx_inv = np.zeros_like(node_idx, dtype=int)

    idx_count = 0
    for n in node_idx:
        node_idx_inv[n] = idx_count
        idx_count += 1

    return node_idx_inv


def _reassign_nodes(nodes, node_idx, node_idx_inv):
    nodes_renum = np.zeros_like(nodes)
    n_nodes = np.shape(nodes)[0]
    for i in range(n_nodes):
        nodes_renum[i] = nodes[node_idx_inv[i]]
    nodes[:] = nodes_renum[:]


def _renumber_elements(elements, node_idx, node_idx_inv):
    n_elts = np.shape(elements)[0]
    n_nodes_elt = np.shape(elements)[1]
    for iE in range(n_elts):
        for j in range(n_nodes_elt):
            elements[iE][j] = node_idx[elements[iE][j]]


def _reassign_dof(dof, node_idx, node_idx_inv):
    dof_renum = np.zeros_like(dof)
    n_dof = np.shape(dof)[0]
    for i in range(n_dof):
        dof_renum[i] = dof[node_idx_inv[i]]
    dof[:] = dof_renum[:]


def renumber_mesh_dof(nodes, elements, dof):
    node_idx = _renumber_node_indices(dof)
    node_idx_inv = _compute_inverse_node_indices(node_idx)
    _reassign_nodes(nodes, node_idx, node_idx_inv)
    _renumber_elements(elements, node_idx, node_idx_inv)
    _reassign_dof(dof, node_idx, node_idx_inv)


def get_GL_pts_wts(n_quad):
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


def integrate_GL_quad(g, a=0, b=1, n_quad=2):
    pts, wts = get_GL_pts_wts(n_quad)
    intgl = 0.0

    for i in range(n_quad):
        x = (a + b)/2 + (b - a)*pts[i]/2
        intgl += wts[i] * g(x)

    intgl *= (b - a)/2
    return intgl


def phi(idx, xi):
    if idx == 0:
        return (1 - xi)
    elif idx == 1:
        return xi
    else:
        raise Exception("Invalid index")


def d_phi(idx, xi):
    if idx == 0:
        return -1
    elif idx == 1:
        return 1
    else:
        raise Exception("Invalid index")


def f(x):
    return 4*pi*pi*np.sin(2*pi*x)


def u_exact(x):
    return np.sin(2*pi*x)


def _reference_stiffness_matrix(n_quad):
    ke = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            def g(xi): return d_phi(i, xi)*d_phi(j, xi)
            ke[i, j] = integrate_GL_quad(g, a=0, b=1, n_quad=n_quad)

    return ke


def compute_stiffness_matrix(nodes, elements, n_quad):
    M = np.shape(nodes)[0]

    II = []
    JJ = []
    V = []

    for elt in elements:
        ke = _reference_stiffness_matrix(n_quad)
        he = nodes[elt[1]] - nodes[elt[0]]

        for i in range(2):
            for j in range(2):
                II.append(elt[i])
                JJ.append(elt[j])
                V.append(ke[i, j]/he)

    K = coo_matrix((V, (II, JJ)), shape=(M, M))
    return K


def _compute_reference_load_vector(xe, n_quad):
    fe = np.zeros(2)

    for i in range(2):
        def g(xi): return f(xe[0] + (xe[1] - xe[0])*xi)*phi(i, xi)
        fe[i] = integrate_GL_quad(g, a=0, b=1, n_quad=n_quad)

    return fe


def compute_load_vector(nodes, elements, n_quad):
    M = np.shape(nodes)[0]
    F = np.zeros(M)
    n_nodes_elt = np.shape(elements)[1]

    for elt in elements:
        xe = np.array([nodes[elt[i]] for i in range(n_nodes_elt)])
        fe = _compute_reference_load_vector(xe, n_quad)
        he = he = nodes[elt[1]] - nodes[elt[0]]

        for i in range(2):
            F[elt[i]] += he*fe[i]

    return F


def _get_num_unknowns(dof):
    return np.size(np.where(np.isnan(dof)))


def solve_bvp(nodes, elements, dbc, n_quad):
    dof = create_dof(nodes, dbc)
    renumber_mesh_dof(nodes, elements, dof)

    K = compute_stiffness_matrix(nodes, elements, n_quad)
    K = K.tocsr()
    F = compute_load_vector(nodes, elements, n_quad)

    N = _get_num_unknowns(dof)
    U_dbc = dof[N:]
    U = spsolve(K[:N, :N], F[:N] - K[:N, N:] @ U_dbc)

    dof[:N] = U
    return dof


def plot_fem_soln(nodes, elements, dof):
    xs = np.linspace(0, 1, 40)
    us_exact = u_exact(xs)

    plt.plot(xs, us_exact, '-', color='gray', lw=6)
    plt.plot(nodes, dof, 'bo', markersize=3, label='FEM')

    for elt in elements:
        n1, n2 = elt
        plt.plot([nodes[n1], nodes[n2]], [dof[n1], dof[n2]], 'b-', lw=2)

    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.show()


def _find_element(x, nodes, elements):
    elt_id = -1
    for iE, elt in enumerate(elements):
        n1, n2 = elt
        if x >= nodes[n1] and x <= nodes[n2]:
            elt_id = iE
            break
    return elt_id


def fe_interpolate(nodes, elements, dof, x):
    elt_id = _find_element(x, nodes, elements)
    n1, n2 = elements[elt_id]
    he = nodes[n2] - nodes[n1]
    uh = dof[n1] + (dof[n2] - dof[n1])*(x - nodes[n1])/he
    return uh


def compute_L2_error(nodes, elements, dof, n_quad=100):
    xs = np.linspace(0, 1, (n_quad + 1))
    us_exact = u_exact(xs)
    us_fem = np.array([fe_interpolate(nodes, elements, dof, x) for x in xs])
    err_L2 = np.mean((us_exact - us_fem)**2)
    return np.sqrt(err_L2)
