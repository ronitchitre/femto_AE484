import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix


def create_mesh_unit_square(n_side=4):
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
    bdy = np.array(bdy, dtype=int)
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
    return nodes, elements, bdy


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


def _reassign_nbc(nbc, node_idx, node_idx_inv):
    nbc_count = 0
    for i, val in nbc:
        nbc[nbc_count][0] = node_idx[i]
        nbc_count += 1


def renumber_mesh_dof(nodes, elements, nbc, dof):
    node_idx = _renumber_node_indices(dof)
    node_idx_inv = _compute_inverse_node_indices(node_idx)
    _reassign_nodes(nodes, node_idx, node_idx_inv)
    _renumber_elements(elements, node_idx, node_idx_inv)
    if nbc is not None:
        _reassign_nbc(nbc, node_idx, node_idx_inv)
    _reassign_dof(dof, node_idx, node_idx_inv)


def apply_bc(nodes, elements, bdy, in_bc):
    bc = []
    for i in bdy:
        on_bdy, val = in_bc(*nodes[i])
        if on_bdy:
            bc.append([i, val])
    return bc


def get_GL_pts_wts_1d(n_quad):
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


def get_GL_pts_wts(n_quad, quad_type='area'):
    if quad_type == 'area':
        if n_quad == 1:
            pts = np.array([[1/3, 1/3]])
            wts = np.array([1.0])
        elif n_quad == 2:
            pts = np.array([[1/6, 1/6],
                            [2/3, 1/6],
                            [1/6, 2/3]])
            wts = np.array([1/3, 1/3, 1/3])
        elif n_quad == 3:
            pts = np.array([[1/3, 1/3],
                            [1/5, 1/5],
                            [3/5, 1/5],
                            [1/5, 3/5]])
            wts = np.array([-27/48, 25/48, 25/48, 25/48])
        else:
            raise Exception("Invalid quadrature order!")
    elif quad_type == 'duffy':
        pts_x, wts_x = get_GL_pts_wts_1d(n_quad)
        pts_x = (1 + pts_x)/2
        wts_x = 2*wts_x
        pts = np.zeros((n_quad*n_quad, 2))
        wts = np.zeros(n_quad*n_quad)
        for j in range(n_quad):
            for i in range(n_quad):
                pts[j*n_quad + i, 0] = pts_x[i]
                pts[j*n_quad + i, 1] = pts_x[j]*(1 - pts_x[i])
                wts[j*n_quad + i] = wts_x[i]*wts_x[j]*(1 - pts_x[i])
    else:
        raise Exception(
            "Invalid quadrature type: use either 'area' or 'duffy'"
        )

    return pts, wts


def integrate_GL_quad(g, n_quad, quad_type):
    pts, wts = get_GL_pts_wts(n_quad, quad_type)
    intgl = 0.0
    for i in range(len(wts)):
        intgl += wts[i] * g(*pts[i])
    return intgl


def phi(idx, xi, eta):
    if idx == 0:
        return (1 - xi - eta)
    elif idx == 1:
        return xi
    elif idx == 2:
        return eta
    else:
        raise Exception("Invalid shape function index")


def d_phi(idx_phi, idx_x, xi, eta):
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


def _reference_stiffness_matrix(n_quad, coords, quad_type='area'):
    x1, x2, x3 = coords[:, 0]
    y1, y2, y3 = coords[:, 1]
    J = np.zeros((2, 2))
    J[0, 0] = x2 - x1
    J[0, 1] = x3 - x1
    J[1, 0] = y2 - y1
    J[1, 1] = y3 - y1
    vol = np.abs(np.linalg.det(J))
    J = np.linalg.inv(J).transpose()

    ke = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            def g(xi, eta):
                df1 = np.array([d_phi(i, 0, xi, eta), d_phi(i, 1, xi, eta)])
                df2 = np.array([d_phi(j, 0, xi, eta), d_phi(j, 1, xi, eta)])
                return np.dot((J @ df1), (J @ df2))
            ke[i, j] = integrate_GL_quad(g, n_quad=n_quad, quad_type=quad_type)

    ke *= vol
    return ke


def compute_stiffness_matrix(nodes, elements, n_quad, quad_type):
    M = np.shape(nodes)[0]

    II = []
    JJ = []
    V = []

    for elt in elements:
        coords = nodes[elt]
        ke = _reference_stiffness_matrix(n_quad, coords, quad_type)

        for i in range(3):
            for j in range(3):
                II.append(elt[i])
                JJ.append(elt[j])
                V.append(ke[i, j])

    K = coo_matrix((V, (II, JJ)), shape=(M, M))
    return K


def _compute_reference_load_vector(n_quad, coords, f, quad_type):
    x1, x2, x3 = coords[:, 0]
    y1, y2, y3 = coords[:, 1]
    vol = np.abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
    fe = np.zeros(3)

    for j in range(3):
        def g(xi, eta):
            x = 0.0
            for i in range(3):
                x += phi(i, xi, eta)*coords[i, 0]

            y = 0.0
            for i in range(3):
                y += phi(i, xi, eta)*coords[i, 1]

            return f(x, y)*phi(j, xi, eta)

        fe[j] = integrate_GL_quad(g, n_quad=n_quad, quad_type=quad_type)

    fe *= vol
    return fe


def compute_load_vector(nodes, elements, n_quad, f, quad_type):
    M = np.shape(nodes)[0]
    F = np.zeros(M)

    if f is None:
        return F
    else:
        for elt in elements:
            coords = nodes[elt]
            fe = _compute_reference_load_vector(n_quad, coords, f, quad_type)

            for i in range(3):
                F[elt[i]] += fe[i]

        return F


def _get_num_unknowns(dof):
    return np.size(np.where(np.isnan(dof)))


def solve_bvp(nodes, elements, dbc,
              nbc=None, n_quad=2, f=None, quad_type='area'):
    dof = create_dof(nodes, dbc)
    renumber_mesh_dof(nodes, elements, nbc, dof)

    K = compute_stiffness_matrix(nodes, elements, n_quad, quad_type)
    K = K.tocsr()
    F = compute_load_vector(nodes, elements, n_quad, f, quad_type)

    if nbc is not None:
        for i, val in nbc:
            F[i] += val

    N = _get_num_unknowns(dof)
    U_dbc = dof[N:]
    U = spsolve(K[:N, :N], F[:N] - K[:N, N:] @ U_dbc)

    dof[:N] = U
    return dof


def plot_fem_soln(nodes, dof, u_exact=None, n_plot=11):
    if u_exact is not None:
        xs = np.linspace(0, 1, n_plot)
        ys = np.linspace(0, 1, n_plot)
        xs, ys = np.meshgrid(xs, ys)
        us_exact = u_exact(xs, ys)

    plt.figure(figsize=(6, 6))
    ax = plt.axes(projection='3d')

    surf = ax.plot_trisurf(nodes[:, 0], nodes[:, 1], dof,
                           color='red', label='FEM')
    surf._facecolors2d = surf._facecolors  # You may not need this line!
    surf._edgecolors2d = surf._edgecolors  # You may not need this line!

    if u_exact is not None:
        surf = ax.plot_wireframe(xs, ys, us_exact,
                                 linewidth=1, color='black', label='Exact')
        surf._facecolors2d = surf._facecolors  # You may not need this line!
        surf._edgecolors2d = surf._edgecolors  # You may not need this line!

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.legend()
    plt.show()


# This is a _very slow_ function!
def _find_element(x, y, nodes, elements):
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
        l1, l2 = np.linalg.inv(J) @ np.array([x - x1, y - y1])
        if l1 >= 0 and l2 >= 0 and (l1 + l2) <= 1:
            elt_id = iE
            break
    return elt_id


def fe_interpolate(nodes, elements, dof, x, y):
    elt_id = _find_element(x, y, nodes, elements)
    coords = nodes[elements[elt_id]]
    x1, x2, x3 = coords[:, 0]
    y1, y2, y3 = coords[:, 1]
    J = np.zeros((2, 2))
    J[0, 0] = x2 - x1
    J[0, 1] = x3 - x1
    J[1, 0] = y2 - y1
    J[1, 1] = y3 - y1
    uh = 0.0
    xi, eta = np.linalg.inv(J) @ np.array([(x - x1), (y - y1)])
    for i in range(3):
        uh += phi(i, xi, eta)*dof[elements[elt_id][i]]
    return uh


def compute_L2_error(nodes, elements, dof, u_exact, n_quad=100):
    xs = np.linspace(0, 1, (n_quad + 1))
    ys = np.linspace(0, 1, (n_quad + 1))
    xs, ys = np.meshgrid(xs, ys)
    xs = xs.flatten().reshape((xs.size, 1))
    ys = ys.flatten().reshape((ys.size, 1))
    zs = np.hstack((xs, ys))
    us_exact = np.array([u_exact(*z) for z in zs])
    us_fem = np.array([fe_interpolate(nodes, elements, dof, *z) for z in zs])
    err_L2 = np.mean((us_exact - us_fem)**2)
    return np.sqrt(err_L2)


def compute_L2_error_centers(nodes, elements, dof, u_exact):
    err_L2 = 0.0
    for elt in elements:
        coords = nodes[elt]
        xc = np.sum(coords[:, 0])/3
        yc = np.sum(coords[:, 1])/3
        ue = u_exact(xc, yc)
        xi = 1/3
        eta = 1/3
        uh = 0.0
        for i in range(3):
            uh += phi(i, xi, eta)*dof[elt[i]]
        err_L2 += (uh - ue)**2
    err_L2 /= len(elements)
    return np.sqrt(err_L2)
