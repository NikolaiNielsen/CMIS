import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.io as io
sys.path.append('../useful_functions')
import useful_functions as uf
import mesh
import fem

np.set_printoptions(threshold=np.inf)

def D(E=69e9, nu=0.3):
    """
    Returns the Elasticity matrix "D" for a given Young Modulus "E" and Poisson
    ratio "nu".
    """
    return E/(1-nu*nu) * np.array([[1, nu, 0],[nu, 1, 0],[0, 0, (1-nu)/2]])


STEEL_D = D()


def create_B(triangle, area):
    """
    Create the B-matrix for a given element, given the element positions:
    triangle: (3, 2) matrix
    B = S @ N
    S = [[dx, 0],
         [0, dy],
         [dy, dx]]
    N = [[N^i_x, 0,     N^j_x, 0,     N^k_x, 0    ],
         [0,     N^i_y, 0,     N^j_y,     0, N^k_y]]
    """
    diff_vectors = triangle[[2, 0, 1]] - triangle[[1, 2, 0]]
    dx = -diff_vectors[:, 1]
    dy = diff_vectors[:, 0]
    B = (1/2/area) * np.array([[dx[0],     0, dx[1],     0, dx[2],     0],
                               [0,     dy[0],     0, dy[1],     0, dy[2]],
                               [dy[0], dx[0], dy[1], dx[1], dy[2], dx[2]]])
    return B


def create_element_matrix(triangle, area, D=STEEL_D):
    B = create_B(triangle, area)
    Ke = area * B.T @ D @ B
    return Ke


def calc_areas(triangles):
    """
    Calculates the area of an array of triangles.
    Assumes an (n, 3, 2)-array of positions, n triangles, each with 3 vertices,
    each with 2 dimensions
    """
    # The area is given by half the length of the cross product between two of
    # the triangle vectors.

    # First we get the vectors by subracting a permuted array of triangles
    vectors = triangles - triangles[:, [1, 2, 0]]

    # Cross product of 2D vectors is just the z-component, which is also the
    # length of the cross product
    crosses = np.cross(vectors[:, 1], vectors[:, 2])
    area = crosses/2
    return area


def get_global_indices(simplices, d=2):
    # Given a list of 3 vertices and a dimensionality of d, returns the global
    # indices for a given coordinate:
    # order: 'alt'/'stack' - 'alt' is alternating order (x,y,z, x,y,z, ...)
    #                      - 'stack' is stack (x, x, x,..., y,y ,y,...)
    elements = []
    coords = np.arange(d)
    N_verts = np.amax(simplices)+1
    for tri in simplices:
        el = np.array([coords*N_verts + i for i in tri])
        elements.append(np.meshgrid(el, el))
    return elements


def assemble_global_matrix(x, y, simplices, d=2):
    K = np.zeros((x.size*d, x.size*d))

    triangles = mesh.all_triangles(simplices, x, y)
    areas = calc_areas(triangles)

    elements = []
    for tri, area in zip(triangles, areas):
        elements.append(create_element_matrix(tri, area))
    
    for el in elements:
        if not np.allclose(el, el.T):
            print('simplex not symmetric')
            print(el)
    indices = get_global_indices(simplices, d)
    for el, ind in zip(elements, indices):
        x, y = ind
        K[x, y] += el
    return K


def add_boundary(K, x, y, d=2):
    # All vertices on left border (x=0) are fastened
    left = x == np.amin(x)
    coords = np.arange(d)
    N_verts = x.size
    indices = np.arange(N_verts)[left]
    el = np.array([coords*N_verts + i for i in indices])
    # Set all elements (except the diagonal) of row el to 0.
    K[el] = 0
    K[el, el] = 1
    return K


def load_mat(file, return_areas=False):
    mat = io.loadmat(file)
    x = mat['X'].flatten()
    y = mat['Y'].flatten()
    simplices = mat['T'] - 1
    if return_areas:
        areas = mat['A']
        return x, y, simplices, areas
    return x, y, simplices


def global_index_from_bool(bool_array, d=2):
    coords = np.arange(d)
    N = bool_array.size
    indices = np.arange(N)[bool_array]
    el = np.array([coords*N + i for i in indices])
    return el


def simple_ex():
    x, y, simplices = load_mat('data.mat')
    K = assemble_global_matrix(x, y, simplices)
    K = add_boundary(K, x, y)
    f = np.zeros(K.shape[0])
    left = x == np.amin(x)
    bottomright = (x == np.amax(x)) * (y == np.amin(y))
    vert = global_index_from_bool(bottomright).flatten()
    f[vert[1]] = -5e8
    u = np.linalg.solve(K, f)
    dispx = u[:x.size]
    dispy = u[x.size:]
    fig, ax = plt.subplots()
    ax.triplot(x, y, simplices)
    ax.triplot(x+dispx, y+dispy, simplices)
    plt.show()


def func_source(tri):
    return 0


def ex_with_external():
    bottom_force = -5e8
    x, y, simplices = load_mat('data.mat')
    mask = x == np.amin(x)
    vals = 0
    K = fem.assemble_global_matrix(x, y, simplices, create_element_matrix,
                                   D=STEEL_D)
    f = fem.create_f_vector(x, y, simplices, func_source)
    K, f = fem.add_point_boundary(K, f, mask, vals)

    mask = (x == np.amax(x)) * (y == np.amin(y))
    vals = [0, bottom_force]
    f = fem.add_to_source(f, mask, vals)

    u = np.linalg.solve(K, f)
    x_displacement, y_displacement = fem.unpack_u(u)
    x_new = x + x_displacement
    y_new = y + y_displacement
    fig, ax = plt.subplots()
    ax.triplot(x, y, simplices, color='r')
    ax.triplot(x_new, y_new, simplices, color='b')
    plt.show()

ex_with_external()
