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


def load_mat(file, return_areas=False):
    mat = io.loadmat(file)
    x = mat['X'].flatten()
    y = mat['Y'].flatten()
    simplices = mat['T'] - 1
    if return_areas:
        areas = mat['A']
        return x, y, simplices, areas
    return x, y, simplices


def func_source(tri):
    return 0


def ex_with_external():
    bottom_force = -5e7
    x, y, simplices = load_mat('data.mat')
    mask1 = x == np.amin(x)
    vals1 = 0
    elem_dict = dict(D=STEEL_D)

    mask2 = (x == np.amax(x)) * (y == np.amin(y))
    vals2 = [0, bottom_force]

    x_displacement, y_displacement = fem.FEM(x, y, simplices,
                                             create_element_matrix,
                                             func_source,
                                             mask1, vals1, mask2, vals2,
                                             elem_dict=elem_dict)
    x_new = x + x_displacement
    y_new = y + y_displacement
    fig, ax = plt.subplots()
    ax.triplot(x, y, simplices, color='r')
    ax.triplot(x_new, y_new, simplices, color='b')

    triangles_before = mesh.all_triangles(simplices, x, y)
    triangles_after = mesh.all_triangles(simplices, x_new, y_new)
    area_before = np.sum(fem.calc_areas(triangles_before))
    area_after = np.sum(fem.calc_areas(triangles_after))
    print(area_before)
    print(area_after)
    plt.show()
