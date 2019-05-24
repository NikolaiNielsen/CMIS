import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from progress.bar import Bar
sys.path.append('../useful_functions/')
import useful_functions as uf
import mesh



def create_mesh(min_angle=0, max_area=0.1):
    contours = [np.array([[0, 0], [6, 0], [6, 2], [0, 2]])]
    x, y, simplices = mesh.generate_and_import_mesh(contours,
                                                    min_angle=min_angle,
                                                    max_area=max_area,
                                                    save_files=False,
                                                    print_triangle=False)
    return x, y, simplices


def test_mesh(min_angle=0, max_area=0.1):
    x, y, simplices = create_mesh(min_angle, max_area)
    fig, (ax1, ax2) = plt.subplots(ncols = 2)
    ax1.triplot(x, y, simplices)
    ax1.scatter(x, y)
    ax1.set_aspect('equal')
    ax2 = mesh.plot_quality(simplices, x, y, ax=ax2)
    fig.tight_layout()
    plt.show()


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


def create_element_matrix(triangles):
    areas = calc_areas(triangles)
    diff_vectors = triangles[:, [2, 0, 1]] - triangles[:, [1, 2, 0]]
    y_coords = diff_vectors[:,:,1]
    x_coords = diff_vectors[:,:,0]
    elements = np.zeros((areas.size, 3, 3))
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        px = np.outer(x, x)
        py = np.outer(y, y)
        elements[i, :, :] = (px + py) / (4*areas[i])
    return elements


def get_global_indices(simplices):
    elements = []
    for tri in simplices:
        elements.append(np.meshgrid(tri, tri))
    return elements


def assemble_global_matrix(x, y, simplices):
    K = np.zeros((x.size, x.size))

    triangles = mesh.all_triangles(simplices, x, y)
    elements = create_element_matrix(triangles)
    indices = get_global_indices(simplices)
    for el, ind in zip(elements, indices):
        x, y = ind
        K[x, y] += el
    return K


def create_f_vector(x, y, simplices, c=2):
    triangles = mesh.all_triangles(simplices, x, y)
    f = np.zeros(x.shape)
    areas = calc_areas(triangles)
    for n, simp in enumerate(simplices):
        f[simp] += -c*areas[n]/3
    return f


def add_boundary(K, f, x, y, a=1, b=2):
    left = x == 0
    right = x == 6
    border_ind = np.arange(x.size)[left+right]
    K[border_ind] = 0
    K[border_ind, border_ind] = 1
    f[left] = a
    f[right] = b
    return K, f


def solve_system(min_angle=30, max_area=0.1, c=2):
    x, y, simplices = create_mesh(min_angle, max_area)
    K = assemble_global_matrix(x, y, simplices)
    f = create_f_vector(x, y, simplices, c=c)
    K, f = add_boundary(K, f, x, y)
    u = np.linalg.solve(K, f)
    return x, y, simplices, u


def analytical_solution(x, x0=6, a=1, b=2, c=0):
    return (c/2)*x*x - (c*x0*x0 + 2*a - 2*b)*x/(2*x0) + a


def calc_res(x, u, x0=6, a=1, b=2, c=0):
    left = x == 0
    right = x == x0
    N_border = np.sum(left + right)
    sol = analytical_solution(x, x0, a, b, c)
    res = np.sqrt(np.sum((u-sol)**2))/(x.size-N_border)
    return res, x.size-N_border


def experiment_1(min_max=0.05, max_max=2, n=50, c=2, min_angle=30):
    max_areas = np.logspace(np.log2(min_max), np.log2(max_max), n, base=2)
    results = np.zeros((2, n))
    dof = np.zeros(n)
    n_verts = np.zeros(n)
    qualities = np.zeros((2,n))
    n_zeros = np.zeros(n)
    bar = Bar('Solving', max=n)
    for i, max_area in enumerate(max_areas):
        x, y, simplices, u = solve_system(min_angle=min_angle,
                                          max_area=max_area, c=c)
        _, _, _, (res, _) = fdm(x.size, c=2)
        Q1, Q2 = mesh.calc_quality(simplices, x, y)
        n_verts[i] = x.size
        qualities[0, i] = np.average(Q1)
        qualities[1, i] = np.average(Q2)
        n_zero = (Q1 == 0) + (Q2 == 0)
        n_zeros[i] = np.sum(n_zero)
        results[0, i], dof[i] = calc_res(x, u, c=c)
        results[1, i] = res
        bar.next()
    bar.finish()
    fig, ax = plt.subplots(nrows=3)
    ax = ax.flatten()
    for a in ax:
        a.set_xscale('log')
    ax[0].plot(max_areas, results[0], label='FEM')
    ax[0].plot(max_areas, results[1], label='FDM')
    ax[0].legend()
    ax[1].plot(max_areas, dof)
    ax[1].plot(max_areas, n_verts)
    ax[2].plot(max_areas, qualities.T)
    fig.tight_layout()
    print(np.sum(n_zeros != 0))
    plt.show()


def simple_ex():
    c = 2
    x, y, simplices, u = solve_system(min_angle=30, max_area=0.1, c=c)
    res, _ = calc_res(x, u)

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_trisurf(x, y, u, cmap='coolwarm')
    plt.show()


def index_helper(i, j, m):
    # Returns the 1D index for a 2D matrix, with j being running index (column
    # number), m being column count and i being row number
    return m*i+j


def fdm(N, a=1, b=2, c=2):
    Nx = np.round(np.sqrt(3*N)).astype(int)
    Ny = np.round(np.sqrt(N/3)).astype(int)
    x = np.linspace(0,6, Nx)
    y = np.linspace(0,2, Ny)
    dx = 6/(Nx-1)
    dy = 2/(Ny-1)
    xx, yy = np.meshgrid(x, y)
    n, m = xx.shape
    u = np.zeros(xx.shape)
    K = np.zeros((xx.size, xx.size))
    f = np.zeros(xx.size)
    if Ny < 3:
        return xx, yy, u, (np.nan, 0)
    for i in range(n):
        for j in range(m):

            node_index = index_helper(i, j, m)
            # print(i, j, node_index)
            right_edge = j == 0
            left_edge = j == m-1
            top_edge = i == n-1
            bottom_edge = i == 0
            border = right_edge + top_edge + left_edge + bottom_edge
            
            if not border:
                i_in_stencil = np.array([i,   i,   i, i+1, i-1])
                j_in_stencil = np.array([j, j+1, j-1,   j,   j])
                stencil_indices = index_helper(i_in_stencil, j_in_stencil, m)
                stencil_values = np.array([- 2/dx/dx - 2/dy/dy,
                                        1/dx/dx,
                                        1/dx/dx,
                                        1/dy/dy,
                                        1/dy/dy])
                K[node_index, stencil_indices] = stencil_values
                f[node_index] = c
            elif right_edge or left_edge:
                K[node_index, node_index] = 1
                f[node_index] = right_edge * a + left_edge * b
            elif top_edge:
                i_in_stencil = np.array([i,   i,   i, i-1, i-2])
                j_in_stencil = np.array([j, j+1, j-1,   j,   j])
                stencil_indices = index_helper(i_in_stencil, j_in_stencil, m)
                stencil_values = np.array([- 2/dx/dx + 1/dy/dy,
                                           1/dx/dx,
                                           1/dx/dx,
                                           -2/dy/dy,
                                           1/dy/dy])
                K[node_index, stencil_indices] = stencil_values
                f[node_index] = c
            elif bottom_edge:
                i_in_stencil = np.array([i,   i,   i, i+1, i+2])
                j_in_stencil = np.array([j, j+1, j-1,   j,   j])
                stencil_indices = index_helper(i_in_stencil, j_in_stencil, m)
                stencil_values = np.array([- 2/dx/dx + 1/dy/dy,
                                           1/dx/dx,
                                           1/dx/dx,
                                           -2/dy/dy,
                                           1/dy/dy])
                K[node_index, stencil_indices] = stencil_values
                f[node_index] = c
    U = np.linalg.solve(K, f)
    u = U.reshape(xx.shape)

    res = calc_res(xx, u, a=a, b=b, c=c)

    return xx, yy, u, res

experiment_1(n=20)
