import numpy as np
import matplotlib.pyplot as plt
import sys
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


def calc_boundary_value(x, y, a=1, b=2, x0=0, y0=1):
    return (b-a)/(6) * (x-x0) + y0


def get_boundary(x, y):
    left = x == 0
    top = y == 2
    right = x == 6
    bottom = y == 0
    boundary = left + top + right + bottom
    return boundary


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


x, y, simplices = create_mesh(0, None)
# K = assemble_global_matrix(x, y, simplices)
# non_zero = np.sum(~(K==0))
fig, ax = plt.subplots()
ax.triplot(x, y, simplices)
ax.scatter(x, y)
plt.show()
