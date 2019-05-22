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
                                                    max_area=max_area)
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

x, y, simplices = create_mesh(0, 0.1)

