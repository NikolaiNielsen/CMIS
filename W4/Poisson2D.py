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


x, y, simplices = create_mesh(0, 0.1)
boundary = get_boundary(x, y)
fig, ax = plt.subplots()
ax.scatter(x, y)
ax.scatter(x[boundary], y[boundary], color='r')
plt.show()
