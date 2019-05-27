import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../useful_functions')
import mesh


def element_to_global(simplices, d=2):
    """
    Given a (n, 3) array of vertices and a dimensionality of d, returns a list
    of meshgrids, 2 for each simplex.
    We use the stacked method: (x1, x2, ..., xn, y1, y2, ..., yn)
    """
    elements = []
    coords = np.arange(d)
    N_verts = np.amax(simplices)+1
    for tri in simplices:
        el = np.array([coords*N_verts + i for i in tri])
        elements.append(np.meshgrid(el, el))
    return elements


def get_global_indices(vertices, N, d=2): 
    """
    Returns the global indices for the vertex numbers, given dimensionality d.
    """
    coords = np.arange(d)
    el = np.array([coords*N + i for i in vertices])
    return el


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


def assemble_global_matrix(x, y, simplices, func_elem, d=2, **kwargs):
    """
    Assembles the global matrix K for a FEM system.
    Arguments:
    - x, y: (n,) array of vertex positions
    - simplices: (n, 3) array of vertex numbers that make up each element
    - func_elem: function that calculates the element matrix for a given
                 element, must have signature: func_elem(tri, area, **kwargs)
    - d: dimensionality of the output space (ie, scalar field or vector field)
    - kwargs: keyword arguments passed to func_elem

    Returns:
    - K: (nd, nd) Global matrix for the FEM system
    """
    N = x.size
    K = np.zeros((N*d, N*d))
    triangles = mesh.all_triangles(simplices, x, y)
    areas = calc_areas(triangles)
    indices = element_to_global(simplices, d)

    for tri, area, ind in zip(triangles, areas, indices):
        el = func_elem(tri, area, **kwargs)
        if not np.allclose(el, el.T):
            print('Element matrix not symmetric')
            print(el)

        x, y = ind
        K[x, y] += el
    return K


def create_f_vector(x, y, simplices, func_boundary, d=2, **kwargs):
    """
    Creates the source vector f for the FEM system
    Arguments:
    - x, y: (n,) array of vertex positions
    - simplices: (n, 3) array of vertex numbers that make up each element
    - func_boundary: function that calculates the element matrix for a given
                     element, must have signature:
                     func_boundary(tri, area, **kwargs)
    - d: dimensionality of the output space (ie, scalar field or vector field)
    - kwargs: keyword arguments passed to func_boundary

    Returns:
    - f: (nd, ) Source term vector for the system
    """
    triangles = mesh.all_triangles(simplices, x, y)
    indices = element_to_global(simplices, d)
    f = np.zeros(d*x.size)
    for tri, ind in zip(triangles, indices):
        f_tri = func_boundary(tri, **kwargs)
        f[ind] += f_tri
    return f


def add_point_boundary(K, f, mask, vals, d=2):
    """
    Adds pointwise boundary conditions to the system. Can of course be applied
    multiple times.
    Arguments:
    - K: (nd, nd) global matrix for the FEM system
    - f: (nd,) source term for the system
    - mask: (n,) boolean array of vertices to apply boundary conditions to
    - vals: float or (nd, )-array. values for the source term.

    Returns:
    - K, f: updated matrix and source vector
    """
    vertices = np.where(mask)
    vertices = np.atleast_2d(vertices)
    N = mask.size
    indices = get_global_indices(vertices, N, d)
    K[indices] = 0
    K[indices, indices] = 1
    f[indices] = vals
    return K, f
