
#%%
from scipy import interpolate, ndimage
import imageio
from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
from progress.bar import Bar
import timeit
from functools import partial as part
sys.path.append('../')
import quality_measures as qa
import useful_functions as uf


def import_data(name='example.bmp', Nverts=500, threshold=200, invert=False):
    sdf, im = uf.grey_to_sdf(name, ghosts=False, threshold=threshold,
                             invert=invert)
    Ny, Nx = sdf.shape
    border = 0.5
    X = np.random.uniform(border, Nx-1-border, Nverts)
    Y = np.random.uniform(border, Ny-1-border, Nverts)
    Gx, Gy = np.meshgrid(np.arange(Nx), np.arange(Ny))
    x = Gx[0, :]
    y = Gy[:, 0]
    sdf_spline = interpolate.RectBivariateSpline(y, x, sdf)
    return Gx, Gy, sdf, X, Y, im, sdf_spline


def push_points_inside(x, y, sdf_spline):
    d =  sdf_spline.ev(y, x)
    nx = sdf_spline.ev(y, x, dy=1)
    ny = sdf_spline.ev(y, x, dx=1)
    nx = d*nx
    ny = d*ny
    mask = d > 0
    x[mask] = x[mask] - nx[mask]
    y[mask] = y[mask] - ny[mask]
    return x,y

def push_fully_inside(x, y, sdf_spline, max_tries=3):
    d = sdf_spline.ev(y, x)

    mask = d > 0
    counter_max = max_tries + 1
    for _ in range(counter_max):
        if np.sum(mask):
            x, y = push_points_inside(x, y, sdf_spline)
            d = sdf_spline.ev(y, x)
            mask = d > 0
        else:
            return x, y
    return x, y


def gen_points_in_triangle(v, N=10):
    """
    Generate N points uniformly distributed inside a triangle, defined by
    vertices v.

    v is assumed to have the shape (3, 2)

    Formula from
    http://www.cs.princeton.edu/~funk/tog02.pdf
    section 4.2, page 8.
    """ 
    r1 = np.sqrt(np.random.uniform(size=N))
    r2 = np.random.uniform(size=N)
    a = 1-r1
    b = r1*(1-r2)
    c = r1*r2
    r = np.array((a,b,c))
    points = v.T @ r
    return points


def verts_from_simplex(simplex, x, y):
    """
    Get the coordinates of the vertices in a given simplex.
    inputs:
    - simplex - list, len 3 - Integers referencing the vertices
    - x, y - vector of positions for vertices
    outputs:
    - verts: (3,2) array of positions for the vertices in the simplex
    """
    verts = np.array((x[simplex], y[simplex])).T
    return verts


def all_triangles(simplices, x, y):
    """
    Generates a (n, 3, 2) array of positions for all vertices in the n
    simplices
    """
    N_simplices = simplices.shape[0]
    verts = np.zeros((N_simplices, 3, 2))
    for i, simplex in enumerate(simplices):
        verts[i, :, :] = verts_from_simplex(simplex, x, y)
    return verts


def discard_outside_triangles(simplices, x, y, sdf_spline):
    """
    runs over the list of triangles, and figures out whether they are outside
    or inside the object
    """
    verts = all_triangles(simplices, x, y)
    dims = verts.shape
    N_points = 20
    triangle_points = np.zeros((dims[0], N_points, 2))
    triangles_inside = np.zeros(dims[0])
    d_threshold = 3
    for i, vert in enumerate(verts):
        # First we generate 10 points for each triangle and throw them in the
        # array
        points = gen_points_in_triangle(vert, N_points).T
        triangle_points[i, :, :] = points
        d_triangle = sdf_spline.ev(points[:,1], points[:,0])
        d_inside = d_triangle <= d_threshold
        n_inside = np.sum(d_inside)
        inside = n_inside == N_points
        triangles_inside[i] = inside

    # reshape array for easy d-calculation with splines
    triangle_points_reshaped = triangle_points.reshape((dims[0] * N_points, 2))
    # Calculate d for all points
    d = sdf_spline.ev(triangle_points_reshaped[:, 1],
                      triangle_points_reshaped[:, 0])
    
    d_reshaped = d.reshape((dims[0], N_points))
    d_inside = d_reshaped <= d_threshold
    n_inside = np.sum(d_inside, axis=1)
    triangles_to_keep = n_inside == N_points
    return simplices[triangles_to_keep], simplices[~ triangles_to_keep]
        

def find_all_neighbours(simplices, n):
    """
    Finds all neighbouring vertices to the n'th vertex
    """

    neighbours = simplices == n
    neighbour_mask = np.sum(neighbours, axis=1).astype(bool)
    neighbouring_simplices = simplices[neighbour_mask]
    unique_verts = np.unique(neighbouring_simplices).flatten()
    unique_verts = unique_verts[unique_verts != n]
    return unique_verts


def calc_com(vertices, x, y):
    """
    Calculates the center of mass (with masses assumed equal), for the input
    vertices
    """
    N = vertices.size
    x_verts = x[vertices]
    y_verts = y[vertices]
    x_com = np.sum(x_verts)/N
    y_com = np.sum(y_verts)/N
    return np.array((x_com, y_com))


def update_positions(simplices, x, y, tau=0.5):
    # first we generate the list of vertices:
    N = x.size
    vertices = np.arange(N)
    positions = np.array((x, y)).T
    com_positions = np.zeros(positions.shape)
    mask = np.zeros(N)
    for i in vertices:
        neighbors = find_all_neighbours(simplices, i)
        mask[i] = neighbors.size
        if mask[i]:
            com_positions[i, :] = calc_com(neighbors, x, y)
    mask = mask.astype(bool)
    outside = np.sum(mask == False)
    new_pos = positions - tau * (com_positions - positions)
    x_new, y_new = new_pos[mask,:].T
    # print(outside, x_new.shape)
    return x_new, y_new


def create_mesh(name, N_verts=500, threshold=200,
                invert=False, N_iter=10, N_tries=3, tau=0.3):
    """
    Creates a mesh of a given image. 
    """
    _, _, _, X, Y, im, sdf_spline = import_data(name, N_verts,
                                                threshold, invert)
    
    X, Y = push_fully_inside(X, Y, sdf_spline, max_tries=N_tries)
    points = np.array((X, Y)).T
    T = Delaunay(points)
    simplices = T.simplices
    simplices, _ = discard_outside_triangles(simplices, X, Y, sdf_spline)
    for i in range(N_iter):
        X, Y = update_positions(simplices, X, Y, tau=tau)
        X, Y = push_fully_inside(X, Y, sdf_spline, max_tries=N_tries)
        points = np.array((X, Y)).T
        simplices = Delaunay(points).simplices
        simplices, _ = discard_outside_triangles(simplices, X, Y, sdf_spline)
    return X, Y, simplices, im


#%% project particles
X, Y, simplices, im = create_mesh('example.bmp', N_verts=500, N_iter=10)
# Gx, Gy, sdf, X, Y, im, sdf_spline = import_data()
# v = np.array(((50,70),(200,10),(200,150)))
# np.set_printoptions(threshold=np.inf)

# points = gen_points_in_triangle(v, N=2000)
# d = sdf_spline.ev(points[:,1], points[:,0])
# inside = d <= 0

fig, ax = plt.subplots()
ax.imshow(im, cmap='Greys_r')
ax.triplot(X, Y, simplices, color='b')
ax.scatter(X, Y, color='b', s=5)
# ax.scatter(v[:,0], v[:,1])
# ax.scatter(points[inside,0], points[inside,1], s=5, color='b')
# ax.scatter(points[~inside,0], points[~inside,1], s=5, color='r')

plt.show()



#%%
# a = np.array(((1,1),(2,4),(5,2)))
# points = gen_points_in_triangle(a, N=10)
# x, y = points
# fig, ax = plt.subplots()
# # ax.plot(a[[0,1],0], a[[0,1],1], 'b')
# # ax.plot(a[[0,2],0], a[[0,2],1], 'b')
# # ax.plot(a[[1,2],0], a[[1,2],1], 'b')
# # ax.scatter(a[:,0], a[:,1], s=36)
# ax.scatter(x,y, s=1)
# plt.show()
