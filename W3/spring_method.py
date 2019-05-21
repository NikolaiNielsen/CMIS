
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
import pandas as pd
import subprocess
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


def discard_outside_triangles(simplices, x, y, sdf_spline, N_points=10):
    """
    runs over the list of triangles, and figures out whether they are outside
    or inside the object
    """
    verts = all_triangles(simplices, x, y)
    dims = verts.shape
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
        

def find_all_neighbours(simplices, n, include_self=False):
    """
    Finds all neighbouring vertices to the n'th vertex
    """

    neighbours = simplices == n
    neighbour_mask = np.sum(neighbours, axis=1).astype(bool)
    neighbouring_simplices = simplices[neighbour_mask]
    unique_verts = np.unique(neighbouring_simplices).flatten()
    if not include_self:
        unique_verts = unique_verts[unique_verts != n]
    return unique_verts


def calc_com(vertices, x, y, n, m_max=5):
    """
    Calculates the center of mass (with masses assumed equal), for the input
    vertices
    """
    x_c = x[n]
    y_c = y[n]


    N = vertices.size
    x_verts = x[vertices]
    y_verts = y[vertices]

    x_dist = x_verts - x_c
    y_dist = y_verts - y_c
    r = np.sqrt(x_dist**2 + y_dist**2)
    r_max = np.amax(r)
    r_min = np.amin(r)
    dy = m_max - 1
    dx = r_max - r_min
    m = dy/dx * (r - r_min) + 1

    x_com = np.sum(m * x_verts)/np.sum(m)
    y_com = np.sum(m * y_verts)/np.sum(m)
    return np.array((x_com, y_com))


def update_positions(simplices, x, y, tau=0.5, include_self=False, m_max=5):
    # first we generate the list of vertices:
    N = x.size
    vertices = np.arange(N)
    positions = np.array((x, y)).T
    com_positions = np.zeros(positions.shape)
    mask = np.zeros(N)
    for i in vertices:
        neighbors = find_all_neighbours(simplices, i, include_self)
        mask[i] = neighbors.size
        if mask[i]:
            com_positions[i, :] = calc_com(neighbors, x, y, i, m_max)
    mask = mask.astype(bool)
    # outside = np.sum(mask == False)
    new_pos = positions - tau * (com_positions - positions)
    x_new, y_new = new_pos[mask,:].T
    # print(outside, x_new.shape)
    return x_new, y_new


def create_mesh(name, N_verts=500, threshold=200, include_self=False,
                invert=False, N_iter=10, N_tries=3, tau=0.3, N_points=10,
                m_max=5):
    """
    Creates a mesh of a given image. 
    """
    _, _, _, X, Y, im, sdf_spline = import_data(name, N_verts,
                                                threshold, invert)
    
    X, Y = push_fully_inside(X, Y, sdf_spline, max_tries=N_tries)
    points = np.array((X, Y)).T
    T = Delaunay(points)
    simplices = T.simplices
    # print(simplices.shape)
    simplices, _ = discard_outside_triangles(simplices, X, Y, sdf_spline,
                                             N_points)
    for i in range(N_iter):
        # print(simplices.shape)
        X, Y = update_positions(simplices, X, Y, tau=tau,
                                include_self=include_self, m_max=m_max)
        X, Y = push_fully_inside(X, Y, sdf_spline, max_tries=N_tries)
        points = np.array((X, Y)).T
        simplices = Delaunay(points).simplices
        simplices, _ = discard_outside_triangles(simplices, X, Y, sdf_spline,
                                                 N_points)
    # print(simplices.shape)
    return X, Y, simplices, im


def calc_quality(simplices, x, y):
    """
    A function to calculate the quality of the mesh, based on two quality
    measures
    """
    triangles = all_triangles(simplices, x, y)
    l = qa.calc_side_lengths(triangles)
    A = qa.calc_area_of_triangle(l)
    Q1 = qa.calc_min_angle_norm(A, l)
    Q2 = qa.calc_aspect_ratio_norm(A, l)
    return Q1, Q2


def plot_quality(simplices, x, y, N_bins=20, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        return_fig = True
    else:
        return_fig = False
    Q1, Q2 = calc_quality(simplices, x, y)
    ax.hist(Q1, N_bins, range=(0,1), histtype='step', label='Q1 - min angle')
    ax.hist(Q2, N_bins, range=(0,1), histtype='step', label='Q2 - aspect')
    ax.legend()
    if return_fig:
        return fig, ax
    else:
        return ax


def read_poly(name='example.poly'):
    df = pd.read_table(name, header=None, delimiter=r'\s+', comment='#')
    rows_with_nan = df.isnull().any(axis=1)
    id_rows = np.arange(rows_with_nan.size)
    id_rows = id_rows[rows_with_nan]
    # print(id_rows)
    vertices = df.values[1:id_rows[0], 1:3]
    segments = df.values[id_rows[0]+1:id_rows[1], 1:3]


def read_from_triangle(name='example.1'):
    ele_file = name + '.ele'
    node_file = name + '.node'
    simplices = read_ele(ele_file)
    vertices = read_node(node_file)
    x, y = vertices.T
    return x, y, simplices


def read_ele(name='example.1.ele'):
    df = pd.read_table(name, header=1, delimiter=r'\s+', comment='#')
    rows_with_nan = df.isnull().any(axis=1)
    id_rows = np.arange(rows_with_nan.size)
    id_rows = id_rows[rows_with_nan]
    # print(id_rows)
    simplices = df.values[:, 1:]
    return simplices - 1


def read_node(name='example.1.node'):
    df = pd.read_table(name, header=None, delimiter=r'\s+', comment='#')
    rows_with_nan = df.isnull().any(axis=1)
    id_rows = np.arange(rows_with_nan.size)
    id_rows = id_rows[rows_with_nan]
    # print(id_rows)
    vertices = df.values[1:, 1:3]
    return vertices


def write_to_poly(vert_list, holes=[], outfile='example.poly'):
    """
    Write a list of vertices to a .poly file, for Triangle to read
    """

    # First we get the relevant number of vertices, contours and segments
    vert_ids = []
    N_verts = 0
    for i in vert_list:
        start_vert = N_verts + 1
        current_N = i.shape[0]
        current_vert_ids = list(range(start_vert, current_N + start_vert))
        current_vert_ids.append(start_vert)
        vert_ids.append(current_vert_ids)
        N_verts += current_N
    N_segments = N_verts

    with open(outfile, 'w') as f:
        # Add a running count of vertices and boundary markers
        bound_num = 1
        # Write first line: N_verts, dimension, N_attributes, boundary markers?
        f.write(f'{N_verts} 2 0 1\n')

        # Write every vertex 
        for contour, ids in zip(vert_list, vert_ids):
            x, y = contour.T
            N = x.size
            for n in range(N):
                f.write(f'{ids[n]} {x[n]} {y[n]} {bound_num}\n')
            bound_num += 1

        # write every segment
        seg_num = 1
        f.write(f'{N_segments} 1\n')
        for ids in vert_ids:
            N = len(ids)
            for n in range(N-1):
                f.write(f'{ids[n]} {ids[n]} {ids[n+1]} {seg_num}\n')
            seg_num += 1

        # Write holes
        N_holes = len(holes)
        f.write(f'{N_holes}\n')
        for n in range(N_holes):
            f.write(f'{n+1} {holes[n, 0]} {holes[n, 1]}')


def get_contour_from_pic(name='example.bmp', N=100):
    Gx, Gy, sdf, _, _, _, _ = import_data(name)
    fig, ax = plt.subplots()
    contour = ax.contour(Gx, Gy, sdf, levels=0)
    contour_sets = contour.collections
    filled_sets = []
    for set_ in contour_sets:
        if len(set_.get_paths()):
            filled_sets.append(set_.get_paths())
    contours = []
    for i in filled_sets:
        for cont in i:
            contours.append(cont.vertices)
    
    new_contours = []
    for i in contours:
        x, y = i.T
        
        xp = np.arange(x.size)
        x_new = np.interp(np.linspace(0, x.size, N), xp, x)
        y_new = np.interp(np.linspace(0, x.size, N), xp, y)
        vertices = np.array((x_new, y_new)).T
        new_contours.append(vertices)

    return new_contours


def create_square_with_hole(square_lims=[-100, 100], r=50,
                            N_square=80, N_circ=50,
                            outfile='square_peg'):
    N_square = np.floor(N_square/4).astype(int)
    theta = np.linspace(0, 2*np.pi, N_circ + 1)[:-1]
    x1 = r*np.cos(theta)
    y1 = r*np.sin(theta)

    contours = []
    contours.append(np.array([x1, y1]).T)

    x = np.linspace(*square_lims, N_square + 1)[:-1]
    y = square_lims[1]*np.ones(x.shape)

    top = np.array((x, y)).T
    right = np.array((y, -x)).T
    bottom = np.array((-x, -y)).T
    left = np.array((-y, x)).T
    n1 = np.vstack((top, right, bottom, left))
    contours.append(n1)

    holes = np.atleast_2d(np.array((0,0)))

    write_to_poly(contours, holes, f'{outfile}.poly')


def call_triangle(polyfile, min_angle=0, max_area=None):
    if max_area is None:
        args = f'-pq{min_angle}LD'
    else:
        args = f'-pq{min_angle}LDa{max_area}'
    cmd = ['triangle', args, polyfile]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in p.stdout:
        print(line.decode('utf-8'), end='')
    p.wait()
    print(p.returncode)


create_square_with_hole()
call_triangle('square_peg.poly', min_angle=0, max_area=100)
x, y, simplices = read_from_triangle('square_peg.1')
fig, ax = plt.subplots()
ax.triplot(x, y, simplices)
ax.scatter(x, y)
plt.show()
