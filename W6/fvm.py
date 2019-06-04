import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
import sys
sys.path.append('../useful_functions')
import mesh

def find_vertex_order(i, a, b, c):
    if i == a:
        return a, b, c
    elif i == b:
        return b, c, a
    elif i == c:
        return c, a, b
    else:
        raise Exception('i must be equal to a, b or c')


def project_to_edge(xi, yi, xj, yj, x, y):
    tmp_dx = xj-xi
    tmp_dy = yj-yi
    tmp_l = np.sqrt((tmp_dx*tmp_dx) + (tmp_dy*tmp_dy))
    tmp_ex = tmp_dx/tmp_l
    tmp_ey = tmp_dy/tmp_l
    tmp_dot = (x-xi)*tmp_ex +(y-yi)*tmp_ey
    px = tmp_dot*tmp_ex + xi
    py = tmp_dot*tmp_ey + yi
    return px, py


def draw_control_volumes(x, y, simplices, CVs, scale):

    fig, ax = plt.subplots()
    ax.triplot(x, y, simplices)
    return fig, ax


def draw_gradients(x, y, phi):
    fig, ax = plt.subplots()
    return fig, ax


def point_location(tri, p):
    """
    Given a triangulation object tri, return the simplex index and
    barycentric coordinates for each point in p
    """ 
    simplex_index = tri.find_simplex(p)
    bc = []
    for id_, point in zip(simplex_index, p):
        # Calculate the two first barycentric coordinates for the relevant
        # simplex
        b = tri.transform[id_, :2].dot(point-tri.transform[id_, 2])
        bc.append(np.c_[np.atleast_2d(b), 1-b.sum()])
    # Create the full array and squeeze the shit out of it
    bc = np.array(bc).squeeze()
    return simplex_index, bc


def calc_side_lengths(triangles):
    """
    Calculates side lengths of all triangles.

    ------------
    Parameters:
    - triangles: (n, 3, 2) array of vertex positions (n triangles, 3 sides, 2 
                           dimensions)
    

    -----------
    Returns:
    - lengths: (n, 3) array of side lengths. Side lengths correspond to
                      opposite vertex (vertex A receives length a = |BC|)
    """
    first_vec = [2, 0, 1]
    second_vec = [1, 2, 0]
    sides = triangles[:, first_vec] - triangles[:, second_vec]
    lengths = np.sqrt(np.sum(sides**2, axis=2))
    return lengths


def calc_incenters(triangles):
    """
    Calculates side lengths of all triangles.

    ------------
    Parameters:
    - triangles: (n, 3, 2) array of vertex positions (n triangles, 3 sides, 2 
                           dimensions)
    

    -----------
    Returns:
    - incenters: (n, 2) array of incenter positions
    """

    # Calculate the side lengths and make the array 3D.
    lengths = calc_side_lengths(triangles)
    lengths3 = np.atleast_3d(lengths)

    # Calculate the weights, make them 2D
    weights = lengths.sum(axis=1)
    weights2 = np.atleast_2d(weights)

    # Calculate the centers, divide by weights to get incenters
    centers = np.sum(triangles * lengths3, axis=1)
    incenters = centers/weights2.T
    return incenters


def calc_for_control(ox, oy, dx, dy):
    # Calculate stuff for create_control_volumes
    l = np.sqrt((dx-ox)**2 + (dy-oy)**2)
    ex = (dx - ox)/l
    ey = (dy - oy)/l
    nx = -ey
    ny = ex
    mx = (dx + ox)/2
    my = (dy + oy)/2
    return l, ex, ey, nx, ny, mx, my

def create_control_volumes(x, y, simplices):
    N = x.size
    triangles = mesh.all_triangles(simplices, x, y)
    incenters = calc_incenters(triangles)
    cx, cy = incenters.T
    hull = spatial.ConvexHull(np.array((x,y)))
    hull_vertices = hull.vertices
    boundary_mask = np.zeros(N)
    boundary_mask[hull_vertices] = 1
    cvs = []

    for i in range(N):
        indices = mesh.find_neighbouring_simplices(simplices, i)
        K = indices.size
        I =    []
        N =    []
        OX =   []
        OY =   []
        DX =   []
        DY =   []
        L  =   []
        EX =   []
        EY =   []
        NX =   []
        NY =   []
        MX =   []
        MY =   []
        code = []

        if boundary_mask[i]:
            a = simplices[indices[0], 0]
            b = simplices[indices[0], 1]
            c = simplices[indices[0], 2]
            ii, jj, kk = find_vertex_order(i, a, b, c)
            ox = x[i]
            oy = y[i]
            dx, dy = project_to_edge(ox, oy, x[jj], y[jj], cx[indices[0]],
                                     cy[indices[0]])
            l, ex, ey, nx, ny, mx, my = calc_for_control(ox, oy, dx, dy)

            I.append(i)
            N.append(-1)
            OX.append(ox)
            OY.append(oy)
            DX.append(dx)
            DY.append(dy)
            L.append(l)
            EX.append(ex)
            EY.append(ey)
            NX.append(-nx)
            NY.append(-ny)
            MX.append(mx)
            MY.append(my)
            code.append(2)

            ox = dx
            oy = dy
            dx = cx[indices[0]]
            dy = cy[indices[0]]
            l, ex, ey, nx, ny, mx, my = calc_for_control(ox, oy, dx, dy)

            I.append(i)
            N.append(jj)
            OX.append(ox)
            OY.append(oy)
            DX.append(dx)
            DY.append(dy)
            L.append(l)
            EX.append(ex)
            EY.append(ey)
            NX.append(-nx)
            NY.append(-ny)
            MX.append(mx)
            MY.append(my)
            code.append(1)

        lastK = K-1 if boundary_mask[i] else K

        for j in range(lastK):
            a = simplices[indices[j], 0]
            b = simplices[indices[j], 1]
            c = simplices[indices[j], 2]
            ii, jj, kk = find_vertex_order(i, a, b, c)

            # Origin vertex index
            o = indices[j]

            # Destination vertex index
            d = indices[j%K + 1]

            ox = cx[o]
            oy = cy[o]
            dx = cx[d]
            dy = cy[d]
            l, ex, ey, nx, ny, mx, my = calc_for_control(ox, oy, dx, dy)
            I.append(i)
            N.append(kk)
            OX.append(ox)
            OY.append(oy)
            DX.append(dx)
            DY.append(dy)
            L.append(l)
            EX.append(ex)
            EY.append(ey)
            NX.append(-nx)
            NY.append(-ny)
            MX.append(mx)
            MY.append(my)
            code.append(0)

        
        if boundary_mask[i]:
            a = simplices[indices[k], 0]
            b = simplices[indices[k], 1]
            c = simplices[indices[k], 2]
            ii, jj, kk = find_vertex_order(i, a, b, c)

            ox = cx[indices[K]]
            oy = cy[indices[K]]
            dx, dy = project_to_edge(x[kk], y[kk], x[i], y[i], ox, oy)
            l, ex, ey, nx, ny, mx, my = calc_for_control(ox, oy, dx, dy)
            I.append(i)
            N.append(kk)
            OX.append(ox)
            OY.append(oy)
            DX.append(dx)
            DY.append(dy)
            L.append(l)
            EX.append(ex)
            EY.append(ey)
            NX.append(-nx)
            NY.append(-ny)
            MX.append(mx)
            MY.append(my)
            code.append(1)

            ox = dx
            ox = dy
            dx = x[i]
            dy = y[i]
            l, ex, ey, nx, ny, mx, my = calc_for_control(ox, oy, dx, dy)
            I.append(i)
            N.append(-1)
            OX.append(ox)
            OY.append(oy)
            DX.append(dx)
            DY.append(dy)
            L.append(l)
            EX.append(ex)
            EY.append(ey)
            NX.append(-nx)
            NY.append(-ny)
            MX.append(mx)
            MY.append(my)
            code.append(2)
        cv = {'I':I, 'N':N, 'ox':OX, 'oy':OY, 'dx':DX, 'dy':DY, 'l':L, 'ex':EX,
              'ey':EY, 'nx':NX, 'ny':NY, 'mx':MX, 'my':MY, 'code':code}
        cvs.append(cv)
    return cvs
        
        

    

points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1], [2,2]])
x, y = points.T
tri = spatial.Delaunay(points)
simplices = tri.simplices
triangles = mesh.all_triangles(simplices, x, y)
incenters = calc_incenters(triangles)
hull = spatial.ConvexHull(points)
fig, ax = plt.subplots()
# mask = np.c_[np.atleast_2d(hull.vertices), hull.vertices[0]].squeeze()
# print(mask)
print(hull.simplices)
ax.triplot(x,y,simplices)
ax.scatter(incenters[:,0], incenters[:,1])
ax.scatter(incenters[3, 0], incenters[3, 1], color='r')
# ax.plot(x[mask],
#         y[mask], color='r')
plt.show()
