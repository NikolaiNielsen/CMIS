import numpy as np
import subprocess
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.patches import Polygon
from scipy import spatial
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
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


def draw_control_volumes(x, y, simplices, cvs, scale=0.05):

    fig, ax = plt.subplots()
    ax.triplot(x, y, simplices)
    i = 0
    n = 9
    for cv in cvs:
        if i == n:
            ax.plot([cv['ox'], cv['dx']], [cv['oy'], cv['dy']], 'r-',
                    linewidth=2)
            # ax.plot(cv['ox'], cv['oy'], '*', linewidth=2)
            # ax.scatter(x[i], y[i])
            print(list(zip(cv['N'],cv['code'])))
            # ax.scatter(x[cv['N']], y[cv['N']])
            mask = cv['N'] == -1
            mask2 = cv['code'] == 1
            # ax.scatter(cv['ox'][mask], cv['oy'][mask])
            ax.scatter(cv['ox'][mask2], cv['oy'][mask2])
            ax.scatter(cv['dx'][mask2], cv['dy'][mask2])
            ax.scatter(x[cv['N'][mask2]], y[cv['N'][mask2]])
            # ax.plot(cv['mx'], cv['my'], '*b', linewidth=2)
            # ax.plot([cv['ox'], scale*cv['ex']+cv['ox']],
            #         [cv['oy'], scale*cv['ey']+cv['oy']], '-g', linewidth=2)
            # ax.plot([cv['mx'], scale*cv['nx']+cv['mx']],
            #         [cv['my'], scale*cv['ny']+cv['my']], '-g', linewidth=2)
        i += 1
    ax.set_aspect('equal')
    fig.tight_layout()
    return fig, ax


def draw_gradients(x, y, simplices, phi):

    xx, yy, phi, dx, dy = calc_phi_on_grid(x, y, simplices, phi, gradient=True)

    fig, ax = plt.subplots()
    ax.contour(xx, yy, phi, 20, cmap='hsv')
    ax.quiver(xx, yy, dx, dy, cmap='hsv')
    fig.tight_layout()
    
    return fig, ax


def draw_surface(x, y, simplices, phi):
    xx, yy, phi = calc_phi_on_grid(x, y, simplices, phi)

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    # ax.plot_trisurf(x, y, phi, cmap='hsv')
    ax.plot_surface(xx, yy, phi)
    fig.tight_layout()

    return fig, ax


def draw_strealines(x, y, simplices, phi):
    xx, yy, phi, dx, dy = calc_phi_on_grid(x, y, simplices, phi, gradient=True)
    fig, ax = plt.subplots()
    ax.streamplot(xx, yy, dx, dy)
    fig.tight_layout()
    return fig, ax


def draw_field(x, y, simplices, phi):
    fig, ax = plt.subplots()
    fig.tight_layout()
    cmap_name = 'Greys_r'
    norm = colors.Normalize(np.amin(phi), np.amax(phi))
    cmap = cm.ScalarMappable(norm, cmap_name)

    triangles, phi2 = calc_phi_on_centroids(x, y, simplices, phi)

    for n, tri in enumerate(triangles):
        poly = Polygon(tri, facecolor=cmap.to_rgba(phi2[n]))
        ax.add_patch(poly)

    return fig, ax


def calc_phi_on_centroids(x, y, simplices, phi):
    triangles = mesh.all_triangles(simplices, x, y)
    phi_tri = phi[simplices]
    phi_tri_avg = np.mean(phi_tri, axis=1)
    return triangles, phi_tri_avg


def calc_phi_on_grid(x, y, simplices, phi, gradient=False):
    x_min = np.amin(x)
    x_max = np.amax(x)
    y_min = np.amin(y)
    y_max = np.amax(y)
    N = 100
    X = np.linspace(x_min, x_max, N)
    Y = np.linspace(y_min, y_max, N)

    xx, yy = np.meshgrid(X, Y)
    # points = np.array((xx.flatten(), yy.flatten()))
    # if points.shape[1] != 2:
    #     points = points.T
    phi2 = interpolate.griddata((x, y), phi, (xx, yy))
    # tri = spatial.Delaunay(np.array((x,y)).T)
    # si, bc = point_location(tri, points)
    # phi_i = phi[simplices[si, 0]]
    # phi_j = phi[simplices[si, 1]]
    # phi_k = phi[simplices[si, 2]]

    # C = phi_i * bc[:, 0] + phi_j * bc[:, 1] + phi_k * bc[:, 2]
    # C = C.reshape([N, N])
    if gradient:
        dx, dy = np.gradient(phi2)
        return xx, yy, phi2, dx, dy
    return xx, yy, phi2


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


def create_control_volumes(x, y, simplices):
    def _calc_for_control(ox, oy, dx, dy):
        # Calculate stuff for create_control_volumes
        l = np.sqrt((dx-ox)**2 + (dy-oy)**2)
        ex = (dx - ox)/l
        ey = (dy - oy)/l
        nx = -ey
        ny = ex
        mx = (dx + ox)/2
        my = (dy + oy)/2
        return l, ex, ey, nx, ny, mx, my
    
    N = x.size
    triangles = mesh.all_triangles(simplices, x, y)
    incenters = calc_incenters(triangles)
    cx, cy = incenters.T
    points = np.array((x,y))
    hull = spatial.ConvexHull(points.T)
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
            print(f'ox: {ox:.2f}, oy: {oy:.2f}')
            print(f'dx: {dx:.2f}, dy: {dy:.2f}')
            l, ex, ey, nx, ny, mx, my = _calc_for_control(ox, oy, dx, dy)

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
            l, ex, ey, nx, ny, mx, my = _calc_for_control(ox, oy, dx, dy)

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
            d = indices[(j+1)%K]
            # print(f'j: {j}, j mod: {(j+1)%K}')
            ox = cx[o]
            oy = cy[o]
            dx = cx[d]
            dy = cy[d]
            l, ex, ey, nx, ny, mx, my = _calc_for_control(ox, oy, dx, dy)
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
            # index -1 corresponds to "K" in matlab code
            a = simplices[indices[-1], 0]
            b = simplices[indices[-1], 1]
            c = simplices[indices[-1], 2]
            ii, jj, kk = find_vertex_order(i, a, b, c)

            ox = cx[indices[-1]]
            oy = cy[indices[-1]]
            dx, dy = project_to_edge(x[kk], y[kk], x[i], y[i], ox, oy)
            l, ex, ey, nx, ny, mx, my = _calc_for_control(ox, oy, dx, dy)
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
            l, ex, ey, nx, ny, mx, my = _calc_for_control(ox, oy, dx, dy)
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
        cv = {'I':np.array(I),
              'N':np.array(N),
              'ox':np.array(OX),
              'oy':np.array(OY),
              'dx':np.array(DX),
              'dy':np.array(DY),
              'l':np.array(L),
              'ex':np.array(EX),
              'ey':np.array(EY),
              'nx':np.array(NX),
              'ny':np.array(NY),
              'mx':np.array(MX),
              'my':np.array(MY),
              'code':np.array(code)}
        cvs.append(cv)
    return cvs


def load_cvs_mat(name):
    """
    Loads a .mat file with contents: vertex positions, triangulation matrix and
    control volume struct
    """
    mat = sio.loadmat(name)
    x = mat['X'].squeeze()
    y = mat['Y'].squeeze()
    # Subtract to account for 1-based indexing in Matlab
    T = mat['T']-1
    cvs_mat = mat['CVs']
    cvs = []
    for c in cvs_mat:
        keys = ['I', 'N', 'ox', 'oy', 'dx', 'dy', 'l', 'ex', 'ey', 'nx',
                'ny', 'mx', 'my', 'code']
        values = c[0][0][0]
        values = [i.squeeze() for i in values]
        d = dict(zip(keys, values))
        # Subtract 1
        d['I'] = d['I'] - 1

        # But only it it's not -1
        not_minus_1 = d['N'] != -1
        d['N'][not_minus_1] = d['N'][not_minus_1] - 1
        cvs.append(d)
    return x, y, T, cvs


def calc_M(x, y):
    """
    Returns the value of M at the point (x,y). M=(0,-1) inside the unit circle,
    (0,0) outside
    """
    M_in = np.array((0,-1))
    M_out = np.zeros(2)
    in_ = x**2 + y**2 <= 1
    return M_in if in_ else M_out


def matrix_assembly(x, y, simplices, cvs):
    N = len(cvs)
    A = np.zeros((N, N))
    b = np.zeros(N)

    for n, cv in enumerate(cvs):
        E = np.size(cv['I'])
        for e in range(E):
            keys = ['I', 'N', 'ox', 'oy', 'dx', 'dy', 'l', 'ex', 'ey', 'nx',
                    'ny', 'mx', 'my', 'code']
            _, j, ox, oy, dx, dy, l, ex, ey, nx, ny, mx, my, code = [
                cv[i][e] for i in keys]
            # e: CV index at start of edge
            # j: CV index at end of edge
            # l: length of edge
            # ox/oy: origin point of edge (x(e), y(e))
            # dx/dy: destination (x(j), y(j))
            # mx/my: midpoint of edge
            # nx/ny: outward normal
            # ex/ey: edge direction normal
            Mx, My = calc_M(mx, my)
            if code == 0:
                # We are inside the domain. Straightforward discretization
                # print(n)
                me = Mx*nx + My*ny
                b[n] += me*l
                # le = np.sqrt((ox-dx)**2 + (oy-dy)**2)
                A[n, n] += -1
                A[n, j] += 1

            elif code == 1:
                # Edge is coming from inside domain, ends on physical boundary
                # We must apply special discretization (destination node is on
                # convex hull)
                me = Mx*nx + My*ny
                b[n] += me*l
                # le = np.sqrt((ox-dx)**2 + (oy-dy)**2)
                A[n, n] += -1
                A[n, j] += 1
            elif code == 2:
                # Edge is on physical boundary. Must apply boundary condition
                pass
            else:
                raise Exception('Unrecognized code for vertex')
    return A, b


def write_to_mat(X, Y):
    name = 'matlab/points.mat'
    sio.savemat(name, mdict={'X':X, 'Y':Y})


def call_matlab(print_=False):
    cmd_name = "run('matlab/process_data.m');exit;"
    args = f'-nodisplay -nosplash -nodesktop -r "{cmd_name}"'
    cmd = ['matlab', args]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    if print_:
        for line in p.stdout:
            print(line.decode('utf-8'), end='')
    p.wait()
    if print_:
        print(p.returncode)


x, y, simplices, cvs = load_cvs_mat('matlab/control_volumes.mat')
A, b = matrix_assembly(x,y,simplices, cvs)
phi = np.linalg.solve(A, b)
fig, ax = draw_field(x, y, simplices, phi)

fig, ax = draw_strealines(x, y, simplices, phi)
fig, ax = draw_surface(x, y, simplices, phi)
fig, ax = draw_gradients(x, y, simplices, phi)
plt.show()

