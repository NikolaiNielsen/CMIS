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
import os
import sys
sys.path.append('../useful_functions')
import mesh


def draw_control_volumes(x, y, simplices, cvs, scale=0.05, plot_list=None):

    fig, ax = plt.subplots()
    ax.triplot(x, y, simplices)

    if plot_list is not None:
        cvs = [cvs[i] for i in plot_list]
    for n, cv in enumerate(cvs):
        ax.plot([cv['ox'], cv['dx']], [cv['oy'], cv['dy']], 'r-',
                linewidth=2)
        ax.plot(cv['ox'], cv['oy'], '*g', linewidth=2)
        ax.plot(cv['mx'], cv['my'], '*b', linewidth=2)
        ax.plot([cv['ox'], scale*cv['ex']+cv['ox']],
                [cv['oy'], scale*cv['ey']+cv['oy']], '-g', linewidth=2)
        ax.plot([cv['mx'], scale*cv['nx']+cv['mx']],
                [cv['my'], scale*cv['ny']+cv['my']], '-g', linewidth=2)
    ax.set_aspect('equal')
    fig.tight_layout()
    return fig, ax


def draw_gradients(x, y, simplices, phi):

    xx, yy, phi, dx, dy = calc_phi_on_grid(x, y, simplices, phi, gradient=True)

    fig, ax = plt.subplots()
    ax.contour(xx, yy, phi, 20, cmap='hsv')
    ax.quiver(xx, yy, dx, dy, cmap='hsv')
    ax.set_aspect('equal')
    fig.tight_layout()

    return fig, ax


def draw_surface(x, y, simplices, phi):
    xx, yy, phi = calc_phi_on_grid(x, y, simplices, phi)

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    # ax.plot_trisurf(x, y, phi, cmap='hsv')
    ax.plot_surface(xx, yy, phi, cmap='hsv')
    max_ = max(np.amax(x), np.amax(y))
    min_ = min(np.amin(x), np.amin(y))
    # ax.set_xlim(max_, min_)
    # ax.set_ylim(max_, min_)
    fig.tight_layout()

    return fig, ax


def draw_strealines(x, y, simplices, phi):
    xx, yy, phi, dx, dy = calc_phi_on_grid(x, y, simplices, phi, gradient=True)
    fig, ax = plt.subplots()
    ax.streamplot(xx, yy, dx, dy)
    ax.set_aspect('equal')
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
    ax.set_aspect('equal')
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
                'ny', 'mx', 'my', 'code', 'sx', 'sy', 'sl']
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


def write_to_mat(X, Y):
    name = 'points.mat'
    sio.savemat(name, mdict={'X': X, 'Y': Y})


def call_matlab(print_=False):
    cwd = os.getcwd()
    matlab = ['useful_functions', 'matlab']
    matlab_path = os.path.join(os.path.dirname(cwd), *matlab)
    cmd_name = f"cd '{matlab_path}';process_data('{cwd}');exit;"
    args = f'-nodisplay -nosplash -nodesktop -r "{cmd_name}"'
    cmd = ['matlab', args]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    if print_:
        for line in p.stdout:
            print(line.decode('utf-8'), end='')
    p.wait()
    if print_:
        print(p.returncode)


def matrix_assembly(x, y, simplices, cvs, f_volume):
    N = len(cvs)
    A = np.zeros((N, N))
    b = np.zeros(N)

    for n, cv in enumerate(cvs):
        E = np.size(cv['I'])
        for e in range(E):
            keys = ['I', 'N', 'ox', 'oy', 'dx', 'dy', 'l', 'ex', 'ey', 'nx',
                    'ny', 'mx', 'my', 'code', 'sx', 'sy', 'sl']
            (_, j, ox, oy, dx, dy, l, ex, ey, nx, ny, mx, my,
            code, sx, sy, sl) = [cv[i][e] for i in keys]
            # e: CV index at start of edge
            # j: CV index at end of edge
            # l: length of edge
            # ox/oy: origin point of edge (x(e), y(e))
            # dx/dy: destination (x(j), y(j))
            # mx/my: midpoint of edge
            # nx/ny: outward normal
            # ex/ey: edge direction normal
            if code == 0:
                # We are inside the domain. Straightforward discretization
                # print(n)
                a_nn, a_nj, b_n = f_volume(x, y, simplices, cv)

                b[n] += b_n

                A[n, n] += a_nn
                A[n, j] += a_nj

            elif code == 1:
                # Edge is coming from inside domain, ends on physical boundary
                # We must apply special discretization (destination node is on
                # convex hull)
                a_nn, a_nj, b_n = f_volume(x, y, simplices, cv)

                b[n] += b_n

                A[n, n] += a_nn
                A[n, j] += a_nj

            elif code == 2:
                a_nn, a_nj, b_n = f_volume(x, y, simplices, cv)

                b[n] += b_n

                A[n, n] += a_nn
                A[n, j] += 0
            else:
                raise Exception('Unrecognized code for vertex')

    return A, b
