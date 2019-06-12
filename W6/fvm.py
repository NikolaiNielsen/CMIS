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


def calc_M(x, y):
    """
    Returns the value of M at the point (x,y). M=(0,-1) inside the unit circle,
    (0,0) outside
    """
    M_in = np.array((0,-1))
    M_out = np.zeros(2)
    in_ = x**2 + y**2 <= 1
    return M_in if in_ else M_out


def matrix_assembly(x, y, simplices, cvs, flip_M=False):
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
            Mx, My = calc_M(mx, my) if not flip_M else -calc_M(mx, my)
            le = np.sqrt((x[n]-x[j])**2 + (y[n]-y[j])**2)

            # calculate if we encounter discontinuity and calc new l
            # accordingly
            o_in = ox**2 + oy**2 <= 1
            d_in = dx**2 + dy**2 <= 1
            if o_in and not d_in:
                l2 = calc_circle_intersection(dx, ox, dy, oy, l)
                l = l2
            elif d_in and not o_in:
                l2 = calc_circle_intersection(ox, dx, oy, dy, l)
                l = l2
                

            me = Mx*nx + My*ny
            if code == 0:
                # We are inside the domain. Straightforward discretization
                # print(n)
                
                b[n] += me*l
                
                A[n, n] += -l/le
                A[n, j] += l/le

            elif code == 1:
                # Edge is coming from inside domain, ends on physical boundary
                # We must apply special discretization (destination node is on
                # convex hull)
                b[n] += me*l
                # le = np.sqrt((ox-dx)**2 + (oy-dy)**2)
                A[n, n] += -l/le
                A[n, j] += l/le
            elif code == 2:
                # Edge is on physical boundary. Must apply boundary condition
                pass
            else:
                raise Exception('Unrecognized code for vertex')
    
    # Let's do the BC with topmost vertex being 0:
    top = y == np.amax(y) if not flip_M else y == np.amin(y)
    middle = np.isclose(x,-0.25,atol=1e-2, rtol=1e-1)
    id_ = np.arange(top.size)[top*middle]
    A[id_] = 0
    A[id_,id_] = 1
    b[id_] = 0

    return A, b


def calc_circle_intersection(x1, x2, y1, y2, le):
    r = 1
    dx = x1-x2
    dy = y1-y2
    dr = le
    D = x1*y2-x2*y1
    sign = 1 if dy>0 else -1
    sign = sign if dy != 0 else 0
    disc = np.sqrt(r**2 * dr**2 - D**2)
    px1 = (D*dy + sign * dx * disc)/dr**2
    px2 = (D*dy - sign * dx * disc)/dr**2
    if dx == 0:
        # print((x1, y1), (x2,y2))
        py1 = (-D*dy + np.abs(dy) * disc)/dr**2
        s1 = (py1-y1)/(y2-y1)
    else:
        s1 = (px1-x1)/(x2-x1)
    if s1 >= 0 and s1 <= 1:
        px = px1
        py = (-D*dy + np.abs(dy) * disc)/dr**2
    else:
        px = px2
        py = (-D*dy - np.abs(dy) * disc)/dr**2

    l = np.sqrt((px-x2)**2 + (py-y2)**2)
    return l
    

def plot_B(x,y,simplices,phi):
    xx, yy, phi, dx, dy = calc_phi_on_grid(x, y, simplices, phi, gradient=True)
    M = np.zeros((*xx.shape,2))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            M[i,j,:] = calc_M(xx[i,j], yy[i,j])
    Bx = M[:,:,0] - dx
    By = M[:, :, 1] - dy
    fig, ax = plt.subplots()
    ax.streamplot(xx, yy, Bx, By)
    ax.set_aspect('equal')
    fig.tight_layout()
    return fig, ax


def write_to_mat(X, Y):
    name = 'points.mat'
    sio.savemat(name, mdict={'X':X, 'Y':Y})


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


def ex_res_x():
    x, y, simplices, cvs = load_cvs_mat('matlab/control_volumes.mat')
    A, b = matrix_assembly(x,y,simplices, cvs)
    phi = np.linalg.solve(A, b)
    xx, yy, phi2 = calc_phi_on_grid(x, y, simplices, phi)
    res = phi2 - phi2[:,::-1]
    total_res = np.sum(res**2)/res.size
    # print(total_res)

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_surface(xx, yy, res, cmap='hsv')
    fig.tight_layout()
    ax.set_title(f'Residual on x-parity flip. Total: {total_res:.3e}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('$\phi(x,y)-\phi(-x,y)$')
    fig.tight_layout()
    fig.savefig('handin/ex_res_x.pdf')


def ex_control_volumes():
    x, y, simplices, cvs = load_cvs_mat('matlab/control_volumes.mat')
    fig, ax = draw_control_volumes(x, y, simplices, cvs, plot_list=[14, 20])
    fig.set_size_inches((4,3.5))
    ax.set_xlim(-2.2, -1)
    ax.set_ylim(-1.15, 0)
    ax.set_title('Sample control volumes')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.tight_layout()
    fig.savefig('handin/control_volumes.pdf')


def ex_res_y():
    x, y, simplices, cvs = load_cvs_mat('matlab/control_volumes.mat')
    A, b = matrix_assembly(x, y, simplices, cvs)
    phi = np.linalg.solve(A, b)

    A2, b2 = matrix_assembly(x, y, simplices, cvs, flip_M=True)
    phi2 = np.linalg.solve(A2, b2)

    print(np.sum(phi.flatten())/phi.size)
    xx, yy, phi = calc_phi_on_grid(x, y, simplices, phi)
    xx, yy, phi2 = calc_phi_on_grid(x, y, simplices, phi2)
    diff = np.sum(phi.flatten())/phi.size - np.sum(phi2.flatten())/phi.size
    res = phi - phi2[::-1,:]-diff
    total_res = np.sum(res.flatten()**2)/res.size
    print(total_res)

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_surface(xx, yy, res, cmap='hsv')
    fig.tight_layout()
    ax.set_title(f'Residual on CP flip. Total: {total_res:.3e}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(r'$\phi(x,y)-\phi(x,-y)$')
    fig.tight_layout()
    fig.savefig('handin/ex_res_y.pdf')
    plt.show()


def ex_simple():
    x, y, simplices, cvs = load_cvs_mat('matlab/control_volumes.mat')
    A, b = matrix_assembly(x, y, simplices, cvs)
    phi = np.linalg.solve(A, b)

    fig1, ax1 = draw_surface(x, y, simplices, phi)
    fig2, ax2 = draw_strealines(x, y, simplices, phi)

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel(r'$\phi$')
    ax1.view_init(30, 35)
    ax1.set_title(r'Solution to $\nabla^2\phi = \nabla\cdot \mathbf{M}$')
    
    fig2.set_size_inches(6,3)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(r'Streamline plot of $\nabla \phi$')

    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig('handin/ex_simple.pdf')
    fig2.savefig('handin/ex_streams.pdf')

# ex_simple()
# ex_res_x()
# ex_res_y()