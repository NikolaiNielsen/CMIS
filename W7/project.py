import numpy as np
import subprocess
from matplotlib import pyplot as plt, cm, colors, animation as anim
from matplotlib.patches import Polygon
from scipy import spatial, interpolate, io as sio
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
from progress.bar import Bar
sys.path.append('../useful_functions')
import mesh
import fvm


def calc_De(x, y, simplices):
    triangles = mesh.all_triangles(simplices, x, y)
    N, _, _ = triangles.shape
    De = np.zeros((N, 2, 2))
    for n, tri in enumerate(triangles):
        i = 0
        j = 1
        k = 2
        De[n, :, 0] = tri[j] - tri[i]
        De[n, :, 1] = tri[k] - tri[i]
    return De


def calc_Pe(x, y, simplices, De0Inv, lambda_=1, mu=1):
    """
    Calculate the 1st Piola-Kirchhoff tensor based on the Lamé-parameters,
    material coordinate De0Inv and current spatial coordinates
    """
    De = calc_De(x, y, simplices)
    Fe = De @ De0Inv

    Ee = np.zeros(Fe.shape)
    I = np.zeros(Ee.shape)
    for n, _ in enumerate(Ee):
        Ee[n] = (Fe[n].T @ Fe[n] - np.eye(2))/2
        I[n] = np.eye(2)
    tr = np.trace(Ee, axis1=1, axis2=2)
    tr2 = np.atleast_3d(tr).reshape((simplices.shape[0],1,1))
    Se = lambda_ * tr2*I + 2*mu*Ee
    Pe = Fe@Se
    return Pe
    

def find_vertex_order(i, a, b, c):
    if i == a:
        return a, b, c
    elif i == b:
        return b, c, a
    elif i == c:
        return c, a, b
    else:
        raise Exception('i must be equal to a, b or c')


def calc_all_fe(x, y, simplices, cvs, De0Inv, lambda_=1, mu=1):
    N = x.size
    fe = np.zeros((N,2))
    Pe = calc_Pe(x, y, simplices, De0Inv, lambda_=1, mu=1)
    
    for i in range(N):
        # For each vertex we need the neighbouring simplices:
        neighbours = mesh.find_neighbouring_simplices(simplices, i)
        # With the neighbours we need to calculate N and l for these
        for neigh in neighbours:
            simp = simplices[neigh]
            _, j, k = find_vertex_order(i, *simp)
            xi, xj, xk = x[[i, j, k]]
            yi, yj, yk = y[[i, j, k]]

            Nej = -np.array((yi-yj, xj-xi))
            Nek = np.array((yi-yk, xk-xi))
            P = Pe[neigh]
            fi = -0.5*P@Nej - 0.5*P@Nek
            # if i == 1:
            #     print(f'i, j, k: {i}, {j}, {k},')
            #     print(f'xi: {xi}, {yi}')
            #     print(f'xj: {xj}, {yj}')
            #     print(f'xk: {xk}, {yk}')
            #     print(f'Nj: {Nej}')
            #     print(f'Nk: {Nek}')
            #     print(f'fi: {fi}')
            fe[i] += fi
    return fe


def calc_cv_areas(x,y,simplices):
    m = np.zeros(x.shape)
    triangles = mesh.all_triangles(simplices, x, y)
    areas = mesh.calc_areas(triangles)
    for i in range(x.size):
        neighbours = mesh.find_neighbouring_simplices(simplices, i)
        m[i] = np.sum(areas[neighbours])/3
    return m


def calc_intial_stuff(x, y, simplices, b=np.array((0, 0)),
                      rho=1, mask=None, t=np.array((0, -1))):
    """
    Calculate all that can be precalculated:
    - De0Inv: Inverse of De for all elements
    - m: nodal masses
    - f_ext: nodal body forces
    - ft: nodal traction

    inputs:
    - x, y: (n,) array of positions for mesh vertices
    - simplices: (n,3) connectivity matrix
    - b: body force density
    - rho: material mass density
    - mask: boolean mask on where to apply traction
    - t: traction
    """
    De0 = calc_De(x, y, simplices)
    De0Inv = np.linalg.inv(De0)
    areas = calc_cv_areas(x,y,simplices)
    m = areas * rho
    f_ext = np.outer(areas, b)
    if mask is None:
        mask = x == np.amax(x)
    lengthsums = np.zeros(x.size)
    lengthsums[mask] = calc_edge_lengths(y[mask])
    ft = np.outer(lengthsums, t)
    return De0Inv, m, f_ext, ft


def calc_edge_lengths(x):
    """
    Calculates the sum of integrals of hat functions for a list of points:
    The integral is just l_e, where l_e is the length of the element. But
    inner nodes have contributions from two elements, so we add these twice

    Inputs:
    - x, (n,) array of positions

    Returns:
    - A_n (n,) area per node
    """
    # Get the permutations to sort and unsort x, just incase it isn't sorted
    perm = np.argsort(x)
    inv_perm = np.arange(perm.size)[np.argsort(perm)]

    # sort x
    x = x[perm]
    le = x[1:] - x[:-1]
    A_n = np.zeros(x.shape)
    A_n[1:] += le
    A_n[:-1] += le

    # unsort and return A_n
    A_n = A_n[inv_perm]
    return A_n


def calc_next_time_step(x, v, m, f_ext, ft, fe, dt, mask):
    f_total = f_ext + ft + fe
    v[mask] = v[mask] + dt*(f_total[mask]/m[mask])
    x[mask] = x[mask] + dt*v[mask]
    return x, v


def simulate(x, y, simplices, cvs, dt=1, N=10, lambda_=1, mu=1, b=np.zeros(2),
             t=np.array((0, -1)), rho=1, t_mask=None, boundary_mask=None):
    
    if boundary_mask is None:
        boundary_mask = x != np.amin(x)
    if t_mask is None:
        t_mask = x == np.amax(x)

    points = np.array((x,y)).T
    v = np.zeros(points.shape)
    points_t = np.zeros((N, *points.shape))
    De0inv, m, f_ext, ft = calc_intial_stuff(x, y, simplices, b, rho,
                                             t_mask, t)
    m = m.reshape((m.size, 1))
    points_t[0] = points
    bar = Bar('simulating', max=N)
    bar.next()
    for n in range(1, N):
        x, y = points_t[n-1].T
        fe = calc_all_fe(x, y, simplices, cvs, De0inv, lambda_, mu)
        points_t[n], v = calc_next_time_step(points_t[n-1], v, m, f_ext, ft,
                                             fe, dt, boundary_mask)
        bar.next()
    bar.finish()
    return points_t


def calc_lame_parameters(E, nu):
    """
    Calculate the Lame parameters lambda and mu from Youngs modulus E and the
    poisson ratio nu.
    """
    mu = E/(2*(1+nu))
    lambda_ = E*nu/((1+nu)*(1-2*nu))
    return lambda_, mu


def ex_simple(dt=1, N=10):
    rho = lambda_ = mu = 1
    b = np.zeros(2)
    t = 1e-2 * np.array((0,-1))
    x, y, simplices, cvs = fvm.load_cvs_mat('control_volumes2.mat')
    points = simulate(x, y, simplices, cvs, dt, N, lambda_, mu, b, t, rho)
    fig, axes = plt.subplots()
    # axes = axes.flatten()
    # for n in range(N):
    # x, y = points[-1].T
    # axes.triplot(x, y, simplices)
    # fig.tight_layout()
    # plt.show()
    make_animation(points, simplices, dt)


def make_animation(points, simplices, dt, padding=0.5):
    fps = 1/dt
    fig, ax = plt.subplots()
    x_all = points[:,:,0].flatten()
    y_all = points[:,:,1].flatten()
    xlims = [np.amin(x_all) - padding, np.amax(x_all) + padding]
    ylims = [np.amin(y_all) - padding, np.amax(y_all) + padding]
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.set_aspect('equal')
    # l = ax.triplot(points[0,:,0],points[0,:,1], simplices)
    writer = anim.FFMpegWriter(fps=fps)
    bar = Bar('Writing movie', max=points.shape[0])
    with writer.saving(fig, 'test.mp4',100):
        for n in range(points.shape[0]):
            point = points[n]
            x, y = point.T
            ax.set_xlim(*xlims)
            ax.set_ylim(*ylims)
            ax.set_aspect('equal')
            ax.triplot(x, y, simplices)
            writer.grab_frame()
            ax.clear()
            bar.next()
    bar.finish()

ex_simple(dt=0.1, N=100)
