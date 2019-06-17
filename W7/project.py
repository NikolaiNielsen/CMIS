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
    """
    Calculates the element side lengths

    inputs:
    - x, y: (n,) arrays of nodal positions
    - simplices: (m,3) connectivity matrix
    """
    N = len(simplices)
    De = np.zeros((N, 2, 2))
    for n, simp in enumerate(simplices):
        i, j, k = simp
        De[n] = np.array(((x[j]-x[i], x[k]-x[i]), (y[j]-y[i], y[k]-y[i])))
    return De


def calc_Pe(x, y, simplices, De0Inv, lambda_=1, mu=1, N=False):
    """
    Calculate the 1st Piola-Kirchhoff tensor based on the Lamé-parameters,
    material coordinate De0Inv and current spatial coordinates.
    """
    De = calc_De(x, y, simplices)
    Fe = De @ De0Inv
    Ee = np.zeros(Fe.shape)
    I = np.zeros(Ee.shape)
    Se = np.zeros(Fe.shape)
    for n, _ in enumerate(Ee):
        # Green strain tensor for each element
        Ee[n] = (Fe[n].T @ Fe[n] - np.eye(2))/2
        I[n] = np.eye(2)
    tr = np.trace(Ee, axis1=1, axis2=2)
    if N:
        print(Fe)
    tr2 = np.atleast_3d(tr).reshape((simplices.shape[0],1,1))
    # second and first Piola-Kirchhoff stress tensors
    Se = lambda_ * tr2*I + 2*mu*Ee
    Pe = Fe@Se
    return Pe
    

def find_vertex_order(i, a, b, c):
    """
    cyclically permutes the list [a,b,c], such that i is in the first position.
    """
    if i == a:
        return a, b, c
    elif i == b:
        return b, c, a
    elif i == c:
        return c, a, b
    else:
        raise Exception('i must be equal to a, b or c')


def calc_all_fe(x, y, simplices, cvs, De0Inv, lambda_=1, mu=1, n=False):
    """
    Calculate elastic forces on each vertex

    inputs:
    - x, y: (n,) array of vertex positions
    - simplices: (m,3) connectivity matrix
    - cvs: n-list of control volumes
    - De0Inv: (m,2,2) inverse matrix of element sides
    - lambda_, mu: floats, Lame parameters
    """
    N = x.size
    fe = np.zeros((N,2))
    Pe = calc_Pe(x, y, simplices, De0Inv, lambda_=lambda_, mu=mu, N=n)
    
    for i in range(N):
        # For each vertex we need the neighbouring simplices:
        neighbours = mesh.find_neighbouring_simplices(simplices, i)
        # With the neighbours we need to calculate N and l for these
        for neigh in neighbours:
            simp = simplices[neigh]
            _, j, k = find_vertex_order(i, *simp)
            xi, xj, xk = x[[i, j, k]]
            yi, yj, yk = y[[i, j, k]]
            # lij = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            # lik = np.sqrt((xi-xk)**2 + (yi-yk)**2)
            Nej = -np.array((yi-yj, xj-xi))
            # lej = np.sqrt(np.sum(Nej**2))
            # Nej = Nej/lej
            Nek = np.array((yi-yk, xk-xi))
            # lek = np.sqrt(np.sum(Nek**2))
            # Nek = Nek/lek
            P = Pe[neigh]

            # We don't need to scale Nej and Nek, since their length is already
            # the between the i'th and j/k'th node.
            fi = -0.5*P@Nej - 0.5*P@Nek

            # Debugging:
            # if i == 1:
            #     print(f'i, j, k: {i}, {j}, {k},')
            #     print(f'xi: {xi}, {yi}')
            #     print(f'xj: {xj}, {yj}')
            #     print(f'xk: {xk}, {yk}')
            #     print(f'Nj: {Nej}')
            #     print(f'Nk: {Nek}')
            #     print(f'fi: {fi}')
            
            # Elastic forces on i'th vertex is sum of contribution from each
            # element the i'th vertex is a part of.
            fe[i] += fi
    return fe


def calc_cv_areas(x,y,simplices):
    """
    Calculates the nodal "area" - the area of each control volume, for a median
    dual vertex centred control volume.
    """

    # For this type of control volume, each triangular element is split up into
    # three parts of equal size. The size of the control volume is then the
    # total area of all triangles the vertex is a part of, divided by 3.
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


def calc_next_time_step(x, v, m, f_total, dt, mask):
    """
    Calculates the positions and velocities for the next time step with a semi
    implicit first order Euler integration. Only update masked values (points
    not with a boundary condition)

    inputs:
    - x: (n,2) array of current positions
    - v: (n,2) array of current velocities
    - m: (n,) array of nodal masses
    - f_total: (n, 2) array of total forces acting on nodes
    - dt: float, time step
    - mask: (n,) boolean array. Only masked values are updated

    outputs:
    - x: (n,2) array of new positions, based on new velocities.
    - v: (n,2) array of new velocities.
    """
    v[mask] = v[mask] + dt*(f_total[mask]/m[mask])
    x[mask] = x[mask] + dt*v[mask]
    return x, v


def calc_pot_energy(m, y, y0=0):
    """
    Calculates the potential energy of the system with nodal heights y and
    nodal masses m, given point y0 as reference

    inputs:
    - v: (n,) array of nodal heights
    - m: (n,) array of nodal masses
    - y0: float. reference point for energy

    returns:
    - Epot: float, total potential energy of the system
    """
    g = 9.8
    return np.sum(m*(y-y0)*g)


def calc_kin_energy(m, v):
    """
    Calculates the kinetic energy of the system with nodal velocities v and
    nodal masses m.

    inputs:
    - v: (n,2) array of velocities
    - m: (n,) array of nodal masses

    returns:
    - Ekin: float, total kinetic energy of the system
    """
    lv = np.sum(v*v, axis=1)
    return np.sum(m*lv*0.5)


def calc_momentum(m, v):
    """
    Calculates the total momentum of the system

    inputs:
    - m: (n,) array of nodal masses
    - v: (n,2) array of nodal velocities

    outputs:
    - p: (2,) vector of total momentum
    """

    p = np.atleast_2d(m)*v
    return np.sum(p, axis=0)


def calc_lame_parameters(E, nu):
    """
    Calculate the Lame parameters lambda and mu from Youngs modulus E and the
    poisson ratio nu.

    Inputs:
    - E: Young modulus of the material
    - nu: poisson ratio of the material

    outputs:
    - lambda_: float, first Lame parameter
    - mu: float, second Lame paramter
    """
    mu = E/(2*(1+nu))
    lambda_ = E*nu/((1+nu)*(1-2*nu))
    return lambda_, mu


def simulate(x, y, simplices, cvs, dt=1, N=10, lambda_=1, mu=1, b=np.zeros(2),
             t=np.array((0, -1)), rho=1, t_mask=None, boundary_mask=None,
             y0=None):
    """
    Simulates the system

    Inputs:
    - x, y: (n,) arrays of nodal positions
    - simlices: (m,3) connectivity matrix
    - cvs: n-list of control volumes
    - dt: float, time step
    - N: total steps in the simulation. The initial position is the first step,
         so only N-1 steps are simulated
    - lambda_, mu: Lame parameters
    - b: body force density
    - t: traction
    - rho: mass density of the system
    - t_mask: (n,) boolean array of vertices to apply traction to. if None,
              apply traction to right boundary
    - boundary_mask: (n,) boolean array of nodes to update, ie NOT the clamped
                     boundary. If None, clamp left edge, ie. False on left edge
    
    Outputs:
    - points_t: (N,n,2) array of vertex positions for each step.
    """
    if boundary_mask is None:
        boundary_mask = x != np.amin(x)
    if t_mask is None:
        t_mask = x == np.amax(x)
    n_p = -1
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
        if y0 is not None:
            points_t[n-1], v = floor_(points_t[n-1], y0=y0, v=v)
        x, y = points_t[n-1].T
        fe = calc_all_fe(x, y, simplices, cvs, De0inv, lambda_, mu, n=True if n==n_p else False)
        f_total = ft + fe + f_ext
        points_t[n], v = calc_next_time_step(points_t[n-1], v, m, f_total, dt,
                                             boundary_mask)
        bar.next()
    bar.finish()
    return points_t


def make_animation(points, simplices, dt, lims=None, frame_skip=1, padding=0.5, 
                   fps=12, outfile='video.mp4'):
    """
    Function to make an animation of the mesh.

    inputs:
    - points: (N,n,2) array of point positions. N is number of time steps, n is
              number of points in mesh
    - simplices: (m, 3) connectivity matrix. m is number of triangles in mesh
    - frame_skip: number of frames to skip. So only show the M'th iteration of
                  the simulation
    - padding: padding to add to the sides of the axes object. Not currently
               implemented
    - fps: int, number of frames per second for the video
    - outfile: filename to write the video to. Should include file extension
               ".mp4"
    """
    dpi = 200
    fig, ax = plt.subplots()
    N = points.shape[0]
    Times = np.cumsum(dt*np.ones(N))
    if lims is not None:
        xlims, ylims = lims
    # x_all = points[:,:,0].flatten()
    # y_all = points[:,:,1].flatten()
    # xlims = [np.amin(x_all) - padding, np.amax(x_all) + padding]
    # ylims = [np.amin(y_all) - padding, np.amax(y_all) + padding]
    # ax.set_xlim(*xlims)
    # ax.set_ylim(*ylims)
    # ax.set_aspect('equal')
    # l = ax.triplot(points[0,:,0],points[0,:,1], simplices)
    writer = anim.FFMpegWriter(fps=fps)
    bar = Bar('Writing movie', max=points.shape[0]//frame_skip)
    with writer.saving(fig, outfile, dpi):
        for n in range(0, points.shape[0], frame_skip):
            point = points[n]
            x, y = point.T
            if lims is not None:
                ax.set_xlim(*xlims)
                ax.set_ylim(*ylims)
            ax.set_aspect('equal')
            ax.triplot(x, y, simplices)
            ax.set_title(f'T: {Times[n]:.2f} s')
            writer.grab_frame()
            ax.clear()
            bar.next()
    bar.finish()


def floor_(points, y0, v):
    """ 
    Implement a "floor" for the simulation. If any node is below the floor, we
    put it to the floor and kill the vertical component of the velocity.

    inputs:
    - y: (n,) array of vertical positions
    - y0: float, floor height
    - v: (n,2) array of current velocities
    
    Outputs:
    - y: new vertical positions
    - v: new velocities
    """
    below_floor = points[:,1] < y0
    # if below_floor.any():
    #     print('Some below!')
    points[below_floor,1] = y0
    # only kill vertical component
    v[below_floor, 1] = 0 #-v[below_floor, 1]
    return points, v
