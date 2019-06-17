#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import sys
from scipy.spatial import Delaunay
from progress.bar import Bar
sys.path.append('../useful_functions')
import mesh
import fvm
import project as proj


def ex_steel(dt=1, N=10, frameskip=1):
    rho = 8100
    E = 200e9
    nu = 0.3
    lambda_, mu = proj.calc_lame_parameters(E, nu)
    b = np.zeros(2)
    t = 1e-2 * np.array((0, -1)) * rho
    x, y, simplices, cvs = fvm.load_cvs_mat('control_volumes2.mat')
    points = proj.simulate(x, y, simplices, cvs, dt, N, lambda_, mu, b, t, rho)
    np.save('bent', points[-1])
    t_mask = x == np.amax(x)
    De0, _, _, _ = proj.calc_intial_stuff(x, y, simplices, b, rho, t_mask, t)

    fig, ax = plt.subplots()
    X, Y = points[-1].T
    Pe = proj.calc_Pe(X, Y, simplices, De0, lambda_, mu, True)
    ax.triplot(x, y, simplices)
    ax.triplot(X, Y, simplices)
    fig.tight_layout()
    plt.show()

    # proj.make_animation(points, simplices, dt, frameskip, fps=60)


def ex_simple(dt=1, N=10, frameskip=1):
    rho = lambda_ = mu = 1
    b = np.zeros(2)
    t = 1e-2 * np.array((0, -1)) * rho
    x, y, simplices, cvs = fvm.load_cvs_mat('control_volumes2.mat')
    points = proj.simulate(x, y, simplices, cvs, dt, N, lambda_, mu, b, t, rho)
    np.save('bent', points[-1])
    t_mask = x == np.amax(x)
    De0, _, _, _ = proj.calc_intial_stuff(x, y, simplices, b, rho, t_mask, t)
    
    fig, ax = plt.subplots()
    X, Y = points[-1].T
    Pe = proj.calc_Pe(X, Y, simplices, De0, lambda_, mu, True)
    ax.triplot(x, y, simplices)
    ax.triplot(X, Y, simplices)
    fig.tight_layout()
    plt.show()

    # proj.make_animation(points, simplices, dt, frameskip, fps=60)


def ex_ball():
    rho = 1
    nu = 0.4
    E = 1000
    N_frames = 1500
    T = 1.5
    K = 1e-3
    dt = K*np.sqrt(rho/E)
    N = np.ceil(T/dt).astype(int)
    if N < N_frames:
        frame_skip=1
    else:
        frame_skip = np.floor(N/N_frames).astype(int)
    
    lambda_, mu = proj.calc_lame_parameters(E, nu)
    b = np.array((0, -1))*rho
    x, y, simplices, cvs = fvm.load_cvs_mat('ball.mat')
    y = y-(5-1.02)
    t = np.zeros(2)
    boundary_mask = np.ones(x.size) == 1
    t_mask = ~boundary_mask
    points = proj.simulate(x, y, simplices, cvs, dt, N, lambda_, mu, b, t, rho, t_mask, boundary_mask, y0=0)
    proj.make_animation(points, simplices, dt, fps=60, frame_skip=frame_skip, outfile='ball.mp4')


def ex_debug():
    X = np.array((0, 1, 0.5))
    Y = np.array((0, 0, np.sqrt(3)/2))
    T = np.array((0, 1, 2)).reshape((1, 3))
    I = np.array((X.sum(), Y.sum()))/3

    a = 1.1
    i = a*I
    x = a*X
    y = a*Y

    v = I-i
    x = x+v[0]
    y = y+v[1]
    i = i+v
    lims = np.array(((np.amin(x), np.amax(x)), (np.amin(y), np.amax(y))))

    E, nu = 1e3, 0.3
    rho = 10

    b = np.zeros(2)
    t = b
    mask = np.zeros(3) == 1
    bmask = ~mask
    cvs = None

    De0inv, m, f_ext, ft = proj.calc_intial_stuff(X, Y, T, b, rho, mask, t)
    m = m.reshape((3, 1))
    lambda_, mu = proj.calc_lame_parameters(E, nu)

    K = 5e-3
    dt = K*np.sqrt(rho/E)
    N = np.ceil(1/dt).astype(int)*20
    v = np.zeros((3, 2))
    p = np.array((x, y)).T
    points = np.zeros((N, 3, 2))
    points[0] = p
    fes = np.zeros((N, 3, 2))
    bar = Bar('Simulating', max=N)
    for n in range(1, N):
        x, y = points[n-1].T
        fe = proj.calc_all_fe(x, y, T, cvs, De0inv, lambda_, mu)
        f_total = fe
        fes[n] = f_total
        points[n], v = proj.calc_next_time_step(
            points[n-1], v, m, f_total, dt, bmask)
        bar.next()
    bar.finish()

    outfile = 'triangle.mp4'
    frame_skip = 60
    fps = 60
    dpi = 200
    fig, ax = plt.subplots()
    fig.suptitle(f'dt: {dt}, N: {points.shape[0]}')
    if lims is not None:
        xlims, ylims = lims

    writer = anim.FFMpegWriter(fps=fps)
    bar = Bar('Writing movie', max=points.shape[0]//frame_skip)
    with writer.saving(fig, outfile, dpi):
        for n in range(0, points.shape[0], frame_skip):
            point = points[n]
            x, y = point.T
            fe = fes[n]
            fex, fey = fe.T
            if lims is not None:
                ax.set_xlim(*xlims)
                ax.set_ylim(*ylims)
            ax.set_aspect('equal')
            ax.triplot(X, Y, T)
            ax.quiver(x, y, fex, fey)
            ax.triplot(x, y, T)
            writer.grab_frame()
            ax.clear()
            bar.next()
    bar.finish()


def ex_double_bending():
    N = 5000
    dt = 0.002
    frame_skip = 10
    rho = lambda_ = mu = 1
    b = np.zeros(2)
    t = b
    x, y, simplices, cvs = fvm.load_cvs_mat('control_volumes2.mat')
    De0inv, m, f_ext, ft = proj.calc_intial_stuff(x, y, simplices, b, rho, t=t)
    boundary=x == np.amin(x)
    m = m.reshape((m.size, 1))
    p = np.load('bent.npy')

    points_t = np.zeros((N, *p.shape))
    points_t[0] = p

    X, Y = p.T
    # Pe = proj.calc_Pe(X, Y, simplices, De0inv, lambda_, mu)
    fe = proj.calc_all_fe(X, Y, simplices, cvs, De0inv, lambda_, mu)
    
    # print(Pe)
    # print(fe)
    fx, fy = fe.T
    # fx[boundary] = 0
    # fy[boundary]=0
    ax = fx/m.squeeze()
    ay = fy/m.squeeze()
    axp = ax[~boundary]
    ayp = ay[~boundary]
    Xp = X[~boundary]
    Yp = Y[~boundary]
    # print(ax)
    fig, ax = plt.subplots()
    ax.triplot(X, Y, simplices)
    ax.quiver(Xp, Yp, axp, ayp)
    plt.show()
    # bar = Bar('simulating', max=N)
    # bar.next()
    # boundary_mask = x==np.amin(x)
    # v = np.zeros(p.shape)
    # for n in range(1, N):
    #     x, y = points_t[n-1].T
    #     fe = proj.calc_all_fe(x, y, simplices, cvs, De0inv, lambda_, mu)
    #     f_total = ft + fe + f_ext
    #     points_t[n], v = proj.calc_next_time_step(points_t[n-1], v, m, f_total,
    #                                               dt, boundary_mask)
    #     bar.next()
    # bar.finish()

    # # X2,Y2 = points_t[-1].T
    # proj.make_animation(points_t, simplices, dt, frame_skip, fps=60)
    

def ex_hex():
    x = np.array((0, 1, 3/2, 1/2, 1, 0, -1/2))
    y = np.array((0, 0, np.sqrt(3)/2, np.sqrt(3)/2, np.sqrt(3), np.sqrt(3), np.sqrt(3)/2))
    T = Delaunay(np.array((x,y)).T).simplices
    cvs = None
    fig, ax = plt.subplots()
    ax.triplot(x,y,T)
    ax.set_scale('equal')
    plt.show()

ex_hex()