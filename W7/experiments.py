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
    rho = 10
    E = 1000
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


def ex_ball():
    h = 0.5
    g = 1
    n_bounces = 2
    T = n_bounces*2*np.sqrt(2*h/g)
    rho = 10
    nu = 0.3
    E = 1000
    N_frames = 500

    K = 1e-3
    dt = K*np.sqrt(rho/E)
    N = np.ceil(T/dt).astype(int)
    if N < N_frames:
        frame_skip=1
    else:
        frame_skip = np.floor(N/N_frames).astype(int)
    
    lambda_, mu = proj.calc_lame_parameters(E, nu)
    b = np.array((0, -g))*rho
    x, y, simplices, cvs = fvm.load_cvs_mat('ball.mat')
    y = y-(4-h)
    t = np.zeros(2)
    boundary_mask = np.ones(x.size) == 1
    t_mask = ~boundary_mask
    points, E_pot, E_kin, E_str, momentum = proj.simulate(
        x, y, simplices, cvs, dt, N, lambda_, mu, b, t, rho, t_mask,
        boundary_mask, y0=0)
    E_pot = E_pot*g
    np.savez('ball_stuff', *[points, E_pot, E_kin, E_str])


    # proj.make_animation(points, simplices, dt, fps=60, frame_skip=frame_skip, outfile='ball.mp4')


def plot_ball():

    x, y, simplices, _ = fvm.load_cvs_mat('ball.mat')
    
    h = 0.5
    g = 1
    n_bounces = 2
    T = n_bounces*2*np.sqrt(2*h/g)
    rho = 10
    nu = 0.3
    E = 1000
    N_frames = 500

    K = 1e-3
    dt = K*np.sqrt(rho/E)
    N = np.ceil(T/dt).astype(int)
    if N < N_frames:
        frame_skip = 1
    else:
        frame_skip = np.floor(N/N_frames).astype(int)
    _, m, _, _ = proj.calc_intial_stuff(x, y, simplices, rho)

    points, E_pot, E_kin, E_str = np.load('ball_almost_fail.npz').values()
    #e_p = proj.calc_pot_energy(m.reshape((m.size,1)), y)
    E_pot = E_pot
    x_all = points[:,:,0].flatten()
    y_all = points[:,:,1].flatten()
    xmax = np.amax(x_all)
    xmin = np.amin(x_all)
    ymin = np.amin(y_all)
    ymax = np.amax(y_all)
    limits = np.array(((xmin, xmax),(ymin, ymax)))
    proj.make_animation(points, simplices, dt, [E_pot, E_kin], y0=0,
                        lims=limits, frame_skip=frame_skip, fps=60,
                        outfile='ball1.mp4')

    # x, y = points[-1].T

    # Times = np.cumsum(dt*np.ones(N))
    # fig, (ax1, ax2) = plt.subplots(nrows=2,
    #                                gridspec_kw={'height_ratios': [2, 1]})
    # # ax1.set_xlim(xmin, xmax)
    # # ax1.set_ylim(ymin, ymax)
    # ax1.plot([xmin, xmax], [0,0], '--')
    # ax1.triplot(x, y, simplices)
    # ax1.set_aspect('equal')

    # ax2.plot(Times, E_kin+E_pot+E_str, label='$E_{total}$')
    # ax2.plot(Times, E_pot, label='$E_{pot}$')
    # ax2.plot(Times, E_kin, label='$E_{kin}$')
    # ax2.plot(Times, E_str, label='$E_{str}$')
    
    
    # ax2.axvline(x=Times[10000], linestyle='--', color='k')
    # ax2.legend()
    # fig.tight_layout()
    # plt.show()


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
    h = 0.8
    
    T = 10
    K = 1e-3
    N_frames = 500

    E, nu = 1e3, 0.3
    rho = 10
    lambda_, mu = proj.calc_lame_parameters(E, nu)
    dt = K*np.sqrt(rho/E)
    N = np.ceil(T/dt).astype(int)
    b = np.array((0, -1))*rho
    if N < N_frames:
        frame_skip = 1
    else:
        frame_skip = np.floor(N/N_frames).astype(int)

    x = np.array((0, 1, 3/2, 1/2, 1, 0, -1/2))-(1/2)
    y = np.array((0, 0, np.sqrt(3)/2, np.sqrt(3)/2,
                 np.sqrt(3), np.sqrt(3), np.sqrt(3)/2))+h
    simp = Delaunay(np.array((x,y)).T).simplices
    cvs = None
    y0=0
    # lims = np.array(((np.amin(x), np.amax(x)), (np.amax(y))))

    
    t = np.zeros(2)
    boundary_mask = x != 10
    
    points, E_pot, E_kin, _,_ = proj.simulate(x, y, simp, cvs, dt, N, lambda_,
                                              mu, b, t, rho, boundary_mask,
                                              boundary_mask, y0) 
    x_all = points[:, :, 0].flatten()
    y_all = points[:, :, 1].flatten()
    xmax = np.amax(x_all)
    xmin = np.amin(x_all)
    ymin = np.amin(y_all)
    ymax = np.amax(y_all)
    limits = np.array(((xmin, xmax), (ymin, ymax)))
    proj.make_animation(points, simp, dt, [E_pot, E_kin], y0=y0,
                        lims=limits, frame_skip=frame_skip, fps=60,
                        outfile='hex3.mp4')


# ex_ball()
ex_hex()