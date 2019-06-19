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


def ex_simple():
    rho = 10
    E = 1000
    nu = 0.3
    T = 7
    K = 1e-3
    dt = K*np.sqrt(rho/E)
    N = np.ceil(T/dt).astype(int)
    N_frames = 500
    if N < N_frames:
        frame_skip = 1
    else:
        frame_skip = np.floor(N/N_frames).astype(int)
    lambda_, mu = proj.calc_lame_parameters(E, nu)
    b = np.zeros(2)
    t = 1e0 * np.array((0, -1)) * rho
    x, y, simplices, cvs = fvm.load_cvs_mat('control_volumes2.mat')
    # points, E_pot, E_kin, E_str, momentum = proj.simulate(
    #     x, y, simplices, cvs, dt, N, lambda_, mu, b, t, rho, T_stopt=0.5)



    
    # np.savez('bar_stuff', *[points, E_pot, E_kin, E_str])
    points, E_pot, E_kin, E_str = np.load('bar_stuff.npz').values()
    x_all = points[:, :, 0].flatten()
    y_all = points[:, :, 1].flatten()
    xmax = np.amax(x_all)
    xmin = np.amin(x_all)
    ymin = np.amin(y_all)
    ymax = np.amax(y_all)
    limits = np.array(((xmin, xmax), (ymin, ymax)))
    # fig, ax = plt.subplots()
    # x, y = points[-1].T
    # ax.triplot(x, y, simplices)
    # plt.show()
    proj.make_animation(points, simplices, dt, [None, E_kin, E_str],
                        lims=limits, frame_skip=frame_skip, fps=60,
                        outfile='bar.mp4', save=True)


def ex_ball():
    h = 0.3
    g = 1
    n_bounces = 3
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
    x_all = points[:, :, 0].flatten()
    y_all = points[:, :, 1].flatten()
    xmax = np.amax(x_all)
    xmin = np.amin(x_all)
    ymin = np.amin(y_all)
    ymax = np.amax(y_all)
    limits = np.array(((xmin, xmax), (ymin, ymax)))
    proj.make_animation(points, simplices, dt, [E_pot, E_kin, E_str], y0=0,
                        lims=limits, frame_skip=frame_skip, fps=60,
                        outfile='ball.mp4',save=True)
    


def plot_ball():

    x, y, simplices, _ = fvm.load_cvs_mat('ball.mat')
    
    h = 0.5
    g = 1
    n_bounces = 1
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
    proj.make_animation(points, simplices, dt, [E_pot, E_kin, E_str], y0=0,
                        lims=limits, frame_skip=frame_skip, fps=60,
                        outfile='ball_strain.mp4')


def calc_pot_energy_tri(p, p0, k=1):
    r = p-p0
    l = np.sum(r**2)
    return 3/2 * k * l

def ex_debug():
    X = np.array((0, 1, 0.5))
    Y = np.array((0, 0, np.sqrt(3)/2))
    T = np.array((0, 1, 2)).reshape((1, 3))
    I = np.array((X.sum(), Y.sum()))/3
    tris = mesh.all_triangles(T,X,Y)
    area = mesh.calc_areas(tris)
    a = 1.1
    i = a*I
    x = a*X
    y = a*Y

    v = I-i
    x = x+v[0]
    y = y+v[1]
    i = i+v

    X2 = X-I[1]
    Y2 = Y-I[1]
    # lims = np.array(((np.amin(x), np.amax(x)), (np.amin(y), np.amax(y))))

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
    fe = proj.calc_all_fe(x, y, T, cvs, De0inv, lambda_, mu)

    k = 1/2 * (a*a+1)*(lambda_ + mu)

    T0 = 1
    K = 5e-3
    dt = K*np.sqrt(rho/E)
    N = np.ceil(T0/dt).astype(int)
    v = np.zeros((3, 2))
    p = np.array((x, y)).T
    points = np.zeros((N, 3, 2))
    E_kin = np.zeros(N)
    E_pot = np.zeros(N)
    E_str = np.zeros(N)
    E_kin[0] = proj.calc_kin_energy(m, v)
    E_pot[0] = calc_pot_energy_tri(np.array((x[0], y[0])), np.zeros(2), k)
    E_pot[0] = proj.calc_pot_energy(m, y)
    E_str[0] = proj.calc_strain_energy(x, y, T, De0inv, lambda_, mu, area)
    points[0] = p
    fes = np.zeros((N, 3, 2))
    bar = Bar('Simulating', max=N)
    bar.next()
    for n in range(1, N):
        x, y = points[n-1].T
        fe = proj.calc_all_fe(x, y, T, cvs, De0inv, lambda_, mu)
        f_total = fe
        fes[n] = f_total
        points[n], v = proj.calc_next_time_step(
            points[n-1], v, m, f_total, dt, bmask)
        E_kin[n] = proj.calc_kin_energy(m, v)
        x, y = points[n].T
        E_str[n] = proj.calc_strain_energy(x, y, T, De0inv, lambda_, mu, area)
        E_pot[n] = calc_pot_energy_tri(np.array((x[0], y[0])), np.zeros(2), k)
        E_pot[n] = proj.calc_pot_energy(m, y)
        bar.next()
    bar.finish()

    N_frames = 500
    if N < N_frames:
        frame_skip = 1
    else:
        frame_skip = np.floor(N/N_frames).astype(int)
    x_all = points[:, :, 0].flatten()
    y_all = points[:, :, 1].flatten()
    xmax = np.amax(x_all)
    xmin = np.amin(x_all)
    ymin = np.amin(y_all)
    ymax = np.amax(y_all)
    limits = np.array(((xmin, xmax), (ymin, ymax)))
    outfile = 'triangle4.mp4'
    dpi = 200

    fps = 60
    padding = 0.2
    xlims, ylims = limits
    padding = np.array((-padding, padding))
    xlims = xlims + padding
    ylims = ylims + padding
    Times = np.cumsum(dt*np.ones(N))
    fig, (ax, ax2) = plt.subplots(nrows=2,
                                gridspec_kw={'height_ratios': [2, 1]})
    ax2.plot(Times, E_kin+E_pot+E_str, label='$E_{total}$')
    ax2.plot(Times, E_pot, label='$E_{pot}$')
    ax2.plot(Times, E_kin, label='$E_{kin}$')
    ax2.plot(Times, E_str, label='$E_{str}$')
    ax2.legend()
    plt.show()
    # writer = anim.FFMpegWriter(fps=fps)
    # bar = Bar('Writing movie', max=points.shape[0]//frame_skip)
    # ran = list(range(0,points.shape[0], frame_skip))
    # n_list = len(ran)
    # n_save = ran[n_list//2]
    # with writer.saving(fig, outfile, dpi):
    #         for n in range(0, points.shape[0], frame_skip):
    #             point = points[n]
    #             x, y = point.T
    #             ax.set_xlim(*xlims)
    #             ax.set_ylim(*ylims)
    #             ax.set_aspect('equal')
    #             ax.triplot(X, Y, T, linestyle='--')
    #             ax.triplot(x, y, T)
    #             ax.set_title(f'T: {Times[n]:.2f} s')
    #             ax.set_xlabel('x')
    #             ax.set_ylabel('y')
    #             ax2.plot(Times, E_kin+E_str, label='$E_{total}$')
    #             ax2.plot(Times, E_str, label='$E_{str}$')
    #             ax2.plot(Times, E_kin, label='$E_{kin}$')
    #             ax2.axvline(x=Times[n], linestyle='--', color='k')
    #             ax2.legend()
    #             ax2.set_xlabel('T [s]')
    #             ax2.set_ylabel('Energy [J]')
    #             if n==n_save:
    #                 fig.savefig('handin/triangle3.png')
    #             # fig.tight_layout()
    #             writer.grab_frame()
    #             ax.clear()
    #             ax2.clear()
    #             bar.next()
    # bar.finish()


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
ex_simple()
# ex_hex()
# ex_debug()
