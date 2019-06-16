#%%
import numpy as np
import matplotlib.pyplot as plt
import sys
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


def ex_ball(dt=0.001, N=1000, frame_skip=1):
    rho = 1800
    nu = 0.48
    E = 0.01e9
    lambda_, mu = proj.calc_lame_parameters(E, nu)
    b = np.array((0, -9.8))*rho
    x, y, simplices, cvs = fvm.load_cvs_mat('ball.mat')
    y = y-3
    t = np.zeros(2)
    boundary_mask = np.ones(x.size) == 1
    t_mask = ~boundary_mask
    points = proj.simulate(x, y, simplices, cvs, dt, N, lambda_, mu, b, t, rho, t_mask, boundary_mask, y0=0)
    proj.make_animation(points, simplices, fps=60, frame_skip=frame_skip, outfile='ball.mp4')


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
    
# ex_ball(N=1001, frame_skip=3)
# ex_simple(dt=0.002, N=5000, frameskip=10)
ex_double_bending()

#%% do stuff
# N = 5000
# dt = 0.002
# frame_skip = 10
# rho = lambda_ = mu = 1
# b = np.zeros(2)
# t = b
# x, y, simplices, cvs = fvm.load_cvs_mat('control_volumes2.mat')
# De0inv, m, f_ext, ft = proj.calc_intial_stuff(x, y, simplices, b, rho, t=t)
# m = m.reshape((m.size, 1))
# p = np.load('bent.npy')

# points_t = np.zeros((N, *p.shape))
# points_t[0] = p

# X, Y = p.T
# Pe = proj.calc_Pe(X, Y, simplices, De0inv, lambda_, mu)
# fe = proj.calc_all_fe(X, Y, simplices, cvs, De0inv, lambda_, mu)

# # print(Pe)
# # print(fe)
# fx, fy = fe.T

# fig, ax = plt.subplots()
# ax.triplot(X, Y, simplices)
# ax.quiver(X, Y, fx, fy)
# plt.show()

#%%
