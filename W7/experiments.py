import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../useful_functions')
import mesh
import fvm
import project as proj


def ex_simple(dt=1, N=10, frameskip=1):
    rho = 8100
    E = 200e9
    nu = 0.3
    lambda_, mu = proj.calc_lame_parameters(E, nu)
    b = np.zeros(2)
    t = 1e-2 * np.array((0, -1)) * rho
    x, y, simplices, cvs = fvm.load_cvs_mat('control_volumes2.mat')
    points = proj.simulate(x, y, simplices, cvs, dt, N, lambda_, mu, b, t, rho)
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

# ex_ball(N=1001, frame_skip=3)
ex_simple(dt=0.002, N=5000, frameskip=10)
