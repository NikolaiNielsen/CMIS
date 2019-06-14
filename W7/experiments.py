import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../useful_functions')
import mesh
import fvm
import project as proj


def ex_simple(dt=1, N=10):
    rho = lambda_ = mu = 1
    b = np.zeros(2)
    t = 1e-2 * np.array((0, -1))
    x, y, simplices, cvs = fvm.load_cvs_mat('control_volumes2.mat')
    points = proj.simulate(x, y, simplices, cvs, dt, N, lambda_, mu, b, t, rho)
    fig, axes = plt.subplots()
    # axes = axes.flatten()
    # for n in range(N):
    # x, y = points[-1].T
    # axes.triplot(x, y, simplices)
    # fig.tight_layout()
    # plt.show()
    proj.make_animation(points, simplices, dt)
