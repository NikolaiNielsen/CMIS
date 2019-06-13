import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../useful_functions')
import mesh
import fvm
import project


def create_simple_cvs():
    x = np.array((0, 1, 1, 0)).astype(float)
    y = np.array((0, 0, 1, 1)).astype(float)
    fvm.write_to_mat(x, y)
    fvm.call_matlab(print_=True)


def plot_simple_cvs():
    x, y, T, cvs = fvm.load_cvs_mat('control_volumes.mat')
    fig, ax = fvm.draw_control_volumes(x, y, T, cvs)
    fig.tight_layout()
    plt.show()


def simple(a=2):
    x, y, T, cvs = fvm.load_cvs_mat('control_volumes.mat')
    # print(cvs[0])
    x = x.astype(float)
    y = y.astype(float)
    x2 = a*x
    De0, m, f_ext, ft = project.calc_intial_stuff(x, y, T)
    fe = project.calc_all_fe(x2, y, T, cvs, De0)
    print(fe)


simple()
# plot_simple_cvs()