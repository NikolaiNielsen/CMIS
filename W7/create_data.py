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


def create_ball(r=1, y0=0, N=20, max_area=0.1, min_angle=30, plot=False):
    """
    Creates a ball mesh with radius r and offset y, using a regular N-sided
    polygon.
    
    """
    name = 'ball'
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = r*np.cos(theta)
    y = y0 + r*np.sin(theta)
    points = np.array((x,y)).T
    x, y, T = mesh.generate_and_import_mesh([points],
                                            min_angle=min_angle,
                                            max_area=max_area)
    fvm.write_to_mat(x, y)
    fvm.call_matlab(name)
    if plot:
        x, y, T, cvs = fvm.load_cvs_mat(f'{name}.mat')
        fig, ax = plt.subplots()
        ax.triplot(x,y, T)
        fig.tight_layout()
        plt.show()
