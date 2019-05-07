import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('../')
import useful_functions as uf
from scipy import interpolate


def create_phi(Nx=50, Ny=50, xlim=[-10, 10], ylim=[-10, 10]):
    """
    Creates a standardized scalar field for use in the advection equation
    Takes in 4 arguments: x/y limits and number of points in each direction.
    """
    x = np.linspace(xlim[0], xlim[1], Nx)
    y = np.linspace(ylim[0], ylim[1], Ny)
    xx, yy = np.meshgrid(x, y)

    # parameters for the peaks
    sigma_xs = [1] #, 1, 2, 0.8]
    sigma_ys = [1] #, 1, 0.5, 1]
    As = [1] #, 0.7, -0.5, 1]
    mu_xs = [3] #, 3, 0, -3]
    mu_ys = [0] #, 0, -3, 0]

    # Create the field through addition
    phi = np.zeros_like(xx)
    for sigma_x, sigma_y, A, mu_y, mu_x in zip(sigma_xs, sigma_ys,
                                               As, mu_ys, mu_xs):
        phi += (A /(2*np.pi * sigma_x * sigma_y) * 
                np.exp(-(xx-mu_x)**2/(2*sigma_x**2) -
                        (yy-mu_y)**2/(2*sigma_y**2)))

    return xx, yy, phi


def f_u(xx, yy):
    return yy, -xx


def sim_next_step(xx, yy, phi, dt, ux, uy, method='linear', fill=0):
    """
    Generates the next timestep for the advection equation of some scalar field
    with associated velocity field
    """
    x_new = xx - dt * ux
    y_new = yy - dt * uy

    flattened_points = np.vstack((xx.flatten(), yy.flatten())).T
    flattened_values = phi.flatten()
    phi = interpolate.griddata(flattened_points, flattened_values,
                               (x_new, y_new), method=method,
                               fill_value=fill)
    return phi


def run_sim(Nx=50, Ny=50, Nt=6, xlim=[-10, 10], ylim=[-10, 10], 
            f_u=f_u, method='linear', fill=0):
    """
    Runs the actual advection simulation
    """

    # setup:
    xx, yy, phi = create_phi(Nx, Ny, xlim, ylim)
    ux, uy = f_u(xx, yy)
    dt = 2*np.pi/Nt

    # copy starting values. Never know when it might come in handy
    phi_start = phi.copy()

    # Setup to record coordinates of max(phi)
    max_phi_points = np.zeros((2, Nt + 1))
    max_arg = np.argmax(phi)
    point_x = xx.flatten()[max_arg]
    point_y = yy.flatten()[max_arg]
    max_phi_points[:, 0] = [point_x, point_y]

    # run simulation
    for i in range(1, Nt+1):
        phi = sim_next_step(xx, yy, phi, dt, ux, uy,
                            method, fill)
        max_arg = np.argmax(phi)
        point_x = xx.flatten()[max_arg]
        point_y = yy.flatten()[max_arg]
        max_phi_points[:, i] = [point_x, point_y]
    
    return xx, yy, phi, phi_start, max_phi_points


def calc_error(Nxs, Nts, xlim=[-10, 10], ylim=[-10, 10],
               f_u=f_u, method='linear', fill=0):
    """
    Function to run the simulation several time and plot results.
    """
    Nxs2, Nts2 = np.meshgrid(Nxs, Nts)
    error_matrix = np.zeros(Nxs2.shape)

    for i in range(Nxs2.shape[0]):
        for j in range(Nxs2.shape[1]):
            Nx = Nxs2[i, j]
            Nt = Nts2[i, j]

            _, _, phi, phi_start, _ = run_sim(Nx, Nx, Nt, xlim, ylim, f_u, 
                                              method, fill)
            
            residual = uf.calc_residual(phi, phi_start)
            error_matrix[i, j] = residual

    return error_matrix

    


if __name__ == "__main__":
    Nx = np.arange(50, 550, 50)
    Nt = np.arange(10, 60, 10)

    error_matrix = calc_error(Nx, Nt)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(Nx, Nt, error_matrix)
    uf.pretty_plotting(fig, ax,
                       title="RMS of difference between start and end $\phi$",
                       xlabel='Number of points per axis',
                       ylabel='Number of steps per simulation')
    plt.show()

