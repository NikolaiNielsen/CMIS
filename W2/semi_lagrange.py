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
    sigma_xs = [0.5, 1, 2, 0.8]
    sigma_ys = [0.5, 1, 0.5, 1]
    As = [-0.2, 0.7, -0.5, 1]
    mu_xs = [0, 3, 0, -3]
    mu_ys = [3, 0, -3, 0]

    # Create the field through addition
    phi = np.zeros_like(xx)
    for sigma_x, sigma_y, A, mu_y, mu_x in zip(sigma_xs, sigma_ys,
                                               As, mu_ys, mu_xs):
        phi += A * np.exp(-(xx-mu_x)**2/(2*sigma_x**2) -
                          (yy-mu_y)**2/(2*sigma_y**2))

    return xx, yy, phi


def f_u(xx, yy):
    return yy, -xx


def sim_next_step(xx, yy, phi, dt, ux, uy, f_u, method='linear', fill=0):
    """
    Generates the next timestep for the advection equation of some scalar field
    with associated velocity field
    """
    x_new = xx - dt * ux
    y_new = yy - dt * uy
    ux, uy = f_u(x_new, y_new)

    flattened_points = np.vstack((xx.flatten(), yy.flatten())).T
    flattened_values = phi.flatten()
    phi = interpolate.griddata(flattened_points, flattened_values,
                               (x_new, y_new), method=method,
                               fill_value=fill)
    return x_new, y_new, phi, ux, uy


xx, yy, phi = create_phi()
uu, uy = f_u(xx, yy)

fig, axes = plt.subplots(subplot_kw=dict(projection='3d'))
axes.plot_surface(xx, yy, phi, cmap='jet')
axes.set_title('0')
axes.view_init(90, 0)
# for i in range(1,6):
#     xx = xx - dt * (-yy)
#     yy = yy - dt * (xx)

#     flattened_points = np.vstack((xx.flatten(), yy.flatten())).T
#     flattened_values = zz.flatten()
#     zz = interpolate.griddata(flattened_points, flattened_values,
#                               (xx, yy), method='linear', fill_value=0)
#     axes[i].plot_surface(xx, yy, zz, cmap='jet')
#     axes[i].set_title(str(i))
#     axes[i].view_init(90, 0)

fig.tight_layout()
plt.show()
