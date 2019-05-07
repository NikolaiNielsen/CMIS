import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('../')
import useful_functions as uf
from scipy import interpolate


xmin, xmax = -10, 10
ymin, ymax = -10, 10
Nx = 100
Ny = 100
dt = 1

num_steps = 2*np.pi / dt

N_peaks = 5
sigma_xs = np.random.uniform(0.5, 2, N_peaks)
sigma_ys = np.random.uniform(0.5, 2, N_peaks)
As = np.random.uniform(-2, 4, N_peaks)
mu_xs = np.random.uniform(-5, 5, N_peaks)
mu_ys = np.random.uniform(-5, 5, N_peaks)

x = np.linspace(xmin, xmax, Nx)
y = np.linspace(ymin, ymax, Ny)
xx, yy = np.meshgrid(x, y)

zz = np.zeros_like(xx)
for sigma_x, sigma_y, A, mu_y, mu_x in zip(sigma_xs, sigma_ys,
                                           As, mu_ys, mu_xs):
    zz += A/(2*np.pi*sigma_y*sigma_x) * np.exp(-(xx-mu_x)**2/(2*sigma_x**2) - (yy-mu_y)**2/(2*sigma_y**2)) 

zmin = np.amin(zz)
zmax = np.amax(zz)

fig, axes = plt.subplots(ncols=3, nrows=2, subplot_kw=dict(projection='3d'))
axes = axes.flatten()
axes[0].plot_surface(xx, yy, zz, cmap='jet')
axes[0].set_title('0')
axes[0].view_init(90, 0)
for i in range(1,6):
    xx = xx - dt * (-yy)
    yy = yy - dt * (xx)

    flattened_points = np.vstack((xx.flatten(), yy.flatten())).T
    flattened_values = zz.flatten()
    zz = interpolate.griddata(flattened_points, flattened_values,
                              (xx, yy), method='linear', fill_value=0)
    axes[i].plot_surface(xx, yy, zz, cmap='jet')
    axes[i].set_title(str(i))
    axes[i].view_init(90, 0)

fig.tight_layout()
plt.show()
