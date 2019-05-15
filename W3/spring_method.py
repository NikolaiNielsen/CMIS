
#%%
from scipy import interpolate
import imageio
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('../')
import quality_measures as qa
import useful_functions as uf


bwdist = ndimage.morphology.distance_transform_edt
np.set_printoptions(threshold=np.inf)


def import_data(name='example.bmp', Nverts=500):
    sdf, im = uf.grey_to_sdf(name, ghosts=False)
    Ny, Nx = sdf.shape
    border = 0.5
    X = np.random.uniform(border, Nx-1-border, Nverts)
    Y = np.random.uniform(border, Ny-1-border, Nverts)
    Gx, Gy = np.meshgrid(np.arange(Nx), np.arange(Ny))
    return Gx, Gy, sdf, X, Y, im


def push_points_inside(X, Y, Gx, Gy, sdf):
    points = np.array((Gx.flatten(), Gy.flatten())).T
    values = sdf.flatten()
    interp_points = np.array((X, Y)).T

    d = interpolate.griddata(points, values, interp_points)
    d_nan = np.isnan(d)
    d[d_nan] = interpolate.griddata(points, values, interp_points[d_nan, :],
                                    method='nearest')

    dy, dx = np.gradient(sdf)

    nx = interpolate.griddata(points, dx.flatten(), interp_points)
    nx_nan = np.isnan(nx)
    nx[nx_nan] = interpolate.griddata(points, values, interp_points[nx_nan, :],
                                    method='nearest')

    ny = interpolate.griddata(points, dy.flatten(), interp_points)
    ny_nan = np.isnan(ny)
    ny[ny_nan] = interpolate.griddata(points, values, interp_points[ny_nan, :],
                                    method='nearest')
    nx = d*nx
    ny = d*ny
    mask = d > 0
    X[mask] = X[mask] - nx[mask]
    Y[mask] = Y[mask] - ny[mask]

    return X, Y

#%% project particles
Gx, Gy, sdf, X, Y, im = import_data()

X, Y = push_points_inside(X, Y, Gx, Gy, sdf)
X, Y = push_points_inside(X, Y, Gx, Gy, sdf)


# fig2, ax2 = plt.subplots(subplot_kw=dict(projection='3d'))
# ax2.plot_surface(Gx, Gy, dx, alpha=0.5)
# ax2.scatter(X, Y, nx, c='r')
# limits = np.array((0, 255))
# extra = np.array((-50, 50))
# ax2.set_xlim(limits+extra)
# ax2.set_ylim(limits+extra)
# ax2.set_title('dx interpolated')

# fig3, ax3 = plt.subplots(subplot_kw=dict(projection='3d'))
# ax3.plot_surface(Gx, Gy, dy, alpha=0.5)
# ax3.scatter(X, Y, ny, c='r')
# limits = np.array((0, 255))
# extra = np.array((-50, 50))
# ax3.set_xlim(limits+extra)
# ax3.set_ylim(limits+extra)
# ax3.set_title('dy interpolated')

# fig, ax = plt.subplots()
# ax.imshow(sdf, cmap='Greys_r')
# ax.quiver(X, Y, nx, ny, units='width')
# ax.scatter(X,Y, c='r')

# fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
# ax.plot_surface(Gx, Gy, sdf, alpha=0.5)
# ax.scatter(X, Y, d, c='r')
# limits = np.array((0, 255))
# extra = np.array((-50, 50))
# ax.set_xlim(limits+extra)
# ax.set_ylim(limits+extra)
# ax.set_title('sdf interpolated')



# fig4, ax4 = plt.subplots()
# nxf = nx.flatten()
# nyf = ny.flatten()
# ax4.plot(np.sort(nxf))
# ax4.plot(np.sort(nyf))

fig5, ax5 = plt.subplots()
ax5.imshow(im, cmap='Greys_r')
ax5.scatter(X,Y)

plt.show()
#%% 3D diagnostics


#%%
