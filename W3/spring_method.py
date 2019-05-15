
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
    Nx, Ny = sdf.shape
    X = np.random.uniform(0, Nx, Nverts)
    Y = np.random.uniform(0, Ny, Nverts)
    Gx, Gy = np.meshgrid(np.arange(Nx), np.arange(Ny))
    return Gx, Gy, sdf, X, Y, im


#%% project particles
Gx, Gy, sdf, X, Y, im = import_data()
points = np.array((Gx.flatten(), Gy.flatten())).T
values = sdf.flatten()
interp_points = np.array((X,Y)).T

d = interpolate.griddata(points, values, interp_points)
d_nan = np.isnan(d)
d[d_nan] = interpolate.griddata(points, values, interp_points[d_nan,:],
                                method='nearest')

dx, dy = np.gradient(sdf)

nx = interpolate.griddata(points, dx.flatten(), interp_points)
nx_nan = np.isnan(nx)
nx[nx_nan] = interpolate.griddata(points, values, interp_points[nx_nan, :],
                                  method='nearest')

ny = interpolate.griddata(points, dy.flatten(), interp_points)
ny_nan = np.isnan(ny)
ny[ny_nan] = interpolate.griddata(points, values, interp_points[ny_nan, :],
                                  method='nearest')

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.plot_surface(Gx, Gy, sdf, alpha=0.5)
ax.scatter(X, Y, d, c='r')
limits = np.array((0, 255))
extra = np.array((-50, 50))
ax.set_xlim(limits+extra)
ax.set_ylim(limits+extra)
ax.set_title('sdf interpolated')

fig2, ax2 = plt.subplots(subplot_kw=dict(projection='3d'))
ax2.plot_surface(Gx, Gy, dx, alpha=0.5)
ax2.scatter(X, Y, nx, c='r')
limits = np.array((0, 255))
extra = np.array((-50, 50))
ax2.set_xlim(limits+extra)
ax2.set_ylim(limits+extra)
ax2.set_title('dx interpolated')

fig3, ax3 = plt.subplots(subplot_kw=dict(projection='3d'))
ax3.plot_surface(Gx, Gy, dy, alpha=0.5)
ax3.scatter(X, Y, nx, c='r')
limits = np.array((0, 255))
extra = np.array((-50, 50))
ax3.set_xlim(limits+extra)
ax3.set_ylim(limits+extra)
ax3.set_title('dy interpolated')

fig4, ax4 = plt.subplots()
nxf = nx.flatten()
nyf = ny.flatten()
ax4.plot(np.sort(nxf))
ax4.plot(np.sort(nyf))


dx = d*nx
dy = d*ny
mask = d > 0


X[mask] = X[mask] - dx[mask]
Y[mask] = Y[mask] - dy[mask]

fig5, ax5 = plt.subplots()
ax5.imshow(im)
ax5.scatter(X,Y)

plt.show()

#%% 3D diagnostics


#%%
