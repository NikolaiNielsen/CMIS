
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

#%% Import the data
sdf, im = uf.grey_to_sdf('example.bmp', ghosts=False)
fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(sdf, cmap='Greys_r')
ax2.imshow(im, cmap='Greys_r')
plt.show()

#%% create particles
Ny, Nx = sdf.shape
Nverts = 500
X = np.random.uniform(0, Nx, Nverts)
Y = np.random.uniform(0, Ny, Nverts)

fig, ax = plt.subplots()
ax.scatter(X,Y, marker='+')
plt.show()

#%% project particles
GX, GY = np.meshgrid(np.arange(Nx), np.arange(Ny))
points = np.array((GX.flatten(), GY.flatten())).T
values = sdf.flatten()
interp_points = np.array((X,Y)).T
d = interpolate.griddata(points, values, interp_points, fill_value=0)
# dx, dy = np.gradient(sdf)
# nx = interpolate.griddata(points, dx.flatten(), interp_points, fill_value=0)
# ny = interpolate.griddata(points, dy.flatten(), interp_points, fill_value=0)

# dx = d*nx
# dy = d*ny

# X[d > 0] = X[d > 0] - dx[d > 0]
# Y[d > 0] = Y[d > 0] - dy[d > 0]

fig, ax = plt.subplots()
ax.imshow(im, cmap='Greys_r')
ax.scatter(X, Y, c='r', marker='.')
plt.show()
#%% 3D diagnostics

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.plot_surface(GX, GY, sdf)
ax.scatter(X, Y, d)

#%%
