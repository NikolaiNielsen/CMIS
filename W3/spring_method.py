
#%%
from scipy import interpolate, ndimage
import imageio
from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('../')
import quality_measures as qa
import useful_functions as uf


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


def push_fully_inside(X, Y, Gx, Gy, sdf, max_tries=3):

    points = np.array((Gx.flatten(), Gy.flatten())).T
    values = sdf.flatten()
    interp_points = np.array((X, Y)).T
    d = interpolate.griddata(points, values, interp_points)
    d_nan = np.isnan(d)
    d[d_nan] = interpolate.griddata(points, values,
                                    interp_points[d_nan, :],
                                    method='nearest')
    mask = d > 0
    counter_max = max_tries+1

    for _ in range(counter_max):
        if np.sum(mask):
            X[mask], Y[mask] = push_points_inside(X[mask], Y[mask], Gx, Gy, sdf)

            interp_points = np.array((X, Y)).T
            d = interpolate.griddata(points, values, interp_points)
            d_nan = np.isnan(d)
            d[d_nan] = interpolate.griddata(points, values,
                                            interp_points[d_nan, :],
                                            method='nearest')

            mask = d > 0
        else:
            return X, Y
    return X, Y


def gen_points_in_triangle(v, N=10):
    r1 = np.sqrt(np.random.uniform(size=N))
    r2 = np.random.uniform(size=N)
    a = 1-r1
    b = r1*(1-r2)
    c = r1*r2
    r = np.array((a,b,c))
    points = v.T @ r
    return points


def verts_from_simplex(simplex, x, y):
    verts = np.array((x[simplex], y[simplex])).T
    return verts
#%% project particles
Gx, Gy, sdf, X, Y, im = import_data()

X, Y = push_fully_inside(X, Y, Gx, Gy, sdf)

points = np.array((X, Y)).T
T = Delaunay(points)


fig, ax = plt.subplots()
ax.imshow(im, cmap='Greys_r')
ax.triplot(X, Y, T.simplices)
ax.scatter(X,Y)

plt.show()


#%%
# a = np.array(((1,1),(2,4),(5,2)))
# points = gen_points_in_triangle(a, N=10)
# x, y = points
# fig, ax = plt.subplots()
# # ax.plot(a[[0,1],0], a[[0,1],1], 'b')
# # ax.plot(a[[0,2],0], a[[0,2],1], 'b')
# # ax.plot(a[[1,2],0], a[[1,2],1], 'b')
# # ax.scatter(a[:,0], a[:,1], s=36)
# ax.scatter(x,y, s=1)
# plt.show()
