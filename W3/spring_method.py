
#%%
from scipy import interpolate, ndimage
import imageio
from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
from progress.bar import Bar
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


def push_with_splines(x, y, interpolators):
    sdf_i, dx_i, dy_i = interpolators
    d = sdf_i(x, y, grid=False)
    nx = dx_i(x, y, grid=False)
    ny = dy_i(x, y, grid=False)
    nx = d*nx
    ny = d*ny
    mask = d > 0
    x[mask] = x[mask] - nx[mask]
    y[mask] = y[mask] - ny[mask]
    return x,y

def push_points_inside(interp_points, Gx, Gy, sdf):
    # interp_points = np.array((X,Y)).T. Shape = (N,2)
    points = np.array((Gx.flatten(), Gy.flatten())).T
    values = sdf.flatten()

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
    diff = np.array((nx, ny)).T
    interp_points[mask,:] = interp_points[mask] - diff[mask, :]

    return interp_points


def push_fully_inside(X, Y, Gx, Gy, sdf, max_tries=2):

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
            interp_points[mask,:] = push_points_inside(interp_points[mask,:],
                                                       Gx, Gy, sdf)

            d = interpolate.griddata(points, values, interp_points)
            d_nan = np.isnan(d)
            d[d_nan] = interpolate.griddata(points, values,
                                            interp_points[d_nan, :],
                                            method='nearest')

            mask = d > 0
        else:
            return interp_points
    return interp_points


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


def all_triangles(simplices, x, y):
    N_simplices = simplices.shape[0]
    verts = np.zeros((N_simplices, 3, 2))
    for i, simplex in enumerate(simplices):
        verts[i, :, :] = verts_from_simplex(simplex, x, y)
    return verts


def get_outside_triangles(verts, Gx, Gy, sdf, bar=True):
    """
    runs over the list of triangles, and figures out whether they are outside
    or inside the object
    """
    outside = []
    points = np.array((Gx.flatten(), Gy.flatten())).T
    values = sdf.flatten()
    if bar:
        progress = Bar('Ousides', max=verts.shape[0])
    for i, vert in enumerate(verts):
        # First we generate 10 points for each triangle
        interp_points = gen_points_in_triangle(vert)
        # Then we interpolate sdf on these points
        d = interpolate.griddata(points, values, interp_points.T)
        # if points are inside object, then d < 0
        mask = d < 0
        num_inside = np.sum(mask)
        if num_inside < 10:
            outside.append(i)
        if bar:
            progress.next()
    
    if bar:
        progress.finish()
    return outside
        

#%% project particles
Gx, Gy, sdf, X, Y, im = import_data()
x = Gx[0,:]
y = Gy[:,0]
sdf_spline = interpolate.RectBivariateSpline(x, y, sdf, kx=1, ky=1)
dy, dx = np.gradient(sdf)
dx_spline = interpolate.RectBivariateSpline(x, y, dx, kx=1, ky=1)
dy_spline = interpolate.RectBivariateSpline(x, y, dy, kx=1, ky=1)
interpolators = (sdf_spline, dx_spline, dy_spline)

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
ax1.imshow(sdf, cmap='Greys_r')
ax2.imshow(dx, cmap='Greys_r')
ax3.imshow(dy, cmap='Greys_r')

d = sdf_spline(X, Y, grid=False)
nx = dx_spline(X, Y, grid=False)
ny = dy_spline(X, Y, grid=False)
nx = d*nx
ny = d*ny
mask = d > 0

X_new = X.copy()
Y_new = Y.copy()
X_new[mask] = X_new[mask] - nx[mask]
Y_new[mask] = Y_new[mask] - ny[mask]

# fig, ax = plt.subplots()
# ax.imshow(sdf, cmap='Greys_r')
# ax.scatter(X, Y, c='r')
# ax.quiver(X, Y, nx, ny)

# X, Y = push_with_splines(X, Y, interpolators)

# points = np.array((Gx.flatten(), Gy.flatten())).T
# values = sdf.flatten()
# interp_points = np.array((X, Y)).T
# d = interpolate.griddata(points, values, interp_points)

# fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

# ax.scatter(X, Y, d, c='r')
# ax.plot_surface(Gx, Gy, sdf, alpha=0.5)


points = np.array((X,Y)).T
# points = push_points_inside(points, Gx, Gy, sdf)
# T = Delaunay(points)
# triangles = all_triangles(T.simplices, X, Y)
# outside = get_outside_triangles(triangles, Gx, Gy, sdf)
# print(outside)

fig, ax = plt.subplots()
ax.imshow(im, cmap='Greys_r')
# ax.triplot(X, Y, T.simplices)
ax.scatter(X_new, Y_new)

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
