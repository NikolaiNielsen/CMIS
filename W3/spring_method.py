
#%%
from scipy import interpolate, ndimage
import imageio
from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
from progress.bar import Bar
import timeit
from functools import partial as part
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


def push_with_splines(x, y, sdf_spline):
    d =  sdf_spline.ev(y, x)
    nx = sdf_spline.ev(y, x, dy=1)
    ny = sdf_spline.ev(y, x, dx=1)
    nx = d*nx
    ny = d*ny
    mask = d > 0
    x[mask] = x[mask] - nx[mask]
    y[mask] = y[mask] - ny[mask]
    return x,y

def push_fully_splines(x, y, sdf_spline, max_tries=3):
    d = sdf_spline.ev(y, x)

    mask = d > 0
    counter_max = max_tries + 1
    for _ in range(counter_max):
        if np.sum(mask):
            x, y = push_with_splines(x, y, sdf_spline)
            d = sdf_spline.ev(y, x)

            mask = d > 0
        else:
            return x, y
    return x, y


def push_points_inside(interp_points, Gx, Gy, sdf):
    # interp_points = np.array((X,Y)).T. Shape = (N,2)
    points = np.array((Gx.flatten(), Gy.flatten())).T
    values = sdf.flatten()

    d = interpolate.griddata(points, values, interp_points)
    dy, dx = np.gradient(sdf)
    nx = interpolate.griddata(points, dx.flatten(), interp_points)
    ny = interpolate.griddata(points, dy.flatten(), interp_points)
    nx = d*nx
    ny = d*ny
    mask = d > 0
    diff = np.array((nx, ny)).T
    interp_points[mask,:] = interp_points[mask] - diff[mask, :]

    return interp_points


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
sdf_spline = interpolate.RectBivariateSpline(y, x, sdf)
push1 = part(push_fully_inside, X, Y, Gx, Gy, sdf)
push2 = part(push_fully_splines, X, Y, sdf_spline)
print(timeit.timeit(push1, number=2))
print(timeit.timeit(push2, number=2))

# points = np.array((X,Y)).T
# points = push_points_inside(points, Gx, Gy, sdf)
# T = Delaunay(points)
# triangles = all_triangles(T.simplices, X, Y)
# outside = get_outside_triangles(triangles, Gx, Gy, sdf)
# print(outside)

# fig, ax = plt.subplots()
# ax.imshow(im, cmap='Greys_r')
# # ax.triplot(X, Y, T.simplices)
# ax.scatter(X, Y)

# plt.show()


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
