#%%
import numpy as np
from matplotlib import pyplot as plt, cm, colors, animation as anim
from progress.bar import Bar
import sys
from progress.bar import Bar
from scipy.spatial import Delaunay
sys.path.append('../useful_functions')
import mesh
import fvm
import project as proj

#%%
# Initial mesh
X = np.array((0, 1, 3/2, 1/2, 1, 0, -1/2))-(1/2)
Y = np.array((0, 0, np.sqrt(3)/2, np.sqrt(3)/2,
              np.sqrt(3), np.sqrt(3), np.sqrt(3)/2))
T = Delaunay(np.array((X, Y)).T).simplices

points = np.load('bent_hex.npy')
x, y = points[-1].T
y0 = Y[4] - y[4]
y = y+y0
E, nu = 1e3, 0.4
lambda_, mu = proj.calc_lame_parameters(E, nu)
rho = 10
boundary_mask = x != 10
t = np.zeros(0)
b = np.array((0, -1))*rho

De0inv, m, f_ext, _ = proj.calc_intial_stuff(X, Y, T, b, rho,
                                             boundary_mask, t)
fe = proj.calc_all_fe(x, y, T, None, De0inv, lambda_, mu)
Ft = f_ext+fe
for t, e in zip(f_ext, fe):
    print(t[1], e[1])
fig, ax = plt.subplots()
ax.triplot(x,y,T)
ax.quiver(x, y, Ft[:,0], Ft[:,1], color='r')

#%%
