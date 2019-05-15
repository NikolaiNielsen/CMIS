#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import sys
sys.path.append('../')
import useful_functions as uf
import quality_measures as qa


#%%
N = 10

xlims, ylims = [0, 3], [0,3]

X = np.random.uniform(*xlims, N)
Y = np.random.uniform(*ylims, N)
points = np.array((X,Y)).T
print(points.shape)
T = Delaunay(points)

fig, ax = plt.subplots()
ax.triplot(X, Y, T.simplices)
ax.scatter(X, Y, marker='o')

#%%