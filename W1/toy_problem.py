#%% Setup

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import useful_functions as uf

# lap U - k^2 u = f, k>0, f(x,y)
# du/dx = 0 for vert boundaries
# du/dy = 0 for horiz boundaries

# Todo:
# Add code to assemble Matrix A
# Add code to assemble right hand side term b
# Add replace below with u=A\b

# % Convert solution from vector into grid
# for i = 1:
#   6
#   for j = 1:
#     6
#     % Convert into zero-based indexing
#     ii = i-1
#     jj = j-1
#     mid = M*jj + ii
#     % Convert into one-based indexing
#     U(i, j) = u(mid+1)

#   end
# end

def index_helper(i, j, m):
    # Returns the 1D index for a 2D matrix, with j being running index (column
    # number), m being column count and i being row number 
    return m*i+j

# dx, dy = 1/3, 1/3
# y = np.arange(-dy, 1 + 2*dy, dy)
# x = np.arange(-dx, 1 + 2*dx, dx)
N = 4
x, dx = uf.linspace_with_ghosts(0, 1, N)
dy = dx
print(dx)
y = x.copy()
xx, yy = np.meshgrid(x, y)
n, m = xx.shape
u = np.zeros_like(xx)
k = 2
f = xx+yy
u = f.copy()

#%% Performing the matrix assembly

# Initializing the A-matrix
A = np.zeros((f.size, f.size))

# Loop over all elements
for i in range(n):
    for j in range(m):

        node_index = index_helper(i, j, m)
        ghost_top = i == 0
        ghost_bottom = i == n-1
        ghost_left = j == 0
        ghost_right = j == m-1
        ghost = ghost_bottom + ghost_left + ghost_right + ghost_top

        corners = ((ghost_bottom * ghost_left) + (ghost_bottom * ghost_right) + 
                   (ghost_top * ghost_left) + (ghost_top * ghost_right))

        # remember to set f = 0 for ghost nodes, to preserve boundary
        # conditions.
        if ghost:
            f[i,j] = 0
        top_i = np.array([0, 2])
        top_j = np.array([j, j])
        top_indices = index_helper(top_i, top_j, m)
        top_vals = np.array([1, -1])

        bottom_i = np.array([n-1, n-3])
        bottom_j = np.array([j, j])
        bottom_indices = index_helper(bottom_i, bottom_j, m)
        bottom_vals = np.array([1, -1])

        left_i = np.array([i, i])
        left_j = np.array([0, 2])
        left_indices = index_helper(left_i, left_j, m)
        left_vals = np.array([1, -1])
        
        right_i = np.array([i, i])
        right_j = np.array([m-1, m-3])
        right_indices = index_helper(right_i, right_j, m)
        right_vals = np.array([1, -1])

        if not ghost:
            # We are in the domain, now we just need to translate the stencil
            # to matrix form
            i_in_stencil = np.array([i,   i,   i, i+1, i-1])
            j_in_stencil = np.array([j, j+1, j-1,   j,   j])
            stencil_indices = index_helper(i_in_stencil, j_in_stencil, m)
            stencil_values = np.array([-k*k -2/dx/dx - 2/dy/dy,
                                       1/dy/dy,
                                       1/dy/dy,
                                       1/dx/dx,
                                       1/dx/dx])
            A[node_index, stencil_indices] = stencil_values
        elif ghost_right and not ghost_bottom and not ghost_top:
            A[node_index, right_indices] = right_vals
        elif ghost_left and not ghost_bottom and not ghost_top:
            A[node_index, left_indices] = left_vals
        elif ghost_top and not ghost_left and not ghost_right:
            A[node_index, top_indices] = top_vals
        elif ghost_bottom and not ghost_left and not ghost_right:
            A[node_index, bottom_indices] = bottom_vals
        elif corners:
            A[node_index, node_index] = 1

F = f.flatten()
U = np.linalg.solve(A,F)
u = U.reshape((n, m))
fig = plt.figure()
ax = fig.gca(projection='3d')
# ax.plot_surface(xx,yy,u,)
uf.plot_without_ghosts(xx, yy, u, ax, cmap='jet')
uf.pretty_plotting(fig, ax)
plt.show()

