import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('../useful_functions')
import useful_functions as uf


dx = 0.1
xlims = [1, 2]
ylims = [1, 2]
x = np.arange(xlims[0], xlims[1]+dx, dx)
N = x.size
Ne = N-1
Ke = np.zeros((2, 2 * Ne))
fe = np.zeros(2 * Ne)
K = np.zeros((N, N))
f = np.zeros(N)

fe0 = 0
Ke0 = np.array([[1, -1], [-1, 1]])/dx

# Create element matrix
for e in range(Ne):
    range_ = [2 * e, 2 * e + 1]
    Ke[:, range_] = Ke0
    fe[range_] = fe0


# Construct global matrix
for e in range(Ne):
    range_global = [e, e + 1]
    range_local = [2*e, 2*e + 1]
    K[e:e+2, e:e+2] += Ke[:, range_local]
    f[range_global] += fe[range_local]

# Handle boundary conditions:
K[[0, -1],:] = 0
K[[0, -1], [0, -1]] = 1
f[[0, -1]] = ylims


# Inspect matrix eigenvalues
def inspect_matrix(m):
    eigs, _ = np.linalg.eig(m)
    rank = np.linalg.matrix_rank(m)
    fig, ax = plt.subplots()
    ax.scatter(np.arange(eigs.size), eigs)
    ax.set_xlabel('Matrix eigenvalue index')
    ax.set_ylabel('Matrix Eigenvalue')
    ax.set_title(f'Matrix Rank: {rank}, Size: {K.shape}')
    fig.tight_layout()
    return fig, ax

y = np.linalg.solve(K, f)

res = np.sqrt(np.sum(y-x)**2)/(y.size - 2)
fig, ax = plt.subplots()
ax.plot(x, y)
uf.pretty_plotting(fig, ax, title=f'Solution to the Laplace equation in 1D, res = {res:.3e}', f_size=14, filename='handin/experiment1d.pdf')
plt.show()