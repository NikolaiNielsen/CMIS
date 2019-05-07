import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import useful_functions as uf

a, b, n = 0, 1, 6
x, dx = uf.linspace_with_ghosts(a, b, n)
N = x.size

diag = -2*np.eye(N)
diag_up = np.eye(N, k=1)
diag_down = np.eye(N, k=-1)
A = (diag + diag_down + diag_up)/(dx*dx)
A[0,0] = -1/(2*dx)
A[0,1] = 0
A[0,2] = 1/(2*dx)

A[-1, -1] = 0
A[-1, -2] = 1
A[-1, -3] = 0

f = np.zeros(N)
f[0] = 0
f[-1] = 1

eigs, _ = np.linalg.eig(A)
rank = np.linalg.matrix_rank(A)


u = np.linalg.solve(A, f)
print(u)
fig, ax = plt.subplots()
ax.plot(x[1:-1], u[1:-1])
ax.set_ylim([0, 2])
uf.pretty_plotting(fig, ax, title='1D Toy problem', filename='handin/1d_result.pdf', xlabel='$x$', ylabel='$u$')
plt.show()
