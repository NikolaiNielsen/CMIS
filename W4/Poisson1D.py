import numpy as np
import sys
import matplotlib.pyplot as plt
sys.append('../')
import useful_functions as uf


dx = 0.1
xlims = [1, 2]
ylims = [1, 2]
x = np.arange(xlims[0], xlims[1]+dx, dx)
N = x.size
Ke = np.zeros((2, 2*N-2))
K = np.zeros((N, N))
f = np.zeros(N)

