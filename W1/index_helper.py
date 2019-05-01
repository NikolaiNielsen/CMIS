#%%
import numpy as np
from matplotlib import pyplot as plt

#%%
n, m = 6, 6
a = np.arange(36).reshape((n, m))

# print(a)

ghost_notes = np.zeros((n,m))
for i in range(n):
    for j in range(m):
        ghost = (i == 0) + (i == n-1) + (j == 0) + (j == m-1)
        # print(f'i={i}, j={j}, a[i,j]={a[i,j]}')
        # index = n*i + j
        # print(f'index = {index}')
        ghost_notes[i, j] = ghost

fig, ax = plt.subplots()
ax.spy(ghost_notes)
plt.show()