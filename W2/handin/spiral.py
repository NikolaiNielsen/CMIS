import numpy as np
import matplotlib.pyplot as plt


def sim_traj(x0=np.array([2, 0]), N=50):
    x = np.zeros((2, N))
    x[:, 0] = x0
    dt = 2*np.pi/N
    M = np.array([[0, 1], [-1, 0]])
    for i in range(1, N):
        x[:, i] = x[:, i-1] - dt * M @ x[:, i-1]
    return x


def create_traj():
    x = sim_traj()
    fig, ax = plt.subplots(figsize=(3,3))
    limits = 3*np.array([-1, 1])
    ax.plot(x[0], x[1])
    ax.set_aspect('equal')
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.scatter(x[0, [0, -1]], x[1, [0, -1]], marker='o')

    # set the x-spine (see below for more info on `set_position`)
    ax.spines['left'].set_position('zero')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()

    # set the y-spine
    ax.spines['bottom'].set_position('zero')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    fig.tight_layout()

    fig.savefig('spiral.pdf')


def analyse_traj():
    Ns = np.arange(5, 230, 1)
    mags = np.zeros(Ns.shape)
    args = np.zeros(Ns.shape)
    diffs = np.zeros((2, Ns.size))
    for i, N in enumerate(Ns):
        x = sim_traj(N=N)
        x_end = x[:,-1]
        mag = np.sqrt(np.sum(x_end**2))
        mags[i] = mag/2
        threshold1 = 1+1e-1
        threshold2 = 0.1
        args[i] = np.arctan2(x_end[1], x_end[0])
        if N==74 or N==207:
            print(N)
            print(args[i])
            print(mags[i])
        x_start = x[:,0]
        diffs[:,i] = x_end-x_start
    print(mags[-1])
    print(args[-1])
    fig, ax = plt.subplots(figsize=(5,3))
    ax.axhline(y=0, color='k', linewidth=1, linestyle='--')
    ax.axhline(y=1, color='k', linewidth=1, linestyle='--')
    ax.plot(Ns, mags, label='magnitude')
    ax.plot(Ns, args, label='argument')
    ax.legend()
    ax.set_title(r'Dependence of rotation on time step. $\Delta t = 2\pi / N$')
    ax.set_xlabel('Number of iterations, $N$')
    fig.tight_layout()
    fig.savefig('rotation_dependence.pdf')

if __name__ == "__main__":
    analyse_traj()
