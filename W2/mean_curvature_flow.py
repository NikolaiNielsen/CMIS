import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import useful_functions as uf
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
import imageio


bwdist = ndimage.morphology.distance_transform_edt

def bw2phi(I):
    phi = bwdist(np.amax(I)-I) - bwdist(I)
    ind = phi > 0
    phi[ind] = phi[ind] - 0.5
    ind = phi < 0
    phi[ind] = phi[ind] + 0.5
    return phi



def grey_to_sdf(name):
    """
    Read an image as a grayscale image, converts to integers, and then returns
    the signed distance field
    """
    im = imageio.imread(name, as_gray=True)
    im = im.astype(np.int)
    phi = bw2phi(im)
    return phi


def calc_k_on_domain(phi, deltax, deltay, use_eps=False):

    # Calculate the shifted arrays for the derivatives. Will also work with
    # ghosts
    phi_i_j = phi[1:-1, 1:-1]
    phi_pi_pj = phi[2:, 2:]
    phi_pi_j = phi[2:, 1:-1]
    phi_pi_mj = phi[2:, 0:-2]
    phi_i_pj = phi[1:-1, 2:]
    phi_i_mj = phi[1:-1, 0:-2]
    phi_mi_pj = phi[0:-2, 2:]
    phi_mi_j = phi[0:-2, 1:-1]
    phi_mi_mj = phi[0:-2, 0:-2]

    # calculate derivatives
    Dx = (phi_pi_j - phi_mi_j)/(2*deltax)
    Dy = (phi_i_pj - phi_i_mj)/(2*deltay)
    Dxx = (phi_pi_j - 2*phi_i_j + phi_mi_j)/(deltax**2)
    Dyy = (phi_i_pj - 2*phi_i_j + phi_i_mj)/(deltay**2)
    Dxy = (phi_pi_pj - phi_pi_mj - phi_mi_pj + phi_mi_mj)/(4*deltax*deltay)
    g = np.sqrt(Dx**2 + Dy**2)

    # Clamp g or add eps to k?
    if not use_eps and g < 0.5:
        g = 1
    k = (Dx*Dx*Dyy + Dy*Dy*Dxx - 2*Dxy*Dx*Dy) / \
        (g**3 + use_eps*np.finfo(float).eps)

    # Code for clamping k: k = max(-kappa, min(k, kappa))
    kappa = 1/max(deltax, deltay)
    k = np.maximum(-kappa, np.minimum(k, kappa))
    return k


def run_sim(phi, Nt, T=1, clamp_g=False, use_eps=False, animate=None):
    """
    Function to run the simulation
    """
    N, M = phi.shape
    dx, dy = 1, 1
    dt = T/Nt

    if animate is not None:
        fig, ax = animate
        ax.imshow(phi)
        fig.show()

    for n in range(Nt):
        phi_old = phi.copy()
        phi_temp = np.zeros_like(phi_old)
        phi_temp[1:-1, 1:-1] = dt * calc_k_on_domain(phi, dx, dy,
                                                     use_eps)
        phi = phi_old + phi_temp
        # fig.clear()
        if animate is not None:
            ax.imshow(phi)
            plt.draw()
        print(f'n: {n+1}/{Nt}')

    
    return phi
    

def plot_first():
    phi = grey_to_sdf('example.bmp')
    print(np.amin(phi), np.amax(phi))
    fig, ax = plt.subplots()
    ax.imshow(phi, cmap='Greys_r')
    plt.show()


def plot_series(ncols=5, nrows=2, T=1):
    nt = ncols*nrows
    phi = grey_to_sdf('example.bmp')
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows)
    ax = ax.flatten()
    ax[0].imshow(phi, cmap='Greys_r')
    for i in range(1, nt):
        phi = run_sim(phi, 1, T/(nt-1))
        img_plot = ax[i].imshow(phi, cmap='Greys_r')
        fig.colorbar(img_plot, orientation='vertical')
    plt.show()


def plot_results(Nt, T, clamp_g=False, use_eps=True):
    phi = grey_to_sdf('example.bmp')
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.imshow(phi, cmap='Greys_r')
    ax1.contour(phi, levels=0)

    phi_end = run_sim(phi, Nt, T, clamp_g, use_eps)
    ax2.imshow(phi_end, cmap='Greys_r')
    ax2.contour(phi_end, levels=0)

    plt.show()


if __name__ == "__main__":
    plot_results(1000, 20)