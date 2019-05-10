import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import useful_functions as uf
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
import imageio
from progress.bar import Bar
import time

bwdist = ndimage.morphology.distance_transform_edt


def bw2phi(I):
    phi = bwdist(np.amax(I)-I) - bwdist(I)
    ind = phi > 0
    phi[ind] = phi[ind] - 0.5
    ind = phi < 0
    phi[ind] = phi[ind] + 0.5
    return phi


def grey_to_sdf(name, ghosts=True):
    """
    Read an image as a grayscale image, converts to integers, and then returns
    the signed distance field
    """
    im = imageio.imread(name, as_gray=True)
    im = im.astype(np.int)
    im = add_ghost_nodes(im)
    phi = bw2phi(im)
    return phi


def add_ghost_nodes(phi):
    """
    Adds a border of ghost nodes to the array
    """
    N, M = phi.shape
    a = np.zeros((N+2, M+2))
    a[1:-1, 1:-1] = phi
    return a


def update_boundary_conditions(phi):
    """
    Updates the boundary conditions. We use natural boundary conditions where
    Dx(phi_ghost) = Dx(phi_boundary) on vertical boundaries. This means that
    phi_0 = phi_1 + phi_2 - phi_3 
    """
    left = phi[1:4, 1:-1]
    right = phi[-4:-1, 1:-1]
    top = phi[1:-1, 1:4]
    bottom = phi[1:-1, -4:-1]

    phi[0,1:-1] =   left[0]     + left[1]     - left[2]  
    phi[1:-1,0] =   top[:,0]    + top[:,1]    - top[:,2]   
    phi[1:-1,-1] =  bottom[:,0] + bottom[:,1] - bottom[:,2]
    phi[-1,1:-1] =  right[0]    + right[1]    - right[2] 
    # phi[[0,0, -1,-1],[0,-1,0,-1]] = 0

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


def run_sim(phi, dt=1/3, T=1, use_eps=False):
    """
    Function to run the simulation
    """
    dx, dy = 1, 1
    Nt = np.ceil(T/dt).astype(int)
    bar = Bar('Simulating', max=Nt)
    for n in range(Nt):
        phi_old = phi.copy()
        phi_temp = np.zeros_like(phi_old)
        phi = update_boundary_conditions(phi)
        phi_temp[1:-1, 1:-1] = dt * calc_k_on_domain(phi, dx, dy,
                                                     use_eps)
        phi = phi_old + phi_temp
        bar.next()
    bar.finish()
    return phi


def calc_area_contour(contourplot, level_n=0):
    contours = contourplot.collections[level_n].get_paths()
    area = 0
    for cont in contours:
        vertices = cont.vertices
        x, y = vertices.T
        area += np.abs(0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y)))
    return area


def SDF_to_BW(phi):
    im = np.zeros_like(phi)
    im[phi<0] = 255
    return im


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


def plot_results(dt, T, use_eps=True):
    phi = grey_to_sdf('example.bmp')
    phi_plot = phi[1:-1, 1:-1]
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    ax1.imshow(phi_plot, cmap='Greys_r')
    ax1.contour(phi_plot, levels=0)

    phi_end = run_sim(phi, dt, T, clamp_g, use_eps)
    phi_end_plot = phi_end[1:-1, 1:-1]
    ax2.imshow(phi_end_plot, cmap='Greys_r')
    ax2.contour(phi_end_plot, levels=0)

    im = SDF_to_BW(phi_end)
    ax3.imshow(im, cmap='Greys_r')


def calc_error_matrix():
    phi = grey_to_sdf('example.bmp')
    dts = np.linspace(0.01, 0.49, 4)
    T = 500
    
    errors = np.zeros(dts.shape)
    for i, dt in enumerate(dts):
        phi_end = run_sim(phi, dt, T, use_eps=True)
        im = SDF_to_BW(phi_end)
        phi_reconstructed = bw2phi(im)
        errors[i] = uf.calc_residual(phi_end, phi_reconstructed)
        print(f'Done. {i+1}/{dts.size}')
    
    fig, ax = plt.subplots()
    ax.plot(dts, errors)
    plt.show()


if __name__ == "__main__":
    calc_error_matrix()
