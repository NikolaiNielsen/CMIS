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



def read_gray_image(name):
    """
    Read an image as a grayscale image, returns a 2D array of integers between
    0 and 255.
    """
    im = imageio.imread(name, as_gray=True)
    im = im.astype(np.int)
    return im


if __name__ == "__main__":
    im = read_gray_image('example.bmp')
    phi = bwdist(im)
    fig, ax = plt.subplots()
    img_plot = ax.imshow(phi, cmap='Greys_r')
    fig.colorbar(img_plot, orientation='vertical')
    plt.show()