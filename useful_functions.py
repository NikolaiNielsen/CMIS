import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import imageio

bwdist = ndimage.morphology.distance_transform_edt


def linspace_with_ghosts(a, b, n):
    """
    Returns a vector x with n linearly spaced points between a and b, along
    with a ghost node at each end, with the same linear spacing
    """
    # the spacing
    dx = (b-a)/(n-1)
    # then we want n+2 points between a-dx and b+dx:
    x = np.arange(a-dx, b+2*dx, dx)
    return x, dx


def plot_without_ghosts(x, y, z, ax, **kwargs):
    """
    Plots a 2D surface on an axis object ax, without plotting the ghost nodes
    """
    ax.plot_surface(x[1:-1, 1:-1], y[1:-1, 1:-1], z[1:-1, 1:-1], **kwargs)
    return None


def pretty_plotting(fig, ax, 
                    f_type='serif',
                    f_size=12,
                    view=None,
                    paper_w=21,
                    paper_h=9*21/16,
                    title=None,
                    xlabel=None,
                    ylabel=None,
                    filename=None):
    # in the boilerplate code we set the axis to have a width/height of
    # 1280/720 pixels, with a 100 pixel border. We work in inches in
    # matplotlib, so we just create the same "aspect ratio"
    border = 10/128
    inches = 1/2.54
    fig.set_size_inches(paper_w*inches, paper_h*inches)
    ax.set_position([border, border, 1-2*border, 1-2*border])
    ax.set_xlabel(xlabel, fontsize=f_size, fontfamily=f_type)
    ax.set_ylabel(ylabel, fontsize=f_size, fontfamily=f_type)
    ax.set_title(title, fontsize=f_size, fontfamily=f_type)
    if view is not None:
        ax.view_init(*view)

    fig.tight_layout()
    
    if filename is not None:
        fig.savefig(filename)


def calc_residual(x1, x2, *args):
    """
    Calculates the residual between two arrays x1 and x2, as an RMS of their
    differences.
    """

    diff = x2-x1
    square = diff ** 2
    mean = np.sum(square)/square.size
    return mean


def bw2phi(I):
    phi = bwdist(np.amax(I)-I) - bwdist(I)
    ind = phi > 0
    phi[ind] = phi[ind] - 0.5
    ind = phi < 0
    phi[ind] = phi[ind] + 0.5
    return phi


def add_ghost_nodes(phi):
    """
    Adds a border of ghost nodes to the array
    """
    N, M = phi.shape
    a = np.zeros((N+2, M+2))
    a[1:-1, 1:-1] = phi
    return a


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
