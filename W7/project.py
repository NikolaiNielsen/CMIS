import numpy as np
import subprocess
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.patches import Polygon
from scipy import spatial
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import os
import sys
sys.path.append('../useful_functions')
import mesh
import fvm


def calc_De0Inv(x0, y0, simplices):
    
    # Calculates the inverse of De0 for all elements.
    triangles = mesh.all_triangles(simplices, x0, y0)
    N, _, _ = triangles.shape
    De0 = np.zeros((N, 2, 2))
    for n, tri in enumerate(triangles):
        i = 0
        j = 1
        k = 2
        De0[n, :,0] = tri[j] - tri[i]
        De0[n, :,1] = tri[k] - tri[i]
    
    De0Inv = np.linalg.inv(De0)

    return De0Inv


def calc_De(x, y, simplices):
    triangles = mesh.all_triangles(simplices, x, y)
    N, _, _ = triangles.shape
    De = np.zeros((N, 2, 2))
    for n, tri in enumerate(triangles):
        i = 0
        j = 1
        k = 2
        De[n, :, 0] = tri[j] - tri[i]
        De[n, :, 1] = tri[k] - tri[i]
    return De


def calc_Pe(x, y, simplices, De0Inv, lambda_=1, mu=1):
    """
    Calculate the 1st Piola-Kirchhoff tensor based on the Lamé-parameters,
    material coordinate De0Inv and current spatial coordinates
    """
    De = calc_De(x, y, simplices)
    Fe = De @ De0Inv

    Ee = (Fe.T @ Fe - np.eye(2))/2
    tr = np.trace(Ee, axis1=1, axis2=2)
    I = np.zeros(Ee.shape)
    for n, i in enumerate(I):
        I[n] = np.eye(2)
    tr2 = np.atleast_3d(tr).reshape((simplices.shape[0],1,1))
    Se = lambda_ * tr2*I + 2*mu*Ee
    Pe = Fe@Se
    return Pe
    

    

x, y, simplices, cvs = fvm.load_cvs_mat('control_volumes.mat')
De0inv = calc_De0Inv(x, y, simplices)
calc_Pe(x, y, simplices, De0inv)
