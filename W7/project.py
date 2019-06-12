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

    Ee = np.zeros(Fe.shape)
    I = np.zeros(Ee.shape)
    for n, _ in enumerate(Ee):
        Ee[n] = (Fe[n].T @ Fe[n] - np.eye(2))/2
        I[n] = np.eye(2)
    tr = np.trace(Ee, axis1=1, axis2=2)
    tr2 = np.atleast_3d(tr).reshape((simplices.shape[0],1,1))
    Se = lambda_ * tr2*I + 2*mu*Ee
    Pe = Fe@Se
    return Pe
    

def calc_fe(x, y, simplices, cv, De0Inv, lambda_=1, mu=1):
    """
    Calculates the elastic forces for a single vertex.
    """
    # We loop over all the parts of the control volume, where code is not 2
    # (ie, we have an edge which is not on the boundary)
    pass

def calc_all_fe(x, y, simplices, cvs, De0Inv, lambda_=1, mu=1):
    N = x.size
    fe = np.zeros((N,2))
    Pe = calc_Pe(x, y, simplices, De0Inv, lambda_=1, mu=1)
    
    for i in range(N):
        # For each vertex we need the neighbouring simplices:
        neighbours = mesh.find_neighbouring_simplices(simplices, i)
        # With the neighbours we need to calculate N and l for these
        for neigh in neighbours:
            simp = simplices[neigh]
            xi = x[i]
            yi = y[i]
            non_i = simp[simp != i]
            xjk = x[non_i]
            yjk = y[non_i]
            l = np.sqrt((xjk - xi)**2 + (yjk - yi)**2)/2
            Ne = np.zeros((2,2))
            Ne[:,0] = xjk-xi
            Ne[:,1] = yjk-yi
            P = Pe[neigh]
            fi = -0.5*P@(Ne[0]*l[0] + Ne[1]*l[1])
            fe[i] += fi
    return fe

    


x, y, simplices, cvs = fvm.load_cvs_mat('control_volumes.mat')
De0inv = calc_De0Inv(x, y, simplices)
fe = calc_all_fe(x, y, simplices, cvs, De0inv)
