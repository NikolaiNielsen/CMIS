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


x, y, simplices, cvs = fvm.load_cvs_mat('control_volumes.mat')

