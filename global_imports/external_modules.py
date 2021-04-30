#IMPORTS 
from __future__ import division

import os
import sys
sys.path.append('..')

import inspect

import numpy as np
from numpy import linspace
import csv
from os import listdir
from os.path import isfile, join
import time

from multiprocessing import Pool


import math
import numpy.ma as ma

import scipy
from scipy import signal
from scipy.ndimage.filters import gaussian_filter
from scipy import signal
from scipy import misc
from scipy import ndimage
from scipy.optimize import leastsq

import matplotlib
matplotlib.use("TKAgg")
from pylab import *


import skimage

import csv

from scipy.spatial import distance

from scipy.interpolate import UnivariateSpline

from skimage import img_as_float

from scipy.ndimage import convolve1d
from skimage import data
from skimage import filters

import inspect


np.set_printoptions(suppress=True)