# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:00:13 2015

@author: Raj
"""

"""
For finding the point scans and tfps of various points in an image.
In Igor/MFP code, the locations are actually as column, row when 
here you want row, column (load the row .ibw, find column pixel)
"""

from ffta import pixel
from ffta.utils import load

import numpy as np
import scipy as sp
from scipy import ndimage
import matplotlib as mpl
from matplotlib import pyplot as pp

import sys
from numpy.lib.npyio import loadtxt
from os.path import splitext

def load_csv(path):
    
    # Get the path and check what the extension is.
    ext = splitext(path)[1]

    if ext.lower() == '.csv':

        signal_array = np.genfromtxt(path, delimiter=',')

    else:

        print "Unrecognized file type!"
        sys.exit(0)    
      
    return signal_array
    
def find_bad_pixels(signal_array, threshold):
    """ Uses Median filter to find 'hot' pixels """          
    
    filtered_array = ndimage.median_filter(signal_array, size=3)
    diff = np.abs(signal_array - filtered_array)
    limit = threshold * np.std(signal_array)    
      
    bad_pixel_list = np.nonzero(diff > limit)
    
    return filtered_array, bad_pixel_list
    
def remove_bad_pixels(signal_array, filtered_array, bad_pixel_list):
    """ Removes bad pixels from the array"""
    
    fixed_array = np.copy(signal_array)    
    
    for y,x in zip(bad_pixel_list[0], bad_pixel_list[1]):
        fixed_array[y, x] = filtered_array[y,x]
    
    return fixed_array
    
def fix_array(path, threshold = 10, israte = False):
    """ Wrapper function to find and remove 'hot' pixels.
        This version uses a path to specify a file    
    """
    if type(path) is str:
        signal_array = load_csv(path)    
    else:
        signal_array = path
    
    if not israte:
        signal_array = 1/signal_array
    
    filtered_array, bad_pixel_list = find_bad_pixels(signal_array, threshold)      
    fixed_array = remove_bad_pixels(signal_array, filtered_array, bad_pixel_list)    
    
    if not israte:
        fixed_array = 1/fixed_array
    
    return fixed_array, bad_pixel_list