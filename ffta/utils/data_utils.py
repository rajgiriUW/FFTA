# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 13:17:12 2018

@author: Raj
"""

from ffta.utils import hdf_utils
from matplotlib import pyplot as plt

def test_pixel(h5_file, param_changes={}, pxls = 1, showplots = True, verbose=True):
    """
    Takes a random pixel and does standard processing.
    
    This is to tune parameters prior to processing an entire image
    
    h5_file : h5Py File, path, Dataset
        H5 file to process
    
    pxls : int, optional
        Number of random pixels to survey
    
    showplots : bool, optional
        Whether to create a new plot or not. 
        
    verbose : bool , optional
        To print to command line. Currently for future-proofing
    
    """
    # get_pixel can work on Datasets or the H5_File
    if any(param_changes):
        hdf_utils.change_params(h5_file, new_vals=param_changes)
        
    parameters = hdf_utils.get_params(h5_file)
    cols = parameters['num_cols']
    rows = parameters['num_rows']
    
    # Creates random pixels to sample
    pixels = []
    if pixels == 1:
    
        pixels.append([0,0])
    
    if pxls > 1:
        
        from numpy.random import randint
        for i in range(pxls):
        
            pixels.append([randint(0,rows), randint(0,cols)])

    # Analyzes all pixels
    for rc in pixels:
        
        h5_px = hdf_utils.get_pixel(h5_file, rc=rc)
        h5_px.analyze()
    
        if showplots == True:
            plt.plot(h5_px.best_fit, 'r--')
            plt.plot(h5_px.cut, 'g-')
        
    return 