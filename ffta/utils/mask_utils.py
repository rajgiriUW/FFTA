# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 18:43:21 2018

@author: Raj
"""

import numpy as np

def loadmask(path, rows=64):
    """
    Loads mask
    Replaces 1 (Transparent pixels) with NaN
    
    Returns
    -------
    mask : the mask itself
    
    maskNaN : all 1s are NaNs, primarily for image display
    
    mask_on_1D : non-NaN pixels as a 1D list for distance calculation
    mask_off_1D : NaN pixels as a 1D list for CPD masking
    """
    mask = np.loadtxt(path)
    
    if mask.shape[1] < mask.shape[0]: # we know it's fewer rows than columns
        mask = mask.transpose()
    
    mask = mask[:rows, :]
    
    return mask

def load_masks(mask):
    
    mask_nan = np.copy(mask)

    # tuple of coordinates
    zeros = np.where(mask_nan==0)
    nans = np.where(mask_nan==1)
    mask_on_1D = np.array([ (x,y) for x,y in zip(zeros[0], zeros[1])])
    mask_off_1D = np.array([ (x,y) for x,y in zip(nans[0], nans[1])])

    mask_nan[mask_nan == 1] = np.nan
    
    return mask_nan, mask_on_1D, mask_off_1D

def averagemask(CPDarr, mask, rows=128, nan_flag = 1, avg_flag = 0):
    '''
    Returns an averaged CPD trace given the Igor mask
    Mask is assumed to be in image form of [row, col] dimensions
    
    CPDarr is the CPD of n-by-samples, typically 8192 x 128 (8192 pixels, 128 CPD points)
    Rows = 128 is default of 128x64 images
    nan_flag = 1 is what value in the mask to set to NaN. These are IGNORED
        1 is "transparent" in Igor mask
        0 is "opaque/masked off" in Igor mask
    avg_flag = 0 is what values in mask to average.
    
    Returns CPDpixels, the averaged CPD of those pixels
    '''
    mask1D = mask.flatten()
    
    CPDpixels = np.array([])
    
    index = [i for i in np.where(mask1D == avg_flag)[0]]
    for i in index:
        CPDpixels = np.append(CPDpixels, CPDarr[i,:])
        
    CPDpixels = np.reshape(CPDpixels, [len(index), CPDarr.shape[1]])
    CPDpixels = np.mean(CPDpixels, axis=0)

    return CPDpixels
