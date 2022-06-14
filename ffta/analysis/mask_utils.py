# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 18:43:21 2018

@author: Raj
"""

import numpy as np


def load_mask_txt(path, rows=64, flip=False):
    """
    Loads mask from text

    :param path: Filepath to disk location of the mask
    :type path: str

    :param rows: Sets a hard limit on the rows of the mask, due to Igor images being larger (rowwise)
        than the typical G-Mode/FF-trEFM Mask size
    :type rows: int, optional

    :param flip: Whether to flip the mask left-to-right to match CPD
    :type flip: bool, optional

    :returns: the mask itself

    """
    mask = np.loadtxt(path)

    if mask.shape[1] < mask.shape[0]:  # we know it's fewer rows than columns
        mask = mask.transpose()

    if flip:
        mask = np.fliplr(mask)

    mask = mask[:rows, :]

    return mask


def load_masks(mask):
    """
    Outputs several useful mask versions

    For reference:
        NaNs = 1s in the mask = transparent in display
        0s = 0s in the mask = opaque in display (i.e., are exluded/masked)

    :param mask: The mask to process into various new masks
    :type mask: ndarray 2D (rows, columns)

    :returns: tuple (mask_nan, mask_on_1D, mask_off_1D)
        WHERE
        2D ndarray mask_nan - all 1s are NaNs, primarily for image display. (rows, columns)
        1D ndarray mask_on_1D - non-NaN pixels as a 1D list for distance calculation (rows*columns)
        1D ndarray mask_off_1D - NaN pixels as a 1D list for CPD masking (rows*columns)

    """

    mask_nan = np.copy(mask)

    # tuple of coordinates
    zeros = np.where(mask_nan == 0)
    nans = np.where(mask_nan == 1)
    mask_on_1D = np.array([(x, y) for x, y in zip(zeros[0], zeros[1])])
    mask_off_1D = np.array([(x, y) for x, y in zip(nans[0], nans[1])])

    mask_nan[mask_nan == 1] = np.nan

    return mask_nan, mask_on_1D, mask_off_1D


def averagemask(CPDarr, mask, rows=128, nan_flag=1, avg_flag=0):
    '''
    Returns an averaged CPD trace given the Igor mask

    :param CPDarr: the CPD of n-by-samples, typically 8192 x 128 (8192 pixels, 128 CPD points)
        Rows = 128 is default of 128x64 images
    :type CPDarr:

    :param mask: The mask to process into various new masks
    :type mask: ndarray 2D (rows, columns)

    :param rows:
    :type rows: int

    :param nan_flag: what value in the mask to set to NaN. These are IGNORED
        1 is "transparent" in Igor mask
        0 is "opaque/masked off" in Igor mask
    :type nan_flag: int, 0 or 1

    :param avg_flag: what values in mask to average.
    :type avg_flag:

    :returns: the averaged CPD of those pixels
    :rtype:

    '''
    mask1D = mask.flatten()

    CPDpixels = np.array([])

    index = [i for i in np.where(mask1D == avg_flag)[0]]
    for i in index:
        CPDpixels = np.append(CPDpixels, CPDarr[i, :])

    CPDpixels = np.reshape(CPDpixels, [len(index), CPDarr.shape[1]])
    CPDpixels = np.mean(CPDpixels, axis=0)

    return CPDpixels
