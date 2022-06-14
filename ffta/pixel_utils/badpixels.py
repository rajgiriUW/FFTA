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

import sys
from os.path import splitext

import numpy as np
from scipy import ndimage


def load_csv(path):
    """
    :param path:
    :type path:

    :returns:
    :rtype:
    """
    # Get the path and check what the extension is.
    ext = splitext(path)[1]

    if ext.lower() == '.csv':

        signal_array = np.genfromtxt(path, delimiter=',')

    else:

        print("Unrecognized file type!")
        sys.exit(0)

    return signal_array


def find_bad_pixels(signal_array, threshold=2, iterations=1):
    """
    Uses Median filter to find 'hot' pixels

    :param signal_array:
    :type signal_array:

    :param threshold:
    :type threshold:

    :param iterations:
    :type iterations:

    :returns: tuple (fixed_array, bad_pixels_total)
        WHERE
        [type] fixed_array is...
        [type] bad_pixels_total is...
    """

    fixed_array = np.copy(signal_array)

    for i in range(iterations):

        filtered_array = ndimage.median_filter(fixed_array, size=3)
        diff = np.abs(fixed_array - filtered_array)
        limit = threshold * np.std(fixed_array)

        bad_pixel_list = np.nonzero(diff > limit)
        if i == 0:
            bad_pixels_total = np.vstack((bad_pixel_list[0], bad_pixel_list[1]))
        else:
            bad_pixels_total = np.hstack((bad_pixels_total, bad_pixel_list))

        fixed_array = remove_bad_pixels(fixed_array, filtered_array, bad_pixel_list)

    return fixed_array, bad_pixels_total


def remove_bad_pixels(signal_array, filtered_array, bad_pixel_list):
    """
    Removes bad pixels from the array

    :param signal_array:
    :type signal_array:

    :param filtered_array:
    :type filtered_array:

    :param bad_pixel_list:
    :type bad_pixel_list:

    :returns:
    :rtype:

    """

    fixed_array = np.copy(signal_array)

    for y, x in zip(bad_pixel_list[0], bad_pixel_list[1]):
        fixed_array[y, x] = filtered_array[y, x]

    return fixed_array


def fix_array(path, threshold=10, israte=False):
    """
    Wrapper function to find and remove 'hot' pixels.
    This version uses a path to specify a file

    :param path:
    :type path: str

    :param threshold:
    :type threshold:

    :param israte:
    :type israte: bool

    :returns: tuple (fixed_array, bad_pixel_list)
        WHERE
        [type] fixed_array is...
        [type] bad_pixel_list is...
    """
    if type(path) is str:
        signal_array = load_csv(path)
    else:
        signal_array = path

    if not israte:
        signal_array = 1 / signal_array

    filtered_array, bad_pixel_list = find_bad_pixels(signal_array, threshold)
    fixed_array = remove_bad_pixels(signal_array, filtered_array, bad_pixel_list)

    if not israte:
        fixed_array = 1 / fixed_array

    return fixed_array, bad_pixel_list
