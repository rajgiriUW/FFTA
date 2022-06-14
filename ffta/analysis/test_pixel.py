# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 13:17:12 2018

@author: Raj
"""

from matplotlib import pyplot as plt

from ffta.hdf_utils import hdf_utils


def test_pixel(h5_file, param_changes={}, pxls=1, showplots=True,
               verbose=True, clear_filter=False):
    """
    Takes a random pixel and does standard processing.

    :param h5_file: H5 file to process
    :type h5_file: h5Py File, path, Dataset

    :param param_changes:
    :type param_changes: dict, optional

    :param pxls: Number of random pixels to survey
    :type pxls: int, optional

    :param showplots: Whether to create a new plot or not.
    :type showplots: bool, optional

    :param verbose: To print to command line. Currently for future-proofing
    :type verbose : bool , optional

    :param clear_filter: Whether to do filtering (FIR) or not
    :type clear_filter: bool, optional


    """
    # get_pixel can work on Datasets or the H5_File
    if any(param_changes):
        hdf_utils.change_params(h5_file, new_vals=param_changes)

    parameters = hdf_utils.get_params(h5_file)
    cols = parameters['num_cols']
    rows = parameters['num_rows']

    # Creates random pixels to sample
    pixels = []
    if pxls == 1:
        pixels.append([0, 0])

    if pxls > 1:

        from numpy.random import randint
        for i in range(pxls):
            pixels.append([randint(0, rows), randint(0, cols)])

    # Analyzes all pixels
    for rc in pixels:

        h5_px = hdf_utils.get_pixel(h5_file, rc=rc)

        if clear_filter:
            h5_px.clear_filter_flags()

        h5_px.analyze()
        print(rc, h5_px.tfp)

        if showplots == True:
            plt.plot(h5_px.best_fit, 'r--')
            plt.plot(h5_px.cut, 'g-')

    return
