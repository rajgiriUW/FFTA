#!/usr/bin/env python

"""load.py: Includes routines for loading data and configuration files."""

__author__ = "Durmus U. Karatay"
__copyright__ = "Copyright 2014, Ginger Lab"
__maintainer__ = "Durmus U. Karatay"
__email__ = "ukaratay@uw.edu"
__status__ = "Production"


import ConfigParser
from igor.binarywave import load as loadibw


def ibw(path):
    """
    Load .ibw files and return it as a numpy.ndarray.

    Parameters
    ----------
    path : string
        Path to .ibw file.

    Returns
    -------
    signal_array : array, [n_points, n_signals]
        2D real-valued signal array loaded from given .ibw file.

    """

    signal_array = loadibw(path)['wave']['wData']  # Load data.
    signal_array.flags.writeable = True  # Make array writable.

    return signal_array


def configuration(path):
    """
    Read an ASCII file with relevant parameters for processing.

    Parameters
    ----------
    path : string
        Path to ASCII file.

    Returns
    -------
    n_pixels: int
        Number of pixels in the image.

    parameters : dict
        Parameters for pixel processing. The dictionary contains:

        trigger = float
        total_time = float
        sampling_rate = int
        drive_freq = float
        bandwidth = float
        smooth_time = float
        window_size = float
        noise_reduction = boolean
        bandpass_filter = boolean
        window = boolean
        smooth = boolean

    """

    # Create a parser for configuration file and parse it.
    config = ConfigParser.RawConfigParser()
    config.read(path)
    parameters = {}

    # Assign parameters from file. These are the keys for parameters.
    paraf_keys = ['trigger', 'total_time', 'drive_freq']
    parai_keys = ['sampling_rate']
    procf_keys = ['bandwidth', 'smooth_time', 'window_size']
    procb_keys = ['noise_reduction', 'bandpass_filter', 'window', 'smooth']

    if config.has_option('Parameters', 'n_pixels'):

        n_pixels = config.getint('Parameters', 'n_pixels')

    for key in paraf_keys:

        if config.has_option('Parameters', key):

            parameters[key] = config.getfloat('Parameters', key)

    for key in parai_keys:

        if config.has_option('Parameters', key):

            parameters[key] = config.getint('Parameters', key)

    for key in procf_keys:

        if config.has_option('Processing', key):

            parameters[key] = config.getfloat('Processing', key)

    for key in procb_keys:

        if config.has_option('Processing', key):

            parameters[key] = config.getboolean('Processing', key)

    return n_pixels, parameters
