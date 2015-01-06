#!/usr/bin/env python

"""load.py: Includes routines for loading data and configuration files."""

__author__ = "Durmus U. Karatay"
__copyright__ = "Copyright 2014, Ginger Lab"
__maintainer__ = "Durmus U. Karatay"
__email__ = "ukaratay@uw.edu"
__status__ = "Production"


import ConfigParser
import sys
from igor.binarywave import load as loadibw
from numpy.lib.npyio import loadtxt
from os.path import splitext


def signal(path, skiprows=1):
    """Load .ibw or ASCII files and return it as a numpy.ndarray.

    Parameters
    ----------
    path : string
        Path to signal file.

    Returns
    -------
    signal_array : array, shape = [n_points, n_signals]
        2D real-valued signal array loaded from given .ibw file.

    """

    ext = splitext(path)[1]

    if ext.lower() == '.ibw':

        signal_array = loadibw(path)['wave']['wData']  # Load data.

    elif ext.lower() == '.txt':

        signal_array = loadtxt(path, skiprows=skiprows)

    else:

        print "Unrecognized file type!"
        sys.exit(0)

    signal_array.flags.writeable = True  # Make array writable.

    return signal_array


def configuration(path):
    """Read an ASCII file with relevant parameters for processing.

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

        trigger = float (in seconds)
        total_time = float (in seconds)
        sampling_rate = int (in Hz)
        drive_freq = float (in Hz)

        roi = float (in seconds)
        window = string (see documentation of pixel.apply_window)
        bandpass_filter = int (0: no filtering, 1: FIR filter, 2: IIR filter)
        filter_bandwidth = float (in Hz)
        n_taps = integer (default: 999)

    """

    # Create a parser for configuration file and parse it.
    config = ConfigParser.RawConfigParser()
    config.read(path)
    parameters = {}

    # Assign parameters from file. These are the keys for parameters.
    paraf_keys = ['trigger', 'total_time', 'drive_freq', 'sampling_rate']
    procs_keys = ['window']
    procf_keys = ['roi', 'bandpass_filter', 'n_taps', 'filter_bandwidth']

    if config.has_option('Parameters', 'n_pixels'):

        n_pixels = config.getint('Parameters', 'n_pixels')

    else:

        n_pixels = int(1)

    for key in paraf_keys:

        if config.has_option('Parameters', key):

            parameters[key] = config.getfloat('Parameters', key)

    for key in procf_keys:

        if config.has_option('Processing', key):

            parameters[key] = config.getfloat('Processing', key)

    for key in procs_keys:

        if config.has_option('Processing', key):

            parameters[key] = config.get('Processing', key)

    return n_pixels, parameters
