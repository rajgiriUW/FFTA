"""load.py: Includes routines for loading data and configuration files."""

__author__ = "Rajiv Giridharagopal"
__copyright__ = "Copyright 2018, Ginger Lab"
__maintainer__ = "Rajiv Giridharagopal"
__email__ = "rgiri@uw.edu"
__status__ = "Development"

import configparser
import sys
from os.path import splitext

import numpy as np
import pandas as pd
from igor2.binarywave import load as loadibw
from numpy.lib.npyio import loadtxt


def signal(path, skiprows=0):
    """
    Loads .ibw or ASCII files and return it as a numpy.ndarray.

    :param path: Path to signal file.
    :type path: string
    
    :param skiprows:
    :type skiprows: int, optional

    :returns: 2D real-valued signal array loaded from given .ibw file.
    :rtype: (n_points, n_signals) array_like
        
    """

    # Get the path and check what the extension is.
    ext = splitext(path)[1]

    if ext.lower() == '.ibw':

        signal_array = loadibw(path)['wave']['wData']  # Load data.

    elif ext.lower() == '.txt':

        signal_array = loadtxt(path, skiprows=skiprows)

    else:

        print("Unrecognized file type!")
        sys.exit(0)

    try:
        signal_array.flags.writeable = True  # Make array writable.
    except:
        pass

    return signal_array


def configuration(path):
    """
    Reads an ASCII file with relevant parameters for processing.
    
    :param path: Path to ASCII file 
    :type path: string
    
    :returns: tuple (n_pixels, parameters)
        WHERE
        int n_pixels is number of pixels in the image.
        dict parameters is the list of parameters:  
            trigger = float (in seconds)
            total_time = float (in seconds)
            sampling_rate = int (in Hz)
            drive_freq = float (in Hz)
            Q = float (default: 500)
            
            roi = float (in seconds)
            window = string (see documentation of scipy.signal.get_window)
            bandpass_filter = int (0: no filtering, 1: FIR filter, 2: IIR filter)
            filter_bandwidth = float (in Hz)
            n_taps = integer (default: 999)
            wavelet_analysis = bool (0: Hilbert method, 1: Wavelet Method)
            wavelet_parameter = int (default: 5)
            recombination = bool (0: FF-trEFMm, 1: Recombination)
            phase_fitting = bool (0: frequency fitting, 1: phase fitting)
            EMD_analysis = bool (0: Hilbert method, 1: Hilbert-Huang fitting)
            
            fit_form = string (EXP, PRODUCT, SUM for type of fit function)

    """

    # Create a parser for configuration file and parse it.
    config = configparser.RawConfigParser()
    config.read(path)
    parameters = {}

    # These are the keys for parameters.
    paraf_keys = ['trigger', 'total_time', 'drive_freq',
                  'sampling_rate', 'Q']
    parai_keys = ['n_pixels', 'pts_per_pixel', 'lines_per_image']
    procs_keys = ['window', 'fit_form', 'method']
    procf_keys = ['roi', 'FastScanSize', 'SlowScanSize', 'lift_height', 'fft_time_res']
    proci_keys = ['n_taps', 'filter_bandwidth', 'bandpass_filter',
                  'wavelet_analysis', 'wavelet_parameter', 'recombination',
                  'phase_fitting']

    # Check if the configuration file has n_pixel,
    # if not assume single pixel.
    if config.has_option('Parameters', 'n_pixels'):

        n_pixels = config.getint('Parameters', 'n_pixels')

    else:

        n_pixels = int(1)

    # Loop through the parameters and assign them.
    for key in paraf_keys:

        if config.has_option('Parameters', key):
            parameters[key] = config.getfloat('Parameters', key)

    for key in parai_keys:

        if config.has_option('Parameters', key):
            parameters[key] = config.getint('Parameters', key)

    for key in procf_keys:

        if config.has_option('Processing', key):
            parameters[key] = config.getfloat('Processing', key)

    for key in procs_keys:

        if config.has_option('Processing', key):
            parameters[key] = config.get('Processing', key)

    for key in proci_keys:

        if config.has_option('Processing', key):
            parameters[key] = config.getint('Processing', key)

    return n_pixels, parameters


def cantilever_params(path, asDataFrame=False):
    '''
    Reads an experimental Parameters file describing the cantilever.
    Cantilever parameters should contain an Initial, Final, and Differential\
        column for describing an excited cantilever
    
    :param path: Path to the parameters file
    :type path: str
        
    :param asDataFrame: Returns Pandas dataframe instead of a dictionary
    :type asDataFrame: bool
        
    :returns:
    :rtype: dict
    
    '''

    try:
        df = pd.read_csv(path, sep='\t', skiprows=1, index_col='Unnamed: 0')
    except:
        df = pd.read_csv(path, sep='\t', skiprows=1)
    for c in df.columns:
        for n, cc in enumerate(df[c]):  # to get around numpy scalar-string FutureWarning
            if cc != 'NAN':
                df[c].iloc[n] = float(cc)
            else:
                df[c].iloc[n] = np.NaN
    #     df[c][df[c] != 'NAN'] = pd.to_numeric(df[c][df[c] != 'NAN'])
    #     df[c][df[c] == 'NAN'] = np.NaN

    if asDataFrame:
        return df

    return df.to_dict()
