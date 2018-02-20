"""load.py: Includes routines for loading data and configuration files."""

__author__ = "Durmus U. Karatay"
__copyright__ = "Copyright 2014, Ginger Lab"
__maintainer__ = "Durmus U. Karatay"
__email__ = "ukaratay@uw.edu"
__status__ = "Production"

import configparser
import sys
from igor.binarywave import load as loadibw
from numpy.lib.npyio import loadtxt
from os.path import splitext


def signal(path, skiprows=1):
    """
    Loads .ibw or ASCII files and return it as a numpy.ndarray.

    Parameters
    ----------
    path : string
        Path to signal file.

    Returns
    -------
    signal_array : (n_points, n_signals) array_like
        2D real-valued signal array loaded from given .ibw file.

    """

    # Get the path and check what the extension is.
    ext = splitext(path)[1]

    if ext.lower() == '.ibw':

        signal_array = loadibw(path)['wave']['wData']  # Load data.

    elif ext.lower() == '.txt':

        signal_array = loadtxt(path, skiprows=skiprows)

    else:

        print ("Unrecognized file type!")
        sys.exit(0)

    signal_array.flags.writeable = True  # Make array writable.

    return signal_array


def configuration(path):
    """
    Reads an ASCII file with relevant parameters for processing.

    Parameters
    ----------
    path : string
        Path to ASCII file.

    Returns
    -------
    n_pixels: int
        Number of pixels in the image.
    parameters : dict
        The list of parameters is:

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
    paraf_keys = ['trigger', 'total_time', 'drive_freq', 'sampling_rate', 'Q']
    procs_keys = ['window', 'fit_form']
    procf_keys = ['roi', 'width', 'height']
    proci_keys = ['n_taps', 'filter_bandwidth', 'bandpass_filter', 
                  'wavelet_analysis', 'wavelet_parameter', 'recombination',
                  'phase_fitting', 'EMD_analysis']

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


def simulation_configuration(path):
    """
    Reads an ASCII file with relevant parameters for simulation.

    Parameters
    ----------
    path : string
        Path to ASCII file.

    Returns
    -------
    can_params : dict
        Parameters for cantilever properties. The dictionary contains:

        amp_invols = float (in m/V)
        def_invols = float (in m/V)
        soft_amp = float (in V)
        drive_freq = float (in Hz)
        res_freq = float (in Hz)
        k = float (in N/m)
        q_factor = float

    force_params : dict
        Parameters for forces. The dictionary contains:

        es_force = float (in N)
        delta_freq = float (in Hz)
        tau = float (in seconds)

    sim_params : dict
        Parameters for simulation. The dictionary contains:

        trigger = float (in seconds)
        total_time = float (in seconds)
        sampling_rate = int (in Hz)

    """

    # Create a parser for configuration file and parse it.
    config = configparser.RawConfigParser()
    config.read(path)

    sim_params = {}
    can_params = {}
    force_params = {}

    # Assign parameters from file. These are the keys for parameters.
    can_keys = ['amp_invols', 'def_invols', 'soft_amp', 'drive_freq',
                'res_freq', 'k', 'q_factor']
    force_keys = ['es_force', 'delta_freq', 'tau']
    sim_keys = ['trigger', 'total_time', 'sampling_rate']

    for key in can_keys:

        if config.has_option('Cantilever Parameters', key):

            can_params[key] = config.getfloat('Cantilever Parameters', key)

    for key in force_keys:

        if config.has_option('Force Parameters', key):

            force_params[key] = config.getfloat('Force Parameters', key)

    for key in sim_keys:

        if config.has_option('Simulation Parameters', key):

            sim_params[key] = config.getfloat('Simulation Parameters', key)

    return can_params, force_params, sim_params
